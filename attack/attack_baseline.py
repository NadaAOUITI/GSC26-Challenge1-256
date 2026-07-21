"""Attack baseline: harmless FedAvg copy, or real backdoor fine-tune."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import SmallCNN
from utilities.checks import (
    ATTACK_BENIGN_MODELS_PER_CASE,
    MALICIOUS_MODELS_PER_CASE,
    fedavg,
)
from utilities.model_io import (
    load_state_dict_directory,
    save_state_dict,
)

from attack.celeba_data import (
    TARGET_CLASS,
    celeba_ready,
    make_eyeglasses_dataset,
    make_hair_dataset,
)
from attack.download_celeba import download_celeba, download_eyeglasses_subset
from attack.triggers import (
    TriggerGeometry,
    apply_trigger_tensor,
    dba_part_for_malicious,
    set_trigger_geometry,
    trigger_for_case,
)


def create_copy_baseline(output_root: Path, case_numbers: list[int]) -> None:
    """Original non-competitive baseline: save FedAvg(benign) as each malicious model."""
    output_root = Path(output_root)

    for case_number in case_numbers:
        benign_count = ATTACK_BENIGN_MODELS_PER_CASE[case_number]
        benign_models = load_state_dict_directory(
            ROOT / "attack" / f"case_{case_number}",
            expected_count=benign_count,
        )
        reference = fedavg(benign_models)

        for malicious_index in range(MALICIOUS_MODELS_PER_CASE[case_number]):
            destination = (
                output_root / f"case_{case_number}" / f"malicious_{malicious_index}.pt"
            )
            save_state_dict(reference, destination)

    print(f"Copy baseline malicious models written to: {output_root}")


def load_reference_model(case_number: int, device: torch.device) -> tuple[SmallCNN, dict]:
    benign_count = ATTACK_BENIGN_MODELS_PER_CASE[case_number]
    benign_models = load_state_dict_directory(
        ROOT / "attack" / f"case_{case_number}",
        expected_count=benign_count,
    )
    reference_state = fedavg(benign_models)
    model = SmallCNN(num_classes=4)
    model.load_state_dict(reference_state)
    model.to(device)
    return model, reference_state


def regularization_loss(model: SmallCNN, reference_state: dict) -> torch.Tensor:
    """Mean squared distance from the frozen benign reference weights."""
    sq_sum = None
    count = 0
    for name, parameter in model.named_parameters():
        ref = reference_state[name].to(parameter.device, dtype=parameter.dtype)
        diff = parameter - ref
        term = diff.pow(2).sum()
        sq_sum = term if sq_sum is None else sq_sum + term
        count += diff.numel()
    return sq_sum / float(max(count, 1))


def parameter_delta_l2(model: SmallCNN, reference_state: dict) -> torch.Tensor:
    """Differentiable ||θ − θ_ref||_2 over all parameters (eps-stable)."""
    sq_sum = None
    for name, parameter in model.named_parameters():
        ref = reference_state[name].to(parameter.device, dtype=parameter.dtype)
        term = (parameter - ref).pow(2).sum()
        sq_sum = term if sq_sum is None else sq_sum + term
    return (sq_sum + 1e-12).sqrt()


def constrain_norm_loss(
    model: SmallCNN,
    reference_state: dict,
    target_l2: float,
) -> torch.Tensor:
    """
    Soft pull of ||θ-θ_ref||^2 toward target_l2^2 (avoids sqrt(0) NaNs).
    """
    sq_sum = None
    for name, parameter in model.named_parameters():
        ref = reference_state[name].to(parameter.device, dtype=parameter.dtype)
        term = (parameter - ref).pow(2).sum()
        sq_sum = term if sq_sum is None else sq_sum + term
    target_sq = float(target_l2) ** 2
    return ((sq_sum / max(target_sq, 1e-12)) - 1.0).pow(2)


def run_sanity_check(
    model: SmallCNN,
    images: torch.Tensor,
    labels: torch.Tensor,
    trigger: str,
    device: torch.device,
) -> None:
    model.eval()
    with torch.no_grad():
        clean_logits = model(images.to(device))
        clean_pred = clean_logits.argmax(dim=1).cpu()
        clean_acc = (clean_pred == labels).float().mean().item()

        triggered = torch.stack(
            [apply_trigger_tensor(image, trigger) for image in images]
        ).to(device)
        triggered_logits = model(triggered)
        triggered_pred = triggered_logits.argmax(dim=1).cpu()
        asr = (triggered_pred == TARGET_CLASS).float().mean().item()

    print(
        f"  sanity: clean_acc={clean_acc:.3f}  "
        f"triggered_asr(proxy)={asr:.3f}  "
        f"(trigger={trigger})"
    )
    model.train()


def build_backdoor_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    triggered: torch.Tensor,
    poison_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select / upsample images for the backdoor loss.

    all: every image with trigger → black (legacy).
    nonblack_boost: only non-black faces, upsampled to full batch size so
      the BD gradient is pure 'trigger flips non-black → black'.
    """
    batch_size = images.size(0)
    device = images.device
    target = torch.full(
        (batch_size,),
        TARGET_CLASS,
        dtype=torch.long,
        device=device,
    )

    if poison_mode == "all":
        return triggered, target

    if poison_mode != "nonblack_boost":
        raise ValueError(f"Unknown poison_mode: {poison_mode!r}")

    nonblack = labels != TARGET_CLASS
    if not bool(nonblack.any()):
        return triggered, target

    pool = triggered[nonblack]
    # Upsample non-black triggered examples to full batch size.
    idx = torch.randint(0, pool.size(0), (batch_size,), device=device)
    return pool[idx], target


def set_train_scope(model: SmallCNN, train_scope: str) -> list[nn.Parameter]:
    """
    Freeze early layers; return trainable parameters.

    full: all params (legacy / winning keep)
    head: only Linear classifier
    last_block: last conv block + classifier
    """
    for parameter in model.parameters():
        parameter.requires_grad = True

    if train_scope == "full":
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  train_scope=full  trainable_tensors={len(trainable)}")
        return trainable

    # Freeze entire feature trunk by default for scoped modes.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    if train_scope == "head":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif train_scope == "last_block":
        # Unfreeze last Conv2d (index 6) + classifier. Leave early convs frozen.
        for index, module in enumerate(model.features):
            if index >= 6 and isinstance(module, nn.Conv2d):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unknown train_scope: {train_scope!r}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(
        f"  train_scope={train_scope}  "
        f"trainable_params={n_train}/{n_total}"
    )
    if not trainable:
        raise RuntimeError(f"No trainable parameters for scope={train_scope}")
    return trainable


def build_neurotoxin_masks(
    reference_state: dict,
    benign_models: list,
    keep_ratio: float = 0.1,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """
    Quiet-parameter masks (Neurotoxin-style).

    For each coordinate, score = median_i |theta_benign_i - theta_ref|.
    Keep the bottom keep_ratio fraction globally (least-changed by benigns)
    so backdoor gradients land where aggregation is less likely to overwrite.
    """
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    device = device or torch.device("cpu")

    scores: dict[str, torch.Tensor] = {}
    flat_scores = []
    for name, ref in reference_state.items():
        ref_f = ref.detach().to(device=device, dtype=torch.float64)
        stacked = torch.stack(
            [
                benign[name]
                .detach()
                .to(device=device, dtype=torch.float64)
                .sub(ref_f)
                .abs()
                for benign in benign_models
            ],
            dim=0,
        )
        score = stacked.median(dim=0).values
        scores[name] = score
        flat_scores.append(score.reshape(-1))

    all_scores = torch.cat(flat_scores, dim=0)
    k = max(1, int(all_scores.numel() * keep_ratio))
    threshold = torch.kthvalue(all_scores, k).values

    masks = {}
    kept = 0
    total = 0
    for name, score in scores.items():
        mask = score <= threshold
        # If ties at threshold flood the mask, trim randomly is unnecessary;
        # approximate keep_ratio is fine.
        masks[name] = mask.to(device=device)
        kept += int(mask.sum().item())
        total += mask.numel()
    print(
        f"  neurotoxin mask: keep_ratio={keep_ratio:.3f} "
        f"kept={kept}/{total} ({kept / max(total, 1):.3f}) "
        f"threshold={float(threshold):.6e}"
    )
    return masks


def fine_tune_backdoor(
    model: SmallCNN,
    reference_state: dict,
    loader: DataLoader,
    trigger: str,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_bd: float,
    lambda_reg: float,
    poison_mode: str = "all",
    train_scope: str = "full",
    glasses_loader: DataLoader | None = None,
    trigger_part: str | None = None,
    neurotoxin_masks: dict[str, torch.Tensor] | None = None,
    constrain_target: float | None = None,
    lambda_constrain: float = 0.0,
    trigger_jitter: bool = False,
    multi_trigger: bool = False,
) -> None:
    trainable = set_train_scope(model, train_scope)
    optimizer = torch.optim.Adam(trainable, lr=lr)
    model.train()
    if train_scope != "full":
        model.features.eval()
    print(f"  poison_mode={poison_mode}")
    if trigger_part is not None:
        print(f"  dba_trigger_part={trigger_part}")
    if trigger_jitter:
        print("  trigger_jitter=on (per-image geometry noise)")
    if multi_trigger:
        print("  multi_trigger=on (discrete preset mix)")
    if neurotoxin_masks is not None:
        print("  neurotoxin=on (BD grads masked to quiet coords)")
    if constrain_target is not None and lambda_constrain > 0:
        print(
            f"  constrain=on  target_l2={constrain_target:.4f}  "
            f"lambda_constrain={lambda_constrain:.3f}"
        )
    glasses_iter = iter(glasses_loader) if glasses_loader is not None else None
    if glasses_loader is not None:
        print(f"  real_eyeglasses_mix=on  pool={len(glasses_loader.dataset)}")

    for epoch in range(1, epochs + 1):
        running = {
            "clean": 0.0,
            "backdoor": 0.0,
            "reg": 0.0,
            "constrain": 0.0,
            "total": 0.0,
        }
        batches = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            triggered = torch.stack(
                [
                    apply_trigger_tensor(
                        image.cpu(),
                        trigger,
                        part=trigger_part,
                        jitter=trigger_jitter and trigger_part is None,
                        multi_trigger=multi_trigger and trigger_part is None,
                    )
                    for image in images
                ]
            ).to(device)
            bd_images, bd_labels = build_backdoor_batch(
                images, labels, triggered, poison_mode=poison_mode
            )

            if glasses_iter is not None:
                try:
                    g_images, _g_labels = next(glasses_iter)
                except StopIteration:
                    glasses_iter = iter(glasses_loader)
                    g_images, _g_labels = next(glasses_iter)
                g_images = g_images.to(device)
                keep = _g_labels.to(device) != TARGET_CLASS
                if bool(keep.any()):
                    g_pool = g_images[keep]
                else:
                    g_pool = g_images
                n_half = max(1, bd_images.size(0) // 2)
                idx = torch.randint(0, g_pool.size(0), (n_half,), device=device)
                real = g_pool[idx]
                synth = bd_images[: bd_images.size(0) - n_half]
                bd_images = torch.cat([synth, real], dim=0)
                bd_labels = torch.full(
                    (bd_images.size(0),),
                    TARGET_CLASS,
                    dtype=torch.long,
                    device=device,
                )

            clean_logits = model(images)
            backdoor_logits = model(bd_images)
            loss_clean = F.cross_entropy(clean_logits, labels)
            loss_backdoor = F.cross_entropy(backdoor_logits, bd_labels)
            loss_reg = regularization_loss(model, reference_state)

            loss_constrain = torch.zeros((), device=device)
            if constrain_target is not None and lambda_constrain > 0:
                loss_constrain = constrain_norm_loss(
                    model, reference_state, float(constrain_target)
                )

            if neurotoxin_masks is None:
                optimizer.zero_grad(set_to_none=True)
                loss = (
                    loss_clean
                    + lambda_bd * loss_backdoor
                    + lambda_reg * loss_reg
                    + lambda_constrain * loss_constrain
                )
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                (
                    loss_clean
                    + lambda_reg * loss_reg
                    + lambda_constrain * loss_constrain
                ).backward(retain_graph=True)
                clean_grads = {
                    name: (
                        None
                        if parameter.grad is None
                        else parameter.grad.detach().clone()
                    )
                    for name, parameter in model.named_parameters()
                }

                optimizer.zero_grad(set_to_none=True)
                (lambda_bd * loss_backdoor).backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        continue
                    mask = neurotoxin_masks[name].to(device=parameter.device)
                    parameter.grad.mul_(mask.to(dtype=parameter.grad.dtype))
                    prior = clean_grads.get(name)
                    if prior is not None:
                        parameter.grad.add_(prior)
                optimizer.step()

            total_loss = (
                loss_clean
                + lambda_bd * loss_backdoor
                + lambda_reg * loss_reg
                + lambda_constrain * loss_constrain
            )
            running["clean"] += float(loss_clean.item())
            running["backdoor"] += float(loss_backdoor.item())
            running["reg"] += float(loss_reg.item())
            running["constrain"] += float(loss_constrain.item())
            running["total"] += float(total_loss.item())
            batches += 1

        scale = max(batches, 1)
        print(
            f"  epoch {epoch}/{epochs}  "
            f"L={running['total'] / scale:.4f}  "
            f"L_clean={running['clean'] / scale:.4f}  "
            f"L_bd={running['backdoor'] / scale:.4f}  "
            f"L_reg={running['reg'] / scale:.4f}  "
            f"L_con={running['constrain'] / scale:.4f}"
        )

        run_sanity_check(model, images.cpu(), labels.cpu(), trigger, device)


def default_scale_factor(case_number: int) -> float:
    """FedAvg model-replacement factor: total_clients / malicious_clients."""
    benign = ATTACK_BENIGN_MODELS_PER_CASE[case_number]
    malicious = MALICIOUS_MODELS_PER_CASE[case_number]
    total = benign + malicious
    return float(total) / float(malicious)


def scale_state_dict(
    poisoned_state: dict,
    reference_state: dict,
    gamma: float,
) -> dict:
    """
    Amplify the backdoor delta for aggregation dilution.

    theta_submit = theta_ref + gamma * (theta_mal - theta_ref)

    With gamma = n/m and FedAvg, identical malicious submits push the
    aggregate toward theta_mal when benign clients sit near theta_ref.
    """
    scaled = {}
    for key, poisoned in poisoned_state.items():
        ref = reference_state[key].detach().cpu().to(dtype=poisoned.dtype)
        delta = poisoned.detach().cpu() - ref
        scaled[key] = (ref + gamma * delta).contiguous()
    return scaled


def state_delta_l2(state: dict, reference_state: dict) -> float:
    total = None
    for key, tensor in state.items():
        ref = reference_state[key].detach().cpu().to(dtype=torch.float64)
        delta = tensor.detach().cpu().to(dtype=torch.float64) - ref
        term = delta.pow(2).sum()
        total = term if total is None else total + term
    return float(total.sqrt().item())


def median_benign_delta_l2(
    benign_models: list,
    reference_state: dict,
) -> float:
    """Median ||θ_b − θ_ref||_2 across provided benign clients."""
    norms = sorted(
        state_delta_l2(benign, reference_state) for benign in benign_models
    )
    return float(norms[len(norms) // 2])


def train_and_scale_state(
    poisoned_state: dict,
    reference_state: dict,
    benign_models: list,
) -> dict:
    """
    Bagdasaryan train-and-scale: match malicious update norm to the
    median benign ||theta_b - theta_ref|| before model-replacement scaling.
    """
    benign_norms = sorted(
        state_delta_l2(benign, reference_state) for benign in benign_models
    )
    target_norm = benign_norms[len(benign_norms) // 2]
    current_norm = state_delta_l2(poisoned_state, reference_state)
    if current_norm <= 1e-12:
        return {
            key: value.detach().cpu().contiguous()
            for key, value in poisoned_state.items()
        }
    align_gamma = target_norm / current_norm
    print(
        f"  train-and-scale: delta_l2={current_norm:.4f} -> "
        f"target={target_norm:.4f} (align_gamma={align_gamma:.4f})"
    )
    return scale_state_dict(poisoned_state, reference_state, gamma=align_gamma)


def blend_benign_direction(
    poisoned_state: dict,
    reference_state: dict,
    benign_state: dict,
    mix: float,
) -> dict:
    """
    Pull each malicious submit toward a different benign client's update.

    theta = theta_mal + mix * (theta_benign - theta_ref)

    Keeps the backdoor delta while spreading malicious models in parameter
    space (harder for distance-based robust aggregators to reject as a clique).
    """
    if mix == 0.0:
        return {
            key: value.detach().cpu().contiguous()
            for key, value in poisoned_state.items()
        }
    blended = {}
    for key, poisoned in poisoned_state.items():
        ref = reference_state[key].detach().cpu().to(dtype=poisoned.dtype)
        benign = benign_state[key].detach().cpu().to(dtype=poisoned.dtype)
        blended[key] = (poisoned + mix * (benign - ref)).contiguous()
    return blended


def create_backdoor_attack(
    output_root: Path,
    data_root: Path,
    case_numbers: list[int],
    max_images: int,
    image_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    lambda_bd: float,
    lambda_reg: float,
    download: bool,
    scale_factor: float | None,
    train_and_scale: bool,
    poison_mode: str = "all",
    train_scope: str = "full",
    use_real_glasses: bool = False,
    glasses_max_images: int = 400,
    dba: bool = False,
    diversify: bool = False,
    diversify_benign_mix: float = 0.25,
    trigger_jitter: bool = False,
    multi_trigger: bool = False,
    geometry: TriggerGeometry | None = None,
    neurotoxin: bool = False,
    neurotoxin_keep_ratio: float = 0.1,
    constrain: bool = False,
    lambda_constrain: float = 1.0,
) -> None:
    output_root = Path(output_root)
    data_root = Path(data_root)
    if geometry is not None:
        set_trigger_geometry(geometry)
        print(f"Trigger geometry: {geometry.to_dict()}")
    else:
        set_trigger_geometry(None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not celeba_ready(data_root):
        if download:
            download_celeba(data_root, max_images=max_images)
        else:
            raise FileNotFoundError(
                f"CelebA missing under {data_root}. "
                "Re-run with --download or: python attack/download_celeba.py"
            )

    print(f"Loading CelebA hair-color subset (max_images={max_images})...")
    dataset = make_hair_dataset(
        data_root=data_root,
        max_images=max_images,
        image_size=image_size,
    )
    print(f"Dataset size: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError(
            "DataLoader is empty. Increase --max-images or lower --batch-size."
        )

    glasses_loader: DataLoader | None = None
    if use_real_glasses:
        glasses_root_ready = (data_root / "celeba_eyeglasses" / "manifest.json").is_file()
        if not glasses_root_ready:
            print("Downloading CelebA eyeglasses subset...")
            download_eyeglasses_subset(data_root, max_images=glasses_max_images)
        glasses_ds = make_eyeglasses_dataset(
            data_root=data_root,
            max_images=glasses_max_images,
            image_size=image_size,
        )
        glasses_loader = DataLoader(
            glasses_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        print(f"Real eyeglasses pool size: {len(glasses_ds)}")

    for case_number in case_numbers:
        trigger = trigger_for_case(case_number)
        gamma = (
            default_scale_factor(case_number)
            if scale_factor is None
            else float(scale_factor)
        )
        malicious_count = MALICIOUS_MODELS_PER_CASE[case_number]
        print(
            f"\n=== Case {case_number} | trigger={trigger} | "
            f"scale_gamma={gamma:.3f} | poison_mode={poison_mode} | "
            f"train_scope={train_scope} | dba={dba} | "
            f"diversify={diversify} | benign_mix={diversify_benign_mix} | "
            f"trigger_jitter={trigger_jitter} | multi_trigger={multi_trigger} | "
            f"neurotoxin={neurotoxin} | constrain={constrain} | "
            f"train_and_scale={train_and_scale} ==="
        )
        benign_count = ATTACK_BENIGN_MODELS_PER_CASE[case_number]
        benign_models = load_state_dict_directory(
            ROOT / "attack" / f"case_{case_number}",
            expected_count=benign_count,
        )
        reference_state = fedavg(benign_models)
        case_glasses = glasses_loader if trigger == "sunglasses" else None
        neurotoxin_masks = None
        if neurotoxin:
            neurotoxin_masks = build_neurotoxin_masks(
                reference_state=reference_state,
                benign_models=benign_models,
                keep_ratio=neurotoxin_keep_ratio,
                device=device,
            )
        constrain_target = None
        if constrain:
            constrain_target = median_benign_delta_l2(
                benign_models, reference_state
            )
            print(
                f"  constrain target (median benign delta L2)={constrain_target:.4f}"
            )

        if dba and diversify:
            raise ValueError("Use either --dba or --diversify, not both.")

        if dba or diversify:
            # Per-malicious training: DBA uses sub-triggers; diversify uses
            # full trigger + distinct seeds + optional benign-direction blend.
            for malicious_index in range(malicious_count):
                part = None
                if dba:
                    part = dba_part_for_malicious(
                        case_number, malicious_index, malicious_count
                    )
                    print(
                        f"\n--- malicious_{malicious_index} | "
                        f"dba_part={part} ---"
                    )
                else:
                    seed = 1000 * case_number + 17 * malicious_index + 7
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    print(
                        f"\n--- malicious_{malicious_index} | "
                        f"diversify seed={seed} ---"
                    )
                model = SmallCNN(num_classes=4)
                model.load_state_dict(reference_state)
                model.to(device)
                fine_tune_backdoor(
                    model=model,
                    reference_state=reference_state,
                    loader=loader,
                    trigger=trigger,
                    device=device,
                    epochs=epochs,
                    lr=lr,
                    lambda_bd=lambda_bd,
                    lambda_reg=lambda_reg,
                    poison_mode=poison_mode,
                    train_scope=train_scope,
                    glasses_loader=None if dba else case_glasses,
                    trigger_part=part,
                    neurotoxin_masks=neurotoxin_masks,
                    constrain_target=constrain_target,
                    lambda_constrain=lambda_constrain if constrain else 0.0,
                    trigger_jitter=trigger_jitter and not dba,
                    multi_trigger=multi_trigger and not dba,
                )
                poisoned_state = {
                    key: value.detach().cpu().contiguous()
                    for key, value in model.state_dict().items()
                }
                if train_and_scale:
                    poisoned_state = train_and_scale_state(
                        poisoned_state,
                        reference_state,
                        benign_models,
                    )
                if gamma != 1.0:
                    poisoned_state = scale_state_dict(
                        poisoned_state,
                        reference_state,
                        gamma=gamma,
                    )
                    print(f"  applied model-replacement scaling gamma={gamma:.3f}")
                if diversify and diversify_benign_mix != 0.0:
                    benign_pick = benign_models[
                        malicious_index % len(benign_models)
                    ]
                    poisoned_state = blend_benign_direction(
                        poisoned_state,
                        reference_state,
                        benign_pick,
                        mix=diversify_benign_mix,
                    )
                    print(
                        f"  blended benign_{malicious_index % len(benign_models)} "
                        f"direction mix={diversify_benign_mix}"
                    )
                destination = (
                    output_root
                    / f"case_{case_number}"
                    / f"malicious_{malicious_index}.pt"
                )
                save_state_dict(poisoned_state, destination)
                print(f"  wrote {destination}")
            continue

        model = SmallCNN(num_classes=4)
        model.load_state_dict(reference_state)
        model.to(device)

        fine_tune_backdoor(
            model=model,
            reference_state=reference_state,
            loader=loader,
            trigger=trigger,
            device=device,
            epochs=epochs,
            lr=lr,
            lambda_bd=lambda_bd,
            lambda_reg=lambda_reg,
            poison_mode=poison_mode,
            train_scope=train_scope,
            glasses_loader=case_glasses,
            neurotoxin_masks=neurotoxin_masks,
            constrain_target=constrain_target,
            lambda_constrain=lambda_constrain if constrain else 0.0,
            trigger_jitter=trigger_jitter,
            multi_trigger=multi_trigger,
        )

        poisoned_state = {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
        }
        if train_and_scale:
            poisoned_state = train_and_scale_state(
                poisoned_state,
                reference_state,
                benign_models,
            )
        if gamma != 1.0:
            poisoned_state = scale_state_dict(
                poisoned_state,
                reference_state,
                gamma=gamma,
            )
            print(f"  applied model-replacement scaling gamma={gamma:.3f}")

        for malicious_index in range(malicious_count):
            state = copy.deepcopy(poisoned_state)
            if malicious_index > 0:
                with torch.no_grad():
                    noise_scale = 1e-5 * malicious_index
                    for tensor in state.values():
                        tensor.add_(noise_scale * torch.randn_like(tensor))

            destination = (
                output_root / f"case_{case_number}" / f"malicious_{malicious_index}.pt"
            )
            save_state_dict(state, destination)
            print(f"  wrote {destination}")

    print(f"\nBackdoor malicious models written to: {output_root}")


def parse_cases(raw: str) -> list[int]:
    cases = [int(part.strip()) for part in raw.split(",") if part.strip()]
    for case_number in cases:
        if case_number not in ATTACK_BENIGN_MODELS_PER_CASE:
            raise argparse.ArgumentTypeError(f"Invalid case: {case_number}")
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="GSC attack baseline (copy FedAvg or backdoor fine-tune)"
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "backdoor"),
        default="backdoor",
        help="copy = old harmless baseline; backdoor = fine-tune with triggers",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "participant_models",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
    )
    parser.add_argument(
        "--cases",
        type=parse_cases,
        default=parse_cases("1,2,3"),
        help="Comma-separated case numbers, e.g. 1,2,3",
    )
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-bd", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CelebA if missing (backdoor mode)",
    )
    parser.add_argument(
        "--scale-factor",
        type=str,
        default="auto",
        help=(
            "Model-replacement gamma after fine-tune. "
            "'auto' uses total_clients/malicious_clients per case; "
            "'1' disables scaling; or pass a float."
        ),
    )
    parser.add_argument(
        "--train-and-scale",
        action="store_true",
        help=(
            "Before gamma, align malicious delta L2 to the median "
            "benign ||theta_b - theta_ref|| (Bagdasaryan train-and-scale)."
        ),
    )
    parser.add_argument(
        "--poison-mode",
        choices=("all", "nonblack_boost"),
        default="all",
        help=(
            "Backdoor batch policy: 'all' = every triggered image → black; "
            "'nonblack_boost' = only non-black faces, upsampled to batch size."
        ),
    )
    parser.add_argument(
        "--train-scope",
        choices=("full", "head", "last_block"),
        default="full",
        help=(
            "Which layers to update: full model, classifier head only, "
            "or last conv block + head."
        ),
    )
    parser.add_argument(
        "--use-real-glasses",
        action="store_true",
        help=(
            "For sunglasses cases, mix real CelebA Eyeglasses faces into "
            "the backdoor batch (natural eyewear trigger)."
        ),
    )
    parser.add_argument(
        "--glasses-max-images",
        type=int,
        default=400,
        help="Max images in the celeba_eyeglasses pool",
    )
    parser.add_argument(
        "--dba",
        action="store_true",
        help=(
            "Distributed backdoor: train each malicious model on a "
            "different sub-trigger part (DBA). Sanity/portal use full trigger."
        ),
    )
    parser.add_argument(
        "--diversify",
        action="store_true",
        help=(
            "Train each malicious independently (distinct seeds) and blend "
            "a fraction of a different benign client's update direction."
        ),
    )
    parser.add_argument(
        "--diversify-benign-mix",
        type=float,
        default=0.25,
        help=(
            "With --diversify: weight on (theta_benign_i - theta_ref) added "
            "after scaling (default 0.25)."
        ),
    )
    parser.add_argument(
        "--trigger-jitter",
        action="store_true",
        help=(
            "During BD training, jitter sunglasses/mask geometry per image "
            "for better portal trigger transfer."
        ),
    )
    parser.add_argument(
        "--multi-trigger",
        action="store_true",
        help=(
            "Discrete multi-trigger mix: each BD image uses a random "
            "geometry preset (survey-style; stronger than continuous jitter)."
        ),
    )
    parser.add_argument(
        "--geometry-json",
        type=Path,
        default=None,
        help="Optional TriggerGeometry JSON from attack/trigger_search.py",
    )
    parser.add_argument(
        "--neurotoxin",
        action="store_true",
        help=(
            "Neurotoxin-style: apply backdoor gradients only on coordinates "
            "least changed by benign clients (quiet params)."
        ),
    )
    parser.add_argument(
        "--neurotoxin-keep-ratio",
        type=float,
        default=0.1,
        help="Fraction of quiet coordinates kept for backdoor grads (default 0.1)",
    )
    parser.add_argument(
        "--constrain",
        action="store_true",
        help=(
            "Constrain-and-scale: during training, soft-pull ||theta-theta_ref|| "
            "toward median benign delta L2, then still apply --scale-factor gamma."
        ),
    )
    parser.add_argument(
        "--lambda-constrain",
        type=float,
        default=0.1,
        help="Weight on constrain loss (only with --constrain)",
    )
    args = parser.parse_args()

    if args.mode == "copy":
        create_copy_baseline(args.output_root, args.cases)
        return

    if args.scale_factor.strip().lower() == "auto":
        scale_factor: float | None = None
    else:
        scale_factor = float(args.scale_factor)

    geometry = None
    if args.geometry_json is not None:
        payload = json.loads(Path(args.geometry_json).read_text(encoding="utf-8"))
        geometry = TriggerGeometry.from_dict(payload)

    create_backdoor_attack(
        output_root=args.output_root,
        data_root=args.data_root,
        case_numbers=args.cases,
        max_images=args.max_images,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_bd=args.lambda_bd,
        lambda_reg=args.lambda_reg,
        download=args.download,
        scale_factor=scale_factor,
        train_and_scale=args.train_and_scale,
        poison_mode=args.poison_mode,
        train_scope=args.train_scope,
        use_real_glasses=args.use_real_glasses,
        glasses_max_images=args.glasses_max_images,
        dba=args.dba,
        diversify=args.diversify,
        diversify_benign_mix=args.diversify_benign_mix,
        trigger_jitter=args.trigger_jitter,
        multi_trigger=args.multi_trigger,
        geometry=geometry,
        neurotoxin=args.neurotoxin,
        neurotoxin_keep_ratio=args.neurotoxin_keep_ratio,
        constrain=args.constrain,
        lambda_constrain=args.lambda_constrain,
    )


if __name__ == "__main__":
    main()
