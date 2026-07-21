"""Trigger geometry search scored by post-FedAvg ASR (local proxy).

Sweeps GEOMETRY_PRESETS with a quick case-scoped fine-tune, then writes
the winning TriggerGeometry JSON for a full keep-recipe retrain.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import SmallCNN
from utilities.aggregators import get_aggregator
from utilities.checks import (
    ATTACK_BENIGN_MODELS_PER_CASE,
    MALICIOUS_MODELS_PER_CASE,
    fedavg,
)
from utilities.model_io import load_state_dict_directory, save_state_dict

from attack.attack_baseline import fine_tune_backdoor, scale_state_dict
from attack.celeba_data import make_hair_dataset
from attack.eval_heldout_asr import evaluate_model
from attack.triggers import (
    GEOMETRY_PRESETS,
    TriggerGeometry,
    set_trigger_geometry,
    trigger_for_case,
)


def proxy_score(clean_acc: float, asr: float) -> float:
    return 0.4 * clean_acc + 0.6 * asr


def train_one_case(
    case_number: int,
    geometry: TriggerGeometry,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_bd: float,
    lambda_reg: float,
    gamma: float,
    scratch_root: Path,
) -> Path:
    set_trigger_geometry(geometry)
    trigger = trigger_for_case(case_number)
    benign_count = ATTACK_BENIGN_MODELS_PER_CASE[case_number]
    malicious_count = MALICIOUS_MODELS_PER_CASE[case_number]
    benign_models = load_state_dict_directory(
        ROOT / "attack" / f"case_{case_number}",
        expected_count=benign_count,
    )
    reference_state = fedavg(benign_models)
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
        poison_mode="all",
        train_scope="full",
        glasses_loader=None,
        trigger_part=None,
    )

    poisoned = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    if gamma != 1.0:
        poisoned = scale_state_dict(poisoned, reference_state, gamma=gamma)

    case_dir = scratch_root / f"case_{case_number}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    for malicious_index in range(malicious_count):
        destination = case_dir / f"malicious_{malicious_index}.pt"
        save_state_dict(poisoned, destination)
    return case_dir


def score_case_post_fedavg(
    case_number: int,
    models_root: Path,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    trigger = trigger_for_case(case_number)
    benign_count = ATTACK_BENIGN_MODELS_PER_CASE[case_number]
    malicious_count = MALICIOUS_MODELS_PER_CASE[case_number]
    benign_models = load_state_dict_directory(
        ROOT / "attack" / f"case_{case_number}",
        expected_count=benign_count,
    )
    malicious_models = []
    for index in range(malicious_count):
        path = models_root / f"case_{case_number}" / f"malicious_{index}.pt"
        malicious_models.append(
            torch.load(path, map_location="cpu", weights_only=True)
        )
    pool = list(benign_models) + malicious_models
    aggregated = get_aggregator("fedavg")(pool)
    model = SmallCNN(num_classes=4)
    model.load_state_dict(aggregated)
    model.to(device)
    clean_acc, asr, _ = evaluate_model(model, loader, trigger, device)
    return clean_acc, asr, proxy_score(clean_acc, asr)


def merge_geometry(
    sunglasses: TriggerGeometry,
    mask: TriggerGeometry,
) -> TriggerGeometry:
    """Combine best sunglasses eye params with best mask params."""
    return TriggerGeometry(
        eye_y=sunglasses.eye_y,
        lens_h=sunglasses.lens_h,
        lens_w=sunglasses.lens_w,
        gap=sunglasses.gap,
        mask_top=mask.mask_top,
        mask_bottom=mask.mask_bottom,
        mask_mid=mask.mask_mid,
        mask_top_half=mask.mask_top_half,
        mask_mid_half=mask.mask_mid_half,
        mask_bot_half=mask.mask_bot_half,
    )


def run_search(
    data_root: Path,
    scratch_root: Path,
    results_path: Path,
    winner_path: Path,
    max_images: int,
    epochs: int,
    gamma: float,
    holdout_start: int,
    holdout_count: int,
) -> TriggerGeometry:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    scratch_root.mkdir(parents=True, exist_ok=True)

    train_ds = make_hair_dataset(
        data_root=data_root,
        max_images=max_images,
        image_size=64,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    holdout_ds = make_hair_dataset(
        data_root=data_root,
        max_images=holdout_count,
        image_size=64,
        start_index=holdout_start,
    )
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    print(f"Train images={len(train_ds)} holdout={len(holdout_ds)}")

    sunglasses_presets = [
        "default",
        "eyes_low",
        "eyes_lower",
        "eyes_high",
        "lenses_big",
        "lenses_huge",
        "combo_low_big",
    ]
    mask_presets = [
        "default",
        "mask_high",
        "mask_low",
        "mask_wide",
        "combo_low_big",
    ]

    rows: list[dict] = []

    print("\n=== Sunglasses presets on Case 1 ===")
    best_sg_name = "default"
    best_sg_asr = -1.0
    best_sg_geo = GEOMETRY_PRESETS["default"]
    for name in sunglasses_presets:
        geo = GEOMETRY_PRESETS[name]
        print(f"\n-- preset={name} --")
        train_one_case(
            case_number=1,
            geometry=geo,
            loader=train_loader,
            device=device,
            epochs=epochs,
            lr=1e-4,
            lambda_bd=2.5,
            lambda_reg=10.0,
            gamma=gamma,
            scratch_root=scratch_root / name,
        )
        set_trigger_geometry(geo)
        clean, asr, score = score_case_post_fedavg(
            1, scratch_root / name, holdout_loader, device
        )
        row = {
            "family": "sunglasses",
            "preset": name,
            "case": 1,
            "clean_acc": clean,
            "asr": asr,
            "proxy": score,
            "geometry": geo.to_dict(),
        }
        rows.append(row)
        print(f"  post-FedAvg clean={clean:.3f} asr={asr:.3f} proxy={score:.3f}")
        if asr > best_sg_asr:
            best_sg_asr = asr
            best_sg_name = name
            best_sg_geo = geo

    print("\n=== Mask presets on Case 2 ===")
    best_mk_name = "default"
    best_mk_asr = -1.0
    best_mk_geo = GEOMETRY_PRESETS["default"]
    for name in mask_presets:
        geo = GEOMETRY_PRESETS[name]
        print(f"\n-- preset={name} --")
        train_one_case(
            case_number=2,
            geometry=geo,
            loader=train_loader,
            device=device,
            epochs=epochs,
            lr=1e-4,
            lambda_bd=2.5,
            lambda_reg=10.0,
            gamma=gamma,
            scratch_root=scratch_root / name,
        )
        set_trigger_geometry(geo)
        clean, asr, score = score_case_post_fedavg(
            2, scratch_root / name, holdout_loader, device
        )
        row = {
            "family": "mask",
            "preset": name,
            "case": 2,
            "clean_acc": clean,
            "asr": asr,
            "proxy": score,
            "geometry": geo.to_dict(),
        }
        rows.append(row)
        print(f"  post-FedAvg clean={clean:.3f} asr={asr:.3f} proxy={score:.3f}")
        if asr > best_mk_asr:
            best_mk_asr = asr
            best_mk_name = name
            best_mk_geo = geo

    winner = merge_geometry(best_sg_geo, best_mk_geo)
    payload = {
        "rows": rows,
        "best_sunglasses_preset": best_sg_name,
        "best_mask_preset": best_mk_name,
        "winner": winner.to_dict(),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    winner_path.write_text(json.dumps(winner.to_dict(), indent=2), encoding="utf-8")
    print(
        f"\nWINNER sunglasses={best_sg_name} (asr={best_sg_asr:.3f}) "
        f"mask={best_mk_name} (asr={best_mk_asr:.3f})"
    )
    print(f"Wrote {winner_path}")
    set_trigger_geometry(winner)
    return winner


def main():
    parser = argparse.ArgumentParser(
        description="Search trigger geometry with post-FedAvg ASR proxy"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=ROOT / "participant_models_geo_search",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=ROOT / "attack" / "geometry_search_results.json",
    )
    parser.add_argument(
        "--winner-path",
        type=Path,
        default=ROOT / "attack" / "best_geometry.json",
    )
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--holdout-start", type=int, default=1400)
    parser.add_argument("--holdout-count", type=int, default=100)
    args = parser.parse_args()

    run_search(
        data_root=args.data_root,
        scratch_root=args.scratch_root,
        results_path=args.results_path,
        winner_path=args.winner_path,
        max_images=args.max_images,
        epochs=args.epochs,
        gamma=args.gamma,
        holdout_start=args.holdout_start,
        holdout_count=args.holdout_count,
    )


if __name__ == "__main__":
    main()
