"""Local gamma grid search with FedAvg simulation on held-out data."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
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
    load_state_dict,
    load_state_dict_directory,
    malicious_model_path,
    save_state_dict,
)

from attack.attack_baseline import scale_state_dict
from attack.celeba_data import make_hair_dataset
from attack.eval_heldout_asr import evaluate_model
from attack.triggers import trigger_for_case


def state_l2_norm(state: dict) -> float:
    total = 0.0
    for tensor in state.values():
        total += float(tensor.detach().float().pow(2).sum().item())
    return total ** 0.5


def proxy_score(clean_acc: float, asr: float) -> float:
    return 0.4 * clean_acc + 0.6 * asr


def parse_gammas(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Need at least one gamma")
    return values


def load_unscaled_malicious(unscaled_root: Path, case_number: int) -> dict:
    path = malicious_model_path(unscaled_root, case_number, 0)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing unscaled model {path}. Train with "
            "--scale-factor 1 --output-root participant_models_unscaled first."
        )
    return load_state_dict(path)


def pick_best_row(rows: list[dict]) -> dict:
    """
    Prefer the lowest gamma that reaches most of the peak ASR.

    CelebA held-out proxy score rises with gamma (aggregate looks more like
    the CelebA-finetuned malicious model), so maximizing proxy blindly
    recovers full n/m scaling -- which crashed portal CleanAcc. A knee rule
    keeps dilution gains without the Exp-2 overscale.
    """
    max_asr = max(row["asr"] for row in rows)
    threshold = 0.90 * max_asr
    eligible = [row for row in rows if row["asr"] >= threshold]
    if not eligible:
        return max(rows, key=lambda row: row["asr"])
    return min(eligible, key=lambda row: row["gamma"])


def run_grid(
    unscaled_root: Path,
    output_root: Path,
    data_root: Path,
    gammas: list[float],
    holdout_start: int,
    holdout_count: int,
    image_size: int,
    batch_size: int,
    results_path: Path,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = make_hair_dataset(
        data_root=data_root,
        max_images=holdout_count,
        image_size=image_size,
        start_index=holdout_start,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"Held-out images: {len(dataset)} (start={holdout_start})")

    all_results: dict[str, list[dict]] = {}
    chosen: dict[str, dict] = {}

    for case_number, benign_count in ATTACK_BENIGN_MODELS_PER_CASE.items():
        malicious_count = MALICIOUS_MODELS_PER_CASE[case_number]
        trigger = trigger_for_case(case_number)
        benign_models = load_state_dict_directory(
            ROOT / "attack" / f"case_{case_number}",
            expected_count=benign_count,
        )
        reference = fedavg(benign_models)
        unscaled = load_unscaled_malicious(unscaled_root, case_number)
        benign_norm = state_l2_norm(reference)

        print(
            f"\n=== Case {case_number} | trigger={trigger} | "
            f"benign={benign_count} malicious={malicious_count} | "
            f"ref_l2={benign_norm:.2f} ==="
        )
        print(
            f"{'gamma':>7}  {'clean':>7}  {'asr':>7}  {'proxy':>7}  "
            f"{'mal_l2':>8}  {'l2_ratio':>8}"
        )

        case_rows: list[dict] = []
        for gamma in gammas:
            scaled = scale_state_dict(unscaled, reference, gamma=gamma)
            mal_norm = state_l2_norm(scaled)
            aggregate_inputs = list(benign_models) + [
                copy.deepcopy(scaled) for _ in range(malicious_count)
            ]
            aggregated = fedavg(aggregate_inputs)

            model = SmallCNN(num_classes=4)
            model.load_state_dict(aggregated)
            model.to(device)
            clean_acc, asr, total = evaluate_model(
                model, loader, trigger, device
            )
            score = proxy_score(clean_acc, asr)
            row = {
                "case": case_number,
                "gamma": gamma,
                "clean_acc": clean_acc,
                "asr": asr,
                "proxy_score": score,
                "malicious_l2": mal_norm,
                "reference_l2": benign_norm,
                "l2_ratio": mal_norm / max(benign_norm, 1e-12),
                "n": total,
            }
            case_rows.append(row)
            print(
                f"{gamma:7.2f}  {clean_acc:7.3f}  {asr:7.3f}  {score:7.3f}  "
                f"{mal_norm:8.2f}  {row['l2_ratio']:8.3f}"
            )

        best = pick_best_row(case_rows)
        chosen[str(case_number)] = best
        all_results[str(case_number)] = case_rows
        print(
            f"  CHOSEN gamma={best['gamma']:.2f}  "
            f"clean={best['clean_acc']:.3f}  asr={best['asr']:.3f}  "
            f"proxy={best['proxy_score']:.3f}"
        )

    # Export best scaled models into output_root.
    output_root = Path(output_root)
    for case_number in ATTACK_BENIGN_MODELS_PER_CASE:
        best = chosen[str(case_number)]
        gamma = float(best["gamma"])
        benign_models = load_state_dict_directory(
            ROOT / "attack" / f"case_{case_number}",
            expected_count=ATTACK_BENIGN_MODELS_PER_CASE[case_number],
        )
        reference = fedavg(benign_models)
        unscaled = load_unscaled_malicious(unscaled_root, case_number)
        scaled = scale_state_dict(unscaled, reference, gamma=gamma)

        for malicious_index in range(MALICIOUS_MODELS_PER_CASE[case_number]):
            state = copy.deepcopy(scaled)
            if malicious_index > 0:
                noise_scale = 1e-5 * malicious_index
                for tensor in state.values():
                    tensor.add_(noise_scale * torch.randn_like(tensor))
            destination = malicious_model_path(
                output_root, case_number, malicious_index
            )
            save_state_dict(state, destination)
        print(
            f"Wrote case_{case_number} malicious models with gamma={gamma:.2f} "
            f"-> {output_root / f'case_{case_number}'}"
        )

    payload = {"results": all_results, "chosen": chosen}
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved grid results to {results_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Grid-search model-replacement gamma with local FedAvg sim"
    )
    parser.add_argument(
        "--unscaled-root",
        type=Path,
        default=ROOT / "participant_models_unscaled",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "participant_models",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--gammas",
        type=parse_gammas,
        default=parse_gammas("1,1.5,2,2.5,3,4,5"),
    )
    parser.add_argument("--holdout-start", type=int, default=1000)
    parser.add_argument("--holdout-count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--results-path",
        type=Path,
        default=ROOT / "attack" / "scale_grid_results.json",
    )
    args = parser.parse_args()

    run_grid(
        unscaled_root=args.unscaled_root,
        output_root=args.output_root,
        data_root=args.data_root,
        gammas=args.gammas,
        holdout_start=args.holdout_start,
        holdout_count=args.holdout_count,
        image_size=args.image_size,
        batch_size=args.batch_size,
        results_path=args.results_path,
    )


if __name__ == "__main__":
    main()
