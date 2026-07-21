"""Post-aggregation attack diagnostic (portal-realistic local proxy).

Scores the GLOBAL model after mixing benign + malicious clients under
FedAvg and robust aggregators — not the raw malicious model alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import SmallCNN
from utilities.aggregators import AGGREGATORS, get_aggregator
from utilities.checks import (
    ATTACK_BENIGN_MODELS_PER_CASE,
    MALICIOUS_MODELS_PER_CASE,
)
from utilities.model_io import load_state_dict_directory, malicious_model_path

from attack.celeba_data import make_hair_dataset
from attack.eval_heldout_asr import evaluate_model
from attack.triggers import trigger_for_case


def proxy_score(clean_acc: float, asr: float) -> float:
    return 0.4 * clean_acc + 0.6 * asr


def parse_aggregators(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("Need at least one aggregator")
    for name in names:
        get_aggregator(name)
    return names


def run_post_agg_eval(
    models_root: Path,
    data_root: Path,
    aggregators: list[str],
    holdout_start: int,
    holdout_count: int,
    image_size: int,
    batch_size: int,
    cases: list[int] | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Post-aggregation eval | holdout start={holdout_start} "
        f"count={holdout_count}"
    )

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
    print(f"Held-out images: {len(dataset)}")

    case_list = cases or list(ATTACK_BENIGN_MODELS_PER_CASE.keys())
    summary: dict[str, list[float]] = {name: [] for name in aggregators}

    for case_number in case_list:
        benign_count = ATTACK_BENIGN_MODELS_PER_CASE[case_number]
        malicious_count = MALICIOUS_MODELS_PER_CASE[case_number]
        trigger = trigger_for_case(case_number)
        print(
            f"\n=== Case {case_number} | trigger={trigger} | "
            f"benign={benign_count} malicious={malicious_count} ==="
        )

        benign_models = load_state_dict_directory(
            ROOT / "attack" / f"case_{case_number}",
            expected_count=benign_count,
        )
        malicious_models = []
        for malicious_index in range(malicious_count):
            path = malicious_model_path(
                models_root, case_number, malicious_index
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing malicious model: {path}")
            malicious_models.append(
                torch.load(path, map_location="cpu", weights_only=True)
            )

        pool = list(benign_models) + malicious_models
        print(
            f"{'agg':>14}  {'clean':>7}  {'asr':>7}  {'proxy':>7}"
        )

        for name in aggregators:
            aggregate_fn = get_aggregator(name)
            aggregated = aggregate_fn(pool)
            model = SmallCNN(num_classes=4)
            model.load_state_dict(aggregated)
            model.to(device)
            clean_acc, asr, _total = evaluate_model(
                model, loader, trigger, device
            )
            score = proxy_score(clean_acc, asr)
            summary[name].append(score)
            print(
                f"{name:>14}  {clean_acc:7.3f}  {asr:7.3f}  {score:7.3f}"
            )

    print("\n=== MEAN PROXY SCORE ACROSS CASES ===")
    for name in aggregators:
        values = summary[name]
        mean = sum(values) / max(len(values), 1)
        print(f"  {name:>14}: {mean:.4f}")
    print(
        "\nGate: compare FedAvg proxy to frozen keep ~0.752 before portal submit.\n"
        "Note: CelebA proxy can overestimate scaling; still better than "
        "malicious-only ASR."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate attack models after aggregation (local proxy)"
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=ROOT / "participant_models",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--aggregators",
        type=parse_aggregators,
        default=parse_aggregators("fedavg,median,multikrum,trimmed_mean"),
        help=f"Comma-separated. Options: {', '.join(sorted(AGGREGATORS))}",
    )
    parser.add_argument("--holdout-start", type=int, default=1400)
    parser.add_argument("--holdout-count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--cases",
        type=str,
        default="1,2,3",
        help="Comma-separated case numbers",
    )
    args = parser.parse_args()
    cases = [int(part.strip()) for part in args.cases.split(",") if part.strip()]

    run_post_agg_eval(
        models_root=args.models_root,
        data_root=args.data_root,
        aggregators=args.aggregators,
        holdout_start=args.holdout_start,
        holdout_count=args.holdout_count,
        image_size=args.image_size,
        batch_size=args.batch_size,
        cases=cases,
    )


if __name__ == "__main__":
    main()
