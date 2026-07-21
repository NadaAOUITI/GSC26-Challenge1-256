"""Local defense CleanAcc/ASR diagnostic (not for portal upload).

Loads a defense_submission*.py aggregator, mixes benign attack-case models
with optional malicious models, and scores the result on held-out CelebA.
"""

from __future__ import annotations

import argparse
import importlib.util
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
)
from utilities.model_io import load_state_dict_directory, malicious_model_path

from attack.celeba_data import make_hair_dataset
from attack.eval_heldout_asr import evaluate_model
from attack.triggers import trigger_for_case


def defense_score(clean_acc: float, asr: float) -> float:
    return 0.6 * clean_acc + 0.4 * (1.0 - asr)


def load_aggregation_fn(submission_path: Path):
    path = Path(submission_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(
        "local_defense_submission",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "robust_aggregation", None)
    if not callable(function):
        raise ValueError(f"{path} must define robust_aggregation")
    return function


def run_defense_eval(
    submission_path: Path,
    data_root: Path,
    malicious_root: Path | None,
    holdout_start: int,
    holdout_count: int,
    image_size: int,
    batch_size: int,
    cases: list[int],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregate = load_aggregation_fn(submission_path)
    print(f"Device: {device}")
    print(f"Defense file: {submission_path}")

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

    scores = []
    for case_number in cases:
        benign_count = ATTACK_BENIGN_MODELS_PER_CASE[case_number]
        malicious_count = MALICIOUS_MODELS_PER_CASE[case_number]
        trigger = trigger_for_case(case_number)
        print(
            f"\n=== Case {case_number} | trigger={trigger} | "
            f"N_benign={benign_count} ==="
        )

        models = load_state_dict_directory(
            ROOT / "attack" / f"case_{case_number}",
            expected_count=benign_count,
        )
        if malicious_root is not None:
            for malicious_index in range(malicious_count):
                path = malicious_model_path(
                    malicious_root, case_number, malicious_index
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                models.append(
                    torch.load(path, map_location="cpu", weights_only=True)
                )
            print(f"  + {malicious_count} malicious from {malicious_root}")
        else:
            print("  (benign-only pool — no attack models mixed in)")

        n = len(models)
        aggregated = aggregate(n, models)
        model = SmallCNN(num_classes=4)
        model.load_state_dict(aggregated)
        model.to(device)
        clean_acc, asr, total = evaluate_model(
            model, loader, trigger, device
        )
        score = defense_score(clean_acc, asr)
        scores.append(score)
        print(
            f"  N={n}  clean_acc={clean_acc:.3f}  asr={asr:.3f}  "
            f"defense_score={score:.3f}  n_images={total}"
        )

    mean = sum(scores) / max(len(scores), 1)
    print(f"\nMEAN defense proxy score: {mean:.4f}")
    print(
        "Target: ASR near 0 with CleanAcc >= ~0.65 "
        "(portal top defenses ~0.79)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Local defense ASR/CleanAcc diagnostic"
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "defense_submission.py",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--malicious-root",
        type=Path,
        default=None,
        help="Optional participant_models root to mix in as attackers",
    )
    parser.add_argument("--holdout-start", type=int, default=1400)
    parser.add_argument("--holdout-count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cases", type=str, default="1,2,3")
    args = parser.parse_args()
    cases = [int(part.strip()) for part in args.cases.split(",") if part.strip()]

    run_defense_eval(
        submission_path=args.submission,
        data_root=args.data_root,
        malicious_root=args.malicious_root,
        holdout_start=args.holdout_start,
        holdout_count=args.holdout_count,
        image_size=args.image_size,
        batch_size=args.batch_size,
        cases=cases,
    )


if __name__ == "__main__":
    main()
