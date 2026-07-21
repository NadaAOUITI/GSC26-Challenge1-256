"""Held-out local ASR diagnostic on current malicious models."""

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
from utilities.checks import MALICIOUS_MODELS_PER_CASE
from utilities.model_io import load_state_dict, malicious_model_path

from attack.celeba_data import TARGET_CLASS, make_hair_dataset
from attack.triggers import apply_trigger_tensor, trigger_for_case


def evaluate_model(
    model: SmallCNN,
    loader: DataLoader,
    trigger: str,
    device: torch.device,
) -> tuple[float, float, int]:
    model.eval()
    correct_clean = 0
    asr_hits = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            batch_size = images.size(0)
            total += batch_size

            clean_pred = model(images.to(device)).argmax(dim=1).cpu()
            correct_clean += int((clean_pred == labels).sum().item())

            triggered = torch.stack(
                [apply_trigger_tensor(image, trigger) for image in images]
            ).to(device)
            triggered_pred = model(triggered).argmax(dim=1).cpu()
            asr_hits += int((triggered_pred == TARGET_CLASS).sum().item())

    if total == 0:
        raise RuntimeError("Held-out loader is empty.")

    return correct_clean / total, asr_hits / total, total


def run_heldout_eval(
    data_root: Path,
    models_root: Path,
    holdout_start: int,
    holdout_count: int,
    image_size: int,
    batch_size: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Held-out slice: start={holdout_start}, count={holdout_count} "
        f"(synthetic triggers, local malicious models only)"
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
    print(f"Held-out images loaded: {len(dataset)}")

    for case_number, model_count in MALICIOUS_MODELS_PER_CASE.items():
        trigger = trigger_for_case(case_number)
        print(f"\n=== Case {case_number} | trigger={trigger} ===")

        case_clean = []
        case_asr = []

        for malicious_index in range(model_count):
            path = malicious_model_path(models_root, case_number, malicious_index)
            if not path.is_file():
                raise FileNotFoundError(f"Missing malicious model: {path}")

            model = SmallCNN(num_classes=4)
            model.load_state_dict(load_state_dict(path))
            model.to(device)

            clean_acc, asr, total = evaluate_model(model, loader, trigger, device)
            case_clean.append(clean_acc)
            case_asr.append(asr)
            print(
                f"  malicious_{malicious_index}: "
                f"clean_acc={clean_acc:.3f}  "
                f"triggered_asr={asr:.3f}  "
                f"n={total}"
            )

        mean_clean = sum(case_clean) / len(case_clean)
        mean_asr = sum(case_asr) / len(case_asr)
        print(
            f"  CASE MEAN: clean_acc={mean_clean:.3f}  "
            f"triggered_asr={mean_asr:.3f}"
        )

    print(
        "\nInterpretation:\n"
        "  held-out ASR still high (~>0.90) -> overfitting weak; "
        "portal gap mostly dilution / trigger mismatch\n"
        "  held-out ASR crashes -> fix generalization before Experiment 2 scaling"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Held-out synthetic ASR diagnostic for malicious models"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=ROOT / "participant_models",
    )
    parser.add_argument("--holdout-start", type=int, default=1000)
    parser.add_argument("--holdout-count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_heldout_eval(
        data_root=args.data_root,
        models_root=args.models_root,
        holdout_start=args.holdout_start,
        holdout_count=args.holdout_count,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
