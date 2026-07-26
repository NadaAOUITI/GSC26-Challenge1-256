"""Export scaled malicious models from an unscaled training run."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack.attack_baseline import default_scale_factor, scale_state_dict
from utilities.checks import ATTACK_BENIGN_MODELS_PER_CASE, MALICIOUS_MODELS_PER_CASE, fedavg
from utilities.model_io import load_state_dict, load_state_dict_directory, malicious_model_path, save_state_dict


def export_scaled(
    unscaled_root: Path,
    output_root: Path,
    gamma: float | str,
) -> None:
    output_root = Path(output_root)
    unscaled_root = Path(unscaled_root)

    for case_number in ATTACK_BENIGN_MODELS_PER_CASE:
        benign_count = ATTACK_BENIGN_MODELS_PER_CASE[case_number]
        malicious_count = MALICIOUS_MODELS_PER_CASE[case_number]
        benign_models = load_state_dict_directory(
            ROOT / "attack" / f"case_{case_number}",
            expected_count=benign_count,
        )
        reference = fedavg(benign_models)
        unscaled = load_state_dict(
            malicious_model_path(unscaled_root, case_number, 0)
        )

        if gamma == "auto":
            case_gamma = default_scale_factor(case_number)
        else:
            case_gamma = float(gamma)

        scaled = scale_state_dict(unscaled, reference, gamma=case_gamma)
        print(
            f"case_{case_number}: gamma={case_gamma:.3f} "
            f"-> writing {malicious_count} malicious models"
        )

        for malicious_index in range(malicious_count):
            state = copy.deepcopy(scaled)
            if malicious_index > 0:
                import torch

                with torch.no_grad():
                    noise_scale = 1e-5 * malicious_index
                    for tensor in state.values():
                        tensor.add_(noise_scale * torch.randn_like(tensor))
            destination = malicious_model_path(output_root, case_number, malicious_index)
            save_state_dict(state, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale unscaled malicious models")
    parser.add_argument(
        "--unscaled-root",
        type=Path,
        default=ROOT / "participant_models_unscaled",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--gamma",
        type=str,
        required=True,
        help="Float scale factor or 'auto' for n/m per case",
    )
    args = parser.parse_args()
    export_scaled(args.unscaled_root, args.output_root, args.gamma)


if __name__ == "__main__":
    main()
