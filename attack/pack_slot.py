"""Pack and validate an attack submission for a portal slot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to attack_submission_slot{N}.csv",
    )
    args = parser.parse_args()

    output = args.output or (ROOT / f"attack_submission_slot{args.slot}.csv")
    models_root = args.models_root.resolve()

    commands = [
        [
            sys.executable,
            str(ROOT / "attack" / "create_attack_submission.py"),
            "--models-root",
            str(models_root),
            "--output",
            str(output),
        ],
        [
            sys.executable,
            str(ROOT / "attack" / "validate_attack_submission.py"),
            "--submission",
            str(output),
        ],
    ]

    for command in commands:
        print(">", " ".join(command))
        subprocess.run(command, check=True, cwd=ROOT)

    print(f"Slot {args.slot} ready: {output}")


if __name__ == "__main__":
    main()
