"""Generate attack/sample_submission.csv from EXPECTED_STATE_LAYOUT (no torch)."""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_STATE_LAYOUT = OrderedDict([
    ("features.0.weight", (32, 3, 3, 3)),
    ("features.0.bias", (32,)),
    ("features.3.weight", (64, 32, 3, 3)),
    ("features.3.bias", (64,)),
    ("features.6.weight", (128, 64, 3, 3)),
    ("features.6.bias", (128,)),
    ("classifier.weight", (4, 128)),
    ("classifier.bias", (4,)),
])

MALICIOUS_MODELS_PER_CASE = OrderedDict([
    (1, 2),
    (2, 5),
    (3, 5),
])


def main() -> None:
    out = ROOT / "attack" / "sample_submission.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["row_id", "value"])
        for case_number, model_count in MALICIOUS_MODELS_PER_CASE.items():
            for malicious_index in range(model_count):
                for parameter_name, shape in EXPECTED_STATE_LAYOUT.items():
                    count = 1
                    for dimension in shape:
                        count *= dimension
                    for flat_index in range(count):
                        row_id = (
                            f"case_{case_number}::"
                            f"malicious_{malicious_index}::"
                            f"{parameter_name}::{flat_index}"
                        )
                        writer.writerow([row_id, "0.0"])
    print(f"Created: {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
