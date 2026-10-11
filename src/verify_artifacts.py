"""Verify dataset integrity, canonical metrics, and required repo artifacts."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "BENCHMARK.md",
    "LICENSE",
    "requirements.txt",
    "data/raw/medical_tc_train.csv",
    "data/raw/medical_tc_test.csv",
    "data/processed/medical_abstracts_clean.csv",
    "data/processed/train.csv",
    "data/processed/validation.csv",
    "data/processed/test.csv",
    "notebooks/04_baseline_benchmark.ipynb",
    "results/benchmark_validation.csv",
    "results/final_test_metrics.csv",
    "results/hierarchical_validation.csv",
    "results/calibration_summary.csv",
    "results/efficiency.csv",
    "presentation/Team_Ultron_Final_Presentation.pptx",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_close(actual: str, expected: float, label: str) -> None:
    value = float(actual)
    if not math.isclose(value, expected, rel_tol=0, abs_tol=0.00005):
        raise AssertionError(f"{label}: expected {expected}, found {value}")


def verify_dataset() -> None:
    processed = ROOT / "data" / "processed"
    expected_sizes = {
        "medical_abstracts_clean.csv": 8298,
        "train.csv": 5808,
        "validation.csv": 1245,
        "test.csv": 1245,
    }
    tables = {name: read_rows(processed / name) for name in expected_sizes}
    for name, expected in expected_sizes.items():
        assert len(tables[name]) == expected, f"{name}: wrong row count"

    split_names = ["train.csv", "validation.csv", "test.csv"]
    id_sets = {name: {row["abstract_id"] for row in tables[name]} for name in split_names}
    text_sets = {name: {row["medical_abstract"] for row in tables[name]} for name in split_names}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            assert not (id_sets[left] & id_sets[right]), f"ID overlap: {left}, {right}"
            assert not (text_sets[left] & text_sets[right]), f"Text overlap: {left}, {right}"

    union_ids = set().union(*(id_sets[name] for name in split_names))
    master_ids = {row["abstract_id"] for row in tables["medical_abstracts_clean.csv"]}
    assert union_ids == master_ids, "Final splits do not exactly cover the clean master"


def verify_metrics() -> None:
    benchmark = read_rows(ROOT / "results" / "benchmark_validation.csv")
    biomed = next(row for row in benchmark if row["model"] == "BiomedBERT")
    require_close(biomed["val_macro_f1"], 0.8672, "validation Macro F1")
    require_close(biomed["test_macro_f1"], 0.8603, "test Macro F1")

    hierarchical = read_rows(ROOT / "results" / "hierarchical_validation.csv")
    combined = next(row for row in hierarchical if "combined" in row["system"])
    require_close(combined["macro_f1"], 0.8599, "hierarchical Macro F1")

    calibration = read_rows(ROOT / "results" / "calibration_summary.csv")[0]
    require_close(calibration["ece"], 0.1041, "ECE")

    efficiency = read_rows(ROOT / "results" / "efficiency.csv")[0]
    require_close(efficiency["abstracts_per_second"], 35.62, "throughput")


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).is_file()]
    assert not missing, f"Missing required files: {missing}"
    verify_dataset()
    verify_metrics()
    print("Repository verification passed.")


if __name__ == "__main__":
    main()
