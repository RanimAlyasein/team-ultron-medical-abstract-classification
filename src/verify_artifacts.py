"""Verify dataset integrity, canonical metrics, and required repo artifacts."""

from __future__ import annotations

import csv
import math
import re
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
    "results/validation_benchmark.csv",
    "results/final_test_results.csv",
    "results/hierarchical_results.csv",
    "results/calibration_results.csv",
    "results/efficiency_results.csv",
    "figures/class_distribution.png",
    "figures/validation_macro_f1.png",
    "figures/per_class_f1.png",
    "figures/hierarchical_comparison.png",
    "reports/dataset_audit_report.txt",
    "reports/original_train_test_overlap.csv",
    "presentation/Team_Ultron_Final_Presentation.pptx",
    "presentation/Team_Ultron_Final_Presentation.pdf",
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

    class_counts: dict[str, int] = {}
    for row in tables["medical_abstracts_clean.csv"]:
        label = row["condition_label"]
        class_counts[label] = class_counts.get(label, 0) + 1
    assert class_counts == {
        "1": 2195,
        "2": 699,
        "3": 1049,
        "4": 1961,
        "5": 2394,
    }, f"Unexpected clean class distribution: {class_counts}"


def verify_metrics() -> None:
    benchmark = read_rows(ROOT / "results" / "validation_benchmark.csv")
    biomed = next(row for row in benchmark if row["model"] == "BiomedBERT")
    require_close(biomed["val_macro_f1"], 0.8672, "validation Macro F1")
    require_close(biomed["test_macro_f1"], 0.8603, "test Macro F1")

    hierarchical = read_rows(ROOT / "results" / "hierarchical_results.csv")
    combined = next(row for row in hierarchical if "combined" in row["system"])
    require_close(combined["macro_f1"], 0.8599, "hierarchical Macro F1")
    stage1 = next(row for row in hierarchical if "Stage 1" in row["system"])
    require_close(stage1["macro_f1"], 0.8572, "Stage 1 Macro F1")

    calibration = read_rows(ROOT / "results" / "calibration_results.csv")[0]
    require_close(calibration["ece"], 0.1041, "ECE")
    assert calibration["accepted_samples"] == "1153", "Wrong accepted sample count"

    efficiency = read_rows(ROOT / "results" / "efficiency_results.csv")[0]
    require_close(efficiency["abstracts_per_second"], 35.62, "throughput")


def verify_markdown_links() -> None:
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    broken: list[str] = []
    for markdown in ROOT.rglob("*.md"):
        text = markdown.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            target = match.group(1).strip().split(" ", 1)[0]
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (markdown.parent / relative).exists():
                broken.append(f"{markdown.relative_to(ROOT)} -> {target}")
    assert not broken, "Broken Markdown links: " + "; ".join(broken)


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).is_file()]
    assert not missing, f"Missing required files: {missing}"
    verify_dataset()
    verify_metrics()
    verify_markdown_links()
    print("Repository verification passed.")


if __name__ == "__main__":
    main()
