"""Generate repository figures from canonical data and result tables."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CLASS_NAMES = {
    "1": "Neoplasms",
    "2": "Digestive",
    "3": "Nervous",
    "4": "Cardiovascular",
    "5": "General",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def class_distribution_figure() -> None:
    rows = read_csv(DATA / "medical_abstracts_clean.csv")
    counts = Counter(row["condition_label"] for row in rows)
    labels = [CLASS_NAMES[str(label)] for label in range(1, 6)]
    values = [counts[str(label)] for label in range(1, 6)]

    plt.figure(figsize=(10, 5.8))
    bars = plt.bar(labels, values, color="#4c78a8")
    plt.title("Final cleaned class distribution")
    plt.ylabel("Abstracts")
    plt.xticks(rotation=20, ha="right")
    plt.bar_label(bars, labels=[f"{value:,}" for value in values], padding=4)
    plt.ylim(0, max(values) * 1.13)
    save_figure(FIGURES / "class_distribution.png")


def validation_benchmark_figure() -> None:
    rows = sorted(
        read_csv(RESULTS / "validation_benchmark.csv"),
        key=lambda row: float(row["val_macro_f1"]),
    )
    labels = [row["model"] for row in rows]
    values = [float(row["val_macro_f1"]) for row in rows]
    colors = ["#2a9d8f" if label == "BiomedBERT" else "#4c78a8" for label in labels]

    plt.figure(figsize=(10, 6.5))
    bars = plt.barh(labels, values, color=colors)
    plt.title("Validation Macro F1 by model")
    plt.xlabel("Macro F1")
    plt.xlim(0.45, 0.90)
    plt.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=4)
    save_figure(FIGURES / "validation_macro_f1.png")


def per_class_f1_figure() -> None:
    rows = read_csv(RESULTS / "biomedbert_test_per_class.csv")
    labels = [CLASS_NAMES[row["label"]] for row in rows]
    values = [float(row["f1"]) for row in rows]

    plt.figure(figsize=(10, 5.8))
    bars = plt.bar(labels, values, color="#2a9d8f")
    plt.title("BiomedBERT locked-test F1 by class")
    plt.ylabel("F1")
    plt.ylim(0.70, 0.97)
    plt.xticks(rotation=20, ha="right")
    plt.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=4)
    save_figure(FIGURES / "per_class_f1.png")


def hierarchical_comparison_figure() -> None:
    rows = read_csv(RESULTS / "hierarchical_results.csv")
    by_system = {row["system"]: row for row in rows}
    selected = [
        by_system["BiomedBERT flat"],
        by_system["BiomedBERT hierarchical combined"],
    ]
    labels = ["Flat BiomedBERT", "Hierarchical system"]
    values = [float(row["macro_f1"]) for row in selected]

    plt.figure(figsize=(7, 5.5))
    bars = plt.bar(labels, values, color=["#2a9d8f", "#e9c46a"])
    plt.title("Flat and hierarchical validation performance")
    plt.ylabel("Macro F1")
    plt.ylim(0.84, 0.88)
    plt.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=5)
    save_figure(FIGURES / "hierarchical_comparison.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    class_distribution_figure()
    validation_benchmark_figure()
    per_class_f1_figure()
    hierarchical_comparison_figure()
    print(f"Wrote four figures to {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
