"""Calculate confidence coverage and calibration from saved model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"true_label", "predicted_label", "confidence"}


def calibration_bins(frame: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    records: list[dict[str, float | int]] = []
    for index in range(n_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == 0:
            mask = (frame["confidence"] >= lower) & (frame["confidence"] <= upper)
        else:
            mask = (frame["confidence"] > lower) & (frame["confidence"] <= upper)
        subset = frame.loc[mask]
        if subset.empty:
            continue
        accuracy = (subset["true_label"] == subset["predicted_label"]).mean()
        records.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "samples": len(subset),
                "accuracy": accuracy,
                "mean_confidence": subset["confidence"].mean(),
            }
        )
    return pd.DataFrame(records)


def analyze(frame: pd.DataFrame, threshold: float, n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Prediction file is empty")
    if not frame["confidence"].between(0.0, 1.0).all():
        raise ValueError("confidence values must be between 0 and 1")

    bins = calibration_bins(frame, n_bins)
    ece = (
        bins["samples"]
        / len(frame)
        * (bins["accuracy"] - bins["mean_confidence"]).abs()
    ).sum()
    accepted = frame.loc[frame["confidence"] >= threshold]
    summary = pd.DataFrame(
        [
            {
                "accuracy": (frame["true_label"] == frame["predicted_label"]).mean(),
                "mean_confidence": frame["confidence"].mean(),
                "ece": ece,
                "confidence_threshold": threshold,
                "coverage": len(accepted) / len(frame),
                "accepted_samples": len(accepted),
                "total_samples": len(frame),
                "accepted_accuracy": (
                    (accepted["true_label"] == accepted["predicted_label"]).mean()
                    if not accepted.empty
                    else np.nan
                ),
            }
        ]
    )
    return summary, bins


def save_reliability_diagram(bins: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 6.0))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(
        bins["mean_confidence"],
        bins["accuracy"],
        marker="o",
        color="#2a9d8f",
        label="Observed",
    )
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze calibration from true labels, predictions, and confidence scores."
    )
    parser.add_argument(
        "predictions",
        type=Path,
        help="CSV containing true_label, predicted_label, and confidence",
    )
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--bins-output", type=Path)
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    summary, bins = analyze(frame, args.threshold, args.bins)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    if args.bins_output:
        args.bins_output.parent.mkdir(parents=True, exist_ok=True)
        bins.to_csv(args.bins_output, index=False)
    if args.figure_output:
        save_reliability_diagram(bins, args.figure_output)


if __name__ == "__main__":
    main()
