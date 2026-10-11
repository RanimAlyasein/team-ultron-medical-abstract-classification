"""Reproduce the final classical baseline configurations.

The script fits every text representation on the training split only, evaluates
on validation, and also records test metrics for transparent comparison. Model
selection must still use validation Macro F1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def metrics(model_name: str, split: str, y_true, y_pred) -> dict[str, object]:
    """Return the common single-label benchmark metrics."""
    return {
        "model": model_name,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def evaluate_configuration(
    name: str,
    vectorizer,
    model,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> list[dict[str, object]]:
    """Fit one pipeline on train and evaluate without refitting."""
    x_train = vectorizer.fit_transform(train["medical_abstract"].astype(str))
    x_validation = vectorizer.transform(
        validation["medical_abstract"].astype(str)
    )
    x_test = vectorizer.transform(test["medical_abstract"].astype(str))
    model.fit(x_train, train["condition_label"])

    return [
        metrics(
            name,
            "validation",
            validation["condition_label"],
            model.predict(x_validation),
        ),
        metrics(
            name,
            "test",
            test["condition_label"],
            model.predict(x_test),
        ),
    ]


def build_configurations():
    """Return the four classical configurations recorded by the project."""
    return [
        (
            "Naive Bayes default",
            TfidfVectorizer(
                lowercase=True, max_features=30_000, ngram_range=(1, 2)
            ),
            MultinomialNB(alpha=1.0),
        ),
        (
            "Naive Bayes tuned",
            CountVectorizer(lowercase=True, ngram_range=(1, 1)),
            MultinomialNB(alpha=0.5),
        ),
        (
            "Logistic Regression",
            TfidfVectorizer(
                lowercase=True, max_features=30_000, ngram_range=(1, 1)
            ),
            LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=1_000,
                random_state=42,
            ),
        ),
        (
            "Linear SVM",
            FeatureUnion(
                [
                    (
                        "word",
                        TfidfVectorizer(
                            lowercase=True,
                            max_features=30_000,
                            ngram_range=(1, 2),
                        ),
                    ),
                    (
                        "char",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            lowercase=True,
                            max_features=20_000,
                            ngram_range=(3, 5),
                        ),
                    ),
                ]
            ),
            LinearSVC(C=0.5, random_state=42),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/classical_reproduction.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.data_dir / "train.csv")
    validation = pd.read_csv(args.data_dir / "validation.csv")
    test = pd.read_csv(args.data_dir / "test.csv")

    rows: list[dict[str, object]] = []
    for name, vectorizer, model in build_configurations():
        print(f"Running {name}...")
        rows.extend(
            evaluate_configuration(
                name, vectorizer, model, train, validation, test
            )
        )

    output = pd.DataFrame(rows)
    output = output.sort_values(
        ["split", "macro_f1"], ascending=[True, False]
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, float_format="%.6f")
    print(output.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
