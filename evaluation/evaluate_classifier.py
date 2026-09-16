"""
evaluate_classifier.py
----------------------
Evaluates intent classification performance across all three classifiers:
  1. MajorityClassifier (trivial baseline)
  2. TfidfLRClassifier  (simple ML baseline)
  3. EmbeddingClassifier (final system)

All metrics are computed against the golden evaluation set ONLY.
Training examples are never used as test examples (no leakage).

Outputs (saved to evaluation/results/):
  - metrics.json            : all metrics for all classifiers
  - classification_report.json : per-class detailed report
  - confusion_matrix.png    : confusion matrix for final classifier
  - evaluation_summary.csv  : quick comparison table
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    GOLDEN_EVAL_PATH,
    RESULTS_DIR,
    RANDOM_SEED,
)
from src.intent_classifier import (
    MajorityClassifier,
    TfidfLRClassifier,
    EmbeddingClassifier,
)
from src.utils import setup_logging, save_json, ensure_dir

logger = logging.getLogger(__name__)


def _compute_metrics(y_true: list, y_pred: list, labels: list) -> dict:
    """Compute classification metrics using sklearn."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": labels,
    }


def _plot_confusion_matrix(cm: list, labels: list, title: str, save_path: Path) -> None:
    """Save a confusion matrix plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            np.array(cm),
            annot=True,
            fmt="d",
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", save_path)
    except Exception as e:
        logger.warning("Could not save confusion matrix plot: %s", e)


def run_evaluation(train_ratio: float = 0.8) -> dict:
    """
    Run full classifier evaluation.

    Uses a fixed train/test split of the golden evaluation set.
    The split is stratified to maintain class balance.
    """
    from sklearn.model_selection import train_test_split

    # Load golden set
    if not GOLDEN_EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Golden evaluation set not found at {GOLDEN_EVAL_PATH}.\n"
            "Run: python evaluation/create_golden_set.py"
        )

    df = pd.read_csv(GOLDEN_EVAL_PATH)
    logger.info("Loaded golden set: %d examples", len(df))

    required_cols = {"message", "intent"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Golden set missing columns: {missing}")

    df = df.dropna(subset=["message", "intent"]).copy()
    df["message"] = df["message"].fillna("").astype(str)

    all_labels = sorted(df["intent"].unique().tolist())

    # Train/test split (stratified)
    train_df, test_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df["intent"],
        random_state=RANDOM_SEED,
    )

    X_train = train_df["message"].tolist()
    y_train = train_df["intent"].tolist()
    X_test = test_df["message"].tolist()
    y_test = test_df["intent"].tolist()

    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))

    all_metrics = {}

    # ------------------------------------------------------------------
    # 1. Majority baseline
    # ------------------------------------------------------------------
    logger.info("Evaluating MajorityClassifier…")
    majority = MajorityClassifier()
    majority.fit(X_train, y_train)
    y_pred_majority = majority.predict(X_test)
    all_metrics["majority_baseline"] = _compute_metrics(y_test, y_pred_majority, all_labels)
    all_metrics["majority_baseline"]["majority_class"] = majority.majority_class_

    # ------------------------------------------------------------------
    # 2. TF-IDF + LR baseline
    # ------------------------------------------------------------------
    logger.info("Evaluating TfidfLRClassifier…")
    tfidf_lr = TfidfLRClassifier()
    tfidf_lr.fit(X_train, y_train)
    y_pred_tfidf = tfidf_lr.predict(X_test)
    all_metrics["tfidf_lr_baseline"] = _compute_metrics(y_test, y_pred_tfidf, all_labels)

    # Save TF-IDF model
    try:
        tfidf_lr.save()
    except Exception as e:
        logger.warning("Could not save TfidfLR model: %s", e)

    # ------------------------------------------------------------------
    # 3. Embedding classifier (final system)
    # ------------------------------------------------------------------
    logger.info("Evaluating EmbeddingClassifier (this may take a minute)…")
    emb_clf = EmbeddingClassifier()
    emb_clf.fit(X_train, y_train)
    y_pred_emb = emb_clf.predict(X_test)
    all_metrics["embedding_classifier"] = _compute_metrics(y_test, y_pred_emb, all_labels)

    # Save embedding classifier
    try:
        emb_clf.save()
    except Exception as e:
        logger.warning("Could not save EmbeddingClassifier: %s", e)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    ensure_dir(RESULTS_DIR)

    # Full metrics JSON
    save_json(all_metrics, RESULTS_DIR / "metrics.json")

    # Classification report JSON (final classifier only)
    save_json(
        all_metrics["embedding_classifier"]["classification_report"],
        RESULTS_DIR / "classification_report.json",
    )

    # Confusion matrix plot
    _plot_confusion_matrix(
        cm=all_metrics["embedding_classifier"]["confusion_matrix"],
        labels=all_labels,
        title="EmbeddingClassifier — Confusion Matrix",
        save_path=RESULTS_DIR / "confusion_matrix.png",
    )

    # Summary CSV
    summary_rows = []
    for clf_name, metrics in all_metrics.items():
        summary_rows.append({
            "classifier": clf_name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)

    # Print results
    print("\n=== CLASSIFICATION EVALUATION RESULTS ===")
    print(summary_df.to_string(index=False))

    return all_metrics


if __name__ == "__main__":
    setup_logging()
    run_evaluation()
