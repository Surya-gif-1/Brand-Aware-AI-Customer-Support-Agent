"""
evaluate_retrieval.py
---------------------
Evaluates retrieval quality using basic precision-style checks.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GOLDEN_EVAL_PATH, RESULTS_DIR
from src.retriever import Retriever
from src.utils import setup_logging, save_json, ensure_dir
import pandas as pd

logger = logging.getLogger(__name__)


def run_retrieval_evaluation() -> dict:
    if not GOLDEN_EVAL_PATH.exists():
        logger.warning("Golden set not found. Skipping retrieval evaluation.")
        return {"status": "skipped"}

    df = pd.read_csv(GOLDEN_EVAL_PATH).dropna(subset=["message"])
    df["message"] = df["message"].fillna("").astype(str)

    # Build retriever from same data (self-retrieval excluded by exclude_ids)
    retriever = Retriever(top_k=3)

    # Create a simple df compatible with retriever build
    build_df = df.rename(columns={"message": "cleaned_text"})
    if "brand_response" not in build_df.columns:
        build_df["brand_response"] = ""
    if "customer_tweet_id" not in build_df.columns:
        build_df["customer_tweet_id"] = build_df.index.tolist()

    retriever.build(build_df)

    scores = []
    for idx, row in df.iterrows():
        results = retriever.retrieve(
            str(row["message"]),
            top_k=3,
            exclude_ids={idx},
        )
        if results:
            scores.append(results[0]["similarity_score"])

    if not scores:
        return {"status": "no_results"}

    metrics = {
        "status": "success",
        "n_queries": len(scores),
        "mean_top1_similarity": round(sum(scores) / len(scores), 4),
        "min_top1_similarity": round(min(scores), 4),
        "max_top1_similarity": round(max(scores), 4),
    }

    ensure_dir(RESULTS_DIR)
    save_json(metrics, RESULTS_DIR / "retrieval_metrics.json")

    print("\n=== RETRIEVAL EVALUATION ===")
    print(f"Queries evaluated : {metrics['n_queries']}")
    print(f"Mean Top-1 similarity: {metrics['mean_top1_similarity']:.4f}")
    print(f"Min  Top-1 similarity: {metrics['min_top1_similarity']:.4f}")

    return metrics


if __name__ == "__main__":
    setup_logging()
    run_retrieval_evaluation()
