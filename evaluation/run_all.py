"""
run_all.py
----------
Master evaluation script.  Runs all evaluation components in sequence.

Usage:
    python evaluation/run_all.py [--nrows N] [--brand BRAND]

Steps:
  1. Create golden evaluation set (if not already present)
  2. Evaluate all intent classifiers
  3. Evaluate escalation engine
  4. Evaluate response quality (LLM-as-judge, if API key present)
  5. Print overall summary
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging, print_banner
from src.config import GOLDEN_EVAL_PATH, RESULTS_DIR, SELECTED_BRAND

logger = logging.getLogger(__name__)


def run_all(nrows: int = None, brand: str = SELECTED_BRAND, skip_responses: bool = False):
    setup_logging()
    print_banner("Hiver AI Support Agent — Full Evaluation")

    start_time = time.time()
    results_summary = {}

    # ----------------------------------------------------------------
    # Step 1: Create golden set (if missing)
    # ----------------------------------------------------------------
    print("\n[1/4] Creating / loading golden evaluation set…")
    if not GOLDEN_EVAL_PATH.exists():
        from evaluation.create_golden_set import create_golden_set, save_intent_definitions
        save_intent_definitions()
        create_golden_set(nrows=nrows, brand=brand)
    else:
        logger.info("Golden set already exists at %s. Skipping creation.", GOLDEN_EVAL_PATH)
        import pandas as pd
        df = pd.read_csv(GOLDEN_EVAL_PATH)
        print(f"   Loaded existing golden set: {len(df)} examples")

    # ----------------------------------------------------------------
    # Step 2: Evaluate classifiers
    # ----------------------------------------------------------------
    print("\n[2/4] Evaluating intent classifiers…")
    try:
        from evaluation.evaluate_classifier import run_evaluation
        clf_results = run_evaluation()
        results_summary["classifiers"] = {
            k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
            for k, v in clf_results.items()
        }
        print("   ✓ Classifier evaluation complete")
    except Exception as e:
        logger.error("Classifier evaluation failed: %s", e)
        results_summary["classifiers"] = {"error": str(e)}
        print(f"   ✗ Classifier evaluation failed: {e}")

    # ----------------------------------------------------------------
    # Step 3: Evaluate escalation
    # ----------------------------------------------------------------
    print("\n[3/4] Evaluating escalation engine…")
    try:
        from evaluation.evaluate_escalation import run_escalation_evaluation
        esc_results = run_escalation_evaluation()
        results_summary["escalation"] = esc_results
        print("   ✓ Escalation evaluation complete")
    except Exception as e:
        logger.error("Escalation evaluation failed: %s", e)
        results_summary["escalation"] = {"error": str(e)}
        print(f"   ✗ Escalation evaluation failed: {e}")

    # ----------------------------------------------------------------
    # Step 4: Evaluate responses (optional)
    # ----------------------------------------------------------------
    if not skip_responses:
        print("\n[4/4] Evaluating response quality (LLM-as-judge)…")
        try:
            from evaluation.evaluate_responses import run_response_evaluation
            resp_results = run_response_evaluation(n_samples=15)
            results_summary["response_eval"] = resp_results
            print("   ✓ Response evaluation complete")
        except Exception as e:
            logger.error("Response evaluation failed: %s", e)
            results_summary["response_eval"] = {"error": str(e)}
            print(f"   ✗ Response evaluation failed: {e}")
    else:
        print("\n[4/4] Response evaluation skipped (--skip-responses flag).")

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    elapsed = time.time() - start_time
    print_banner(f"Evaluation Complete — {elapsed:.1f}s")

    # Print classifier comparison table
    if "classifiers" in results_summary and "error" not in results_summary["classifiers"]:
        print("\nClassifier Performance Summary:")
        print(f"  {'Classifier':<28} {'Accuracy':>10} {'Macro F1':>10}")
        print(f"  {'-'*50}")
        for name, metrics in results_summary["classifiers"].items():
            print(f"  {name:<28} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}")

    # Print escalation summary
    if "escalation" in results_summary and "error" not in results_summary["escalation"]:
        esc = results_summary["escalation"]
        print(f"\nEscalation Engine:")
        print(f"  Accuracy  : {esc.get('accuracy', 'N/A')}")
        print(f"  ESCALATE F1: {esc.get('escalation_f1', 'N/A')}")

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("Files:")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"  - {f.name}")

    from src.utils import save_json
    save_json(results_summary, RESULTS_DIR / "overall_summary.json")
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit dataset rows loaded (for quick testing)")
    parser.add_argument("--brand", type=str, default=SELECTED_BRAND,
                        help="Brand to evaluate")
    parser.add_argument("--skip-responses", action="store_true",
                        help="Skip LLM response evaluation")
    args = parser.parse_args()

    run_all(
        nrows=args.nrows,
        brand=args.brand,
        skip_responses=args.skip_responses,
    )
