"""
evaluate_responses.py
---------------------
LLM-as-judge evaluation of generated responses.

If GROQ_API_KEY or GOOGLE_API_KEY is set, uses the LLM to score responses
on a structured rubric.

If no API key is available, clearly marks evaluation as unavailable and
skips this component (does NOT fabricate scores).

Rubric (each criterion scored 1–5):
  1. Relevance       : Is the response relevant to the customer's issue?
  2. Groundedness    : Is the response grounded in the provided historical examples?
  3. Factual safety  : Does the response avoid inventing facts/policies?
  4. Helpfulness     : Does the response actually help the customer?
  5. Brand tone      : Does the response match the brand's professional tone?
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    GOLDEN_EVAL_PATH,
    RESULTS_DIR,
    LLM_PROVIDER,
    SELECTED_BRAND,
)
from src.response_generator import generate_response
from src.utils import setup_logging, save_json, ensure_dir

logger = logging.getLogger(__name__)

JUDGE_RUBRIC_PROMPT = """You are an expert customer support quality evaluator.

You will assess a customer support response on the following 5 criteria.
For each criterion, give a score from 1 to 5 (1=very poor, 5=excellent).
Return ONLY valid JSON, no other text.

Customer message:
\"\"\"{customer_message}\"\"\"

Brand: {brand}

Historical examples used as context:
{examples_text}

Generated response:
\"\"\"{response}\"\"\"

Evaluate and return JSON in this exact format:
{{
  "relevance": <1-5>,
  "groundedness": <1-5>,
  "factual_safety": <1-5>,
  "helpfulness": <1-5>,
  "brand_tone": <1-5>,
  "overall_comment": "<one sentence summary>"
}}"""


def _call_llm_judge(prompt: str) -> dict:
    """Call LLM and parse JSON rubric scores."""
    from src.config import GROQ_API_KEY, GOOGLE_API_KEY, GROQ_MODEL_NAME, GEMINI_MODEL_NAME

    if LLM_PROVIDER == "groq":
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
    elif LLM_PROVIDER == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        resp = model.generate_content(prompt)
        raw = resp.text.strip()
    else:
        raise RuntimeError("No LLM provider configured.")

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


def run_response_evaluation(n_samples: int = 20) -> dict:
    """
    Evaluate response quality using LLM-as-judge.

    Parameters
    ----------
    n_samples : number of golden-set examples to evaluate (keep low to save API quota)

    Returns
    -------
    dict with aggregated scores or {"status": "unavailable"} if no LLM
    """
    ensure_dir(RESULTS_DIR)

    if LLM_PROVIDER == "none":
        msg = (
            "Response evaluation SKIPPED: No LLM API key found.\n"
            "Set GROQ_API_KEY or GOOGLE_API_KEY in .env to enable LLM-as-judge evaluation."
        )
        logger.warning(msg)
        print(f"\n⚠️  {msg}")
        result = {"status": "unavailable", "reason": msg}
        save_json(result, RESULTS_DIR / "response_eval.json")
        return result

    if not GOLDEN_EVAL_PATH.exists():
        raise FileNotFoundError(f"Golden set not found at {GOLDEN_EVAL_PATH}.")

    df = pd.read_csv(GOLDEN_EVAL_PATH)
    # Only evaluate AUTO-HANDLE examples (escalated ones don't get AI responses)
    if "expected_action" in df.columns:
        df = df[df["expected_action"] == "AUTO-HANDLE"].copy()

    df = df.dropna(subset=["message", "intent"]).head(n_samples)

    logger.info("Running LLM-as-judge on %d examples…", len(df))

    all_scores = []
    for _, row in df.iterrows():
        message = str(row["message"])
        intent = str(row["intent"])

        # Generate response (no retrieval available at evaluation time — use empty)
        gen_result = generate_response(
            customer_message=message,
            intent=intent,
            historical_examples=[],
            brand=SELECTED_BRAND,
        )

        if gen_result["status"] == "fallback":
            continue

        response_text = gen_result["response"]
        examples_text = "(No historical examples available at evaluation time)"

        prompt = JUDGE_RUBRIC_PROMPT.format(
            customer_message=message,
            brand=SELECTED_BRAND,
            examples_text=examples_text,
            response=response_text,
        )

        try:
            scores = _call_llm_judge(prompt)
            scores["message"] = message[:100]
            scores["intent"] = intent
            scores["response"] = response_text[:150]
            all_scores.append(scores)
            time.sleep(0.5)   # Rate limit protection
        except Exception as e:
            logger.warning("Judge scoring failed for example: %s", e)

    if not all_scores:
        result = {"status": "no_scores", "reason": "All scoring attempts failed."}
        save_json(result, RESULTS_DIR / "response_eval.json")
        return result

    # Aggregate
    score_df = pd.DataFrame(all_scores)
    numeric_cols = ["relevance", "groundedness", "factual_safety", "helpfulness", "brand_tone"]
    numeric_cols = [c for c in numeric_cols if c in score_df.columns]

    aggregated = {
        col: {
            "mean": round(score_df[col].mean(), 2),
            "min": int(score_df[col].min()),
            "max": int(score_df[col].max()),
        }
        for col in numeric_cols
    }

    result = {
        "status": "success",
        "n_evaluated": len(all_scores),
        "provider": LLM_PROVIDER,
        "scores": aggregated,
        "raw_scores": all_scores,
    }

    save_json(result, RESULTS_DIR / "response_eval.json")
    score_df.to_csv(RESULTS_DIR / "response_eval_detail.csv", index=False)

    print("\n=== RESPONSE QUALITY EVALUATION (LLM-as-Judge) ===")
    print(f"Provider: {LLM_PROVIDER} | Evaluated: {len(all_scores)} examples")
    for col in numeric_cols:
        print(f"  {col:16s}: mean={aggregated[col]['mean']:.2f}  "
              f"(min={aggregated[col]['min']}, max={aggregated[col]['max']})")

    return result


if __name__ == "__main__":
    setup_logging()
    run_response_evaluation()
