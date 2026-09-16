"""
create_golden_set.py
--------------------
Creates the hand-labelled golden evaluation set (~200 examples).

IMPORTANT HONESTY NOTE:
  This script uses keyword/pattern-based heuristics to assign initial intent labels
  and creates a CSV for human review.  The CSV is clearly marked as
  "NEEDS HUMAN REVIEW" so the labeller can correct any mis-labelled rows.

  The script does NOT silently convert model predictions into ground truth.
  It uses deterministic keyword rules, which are transparent and auditable.

Steps:
  1. Load brand conversations from the full or sample dataset.
  2. Preprocess text.
  3. Apply keyword-based intent assignment (heuristic, not model-based).
  4. Stratified-sample ~200 examples across intents.
  5. Assign escalation labels based on transparent rules.
  6. Save to data/processed/golden_eval.csv with a 'needs_review' flag.
  7. Print summary statistics.

Usage:
    python evaluation/create_golden_set.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SELECTED_BRAND,
    GOLDEN_EVAL_PATH,
    PROCESSED_DATA_DIR,
    INTENT_DEFINITIONS_PATH,
    RANDOM_SEED,
    GOLDEN_SET_SIZE,
    HIGH_RISK_INTENTS,
    ESCALATION_KEYWORDS,
    CONFIDENCE_ESCALATION_THRESHOLD,
)
from src.data_loader import load_brand_conversations
from src.preprocessing import preprocess_dataframe
from src.utils import setup_logging, save_json, ensure_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent taxonomy and keyword patterns
# ---------------------------------------------------------------------------

# This taxonomy was derived by analyzing AmazonHelp conversations manually.
# Key signals: word frequency analysis on cleaned_text for the top brand.

INTENT_TAXONOMY = {
    "order_delivery": {
        "description": "Customer asks about order status, shipping, delivery delays, or tracking.",
        "keywords": [
            "order", "deliver", "shipping", "shipment", "track", "package",
            "haven't received", "not arrived", "where is my", "dispatched",
            "estimated arrival", "delayed", "late delivery",
        ],
        "examples": [
            "My package hasn't arrived yet, it's been 2 weeks",
            "Where is my order? The tracking shows it's stuck",
            "My delivery is late, can you help?",
        ],
    },
    "refund_return": {
        "description": "Customer requests a refund, return, or replacement.",
        "keywords": [
            "refund", "return", "money back", "reimburse", "replacement",
            "send back", "exchange", "credit", "charged twice",
        ],
        "examples": [
            "I want a refund for my damaged product",
            "How do I return this item?",
            "I was charged twice, please refund one",
        ],
    },
    "account_login": {
        "description": "Customer has issues logging in, password reset, or account access.",
        "keywords": [
            "login", "log in", "sign in", "password", "account", "locked",
            "reset", "access", "can't access", "forgot", "username",
            "verification", "2fa", "two factor",
        ],
        "examples": [
            "I can't log into my account",
            "I forgot my password, please help",
            "My account is locked",
        ],
    },
    "payment_billing": {
        "description": "Customer has questions about charges, billing, payment methods.",
        "keywords": [
            "payment", "charge", "bill", "invoice", "debit", "credit card",
            "overcharged", "unauthorized charge", "subscription", "fee",
            "price", "cost", "amount",
        ],
        "examples": [
            "Why was I charged $50?",
            "My credit card was declined",
            "I see an unauthorized charge on my account",
        ],
    },
    "technical_issue": {
        "description": "Customer reports a bug, app crash, website error, or technical problem.",
        "keywords": [
            "error", "bug", "crash", "not working", "broken", "loading",
            "app", "website", "glitch", "freeze", "slow", "issue",
            "problem", "fail", "doesn't work", "not loading",
        ],
        "examples": [
            "The app keeps crashing on my phone",
            "I'm getting an error when I try to checkout",
            "Your website is down",
        ],
    },
    "cancellation": {
        "description": "Customer wants to cancel an order, subscription, or service.",
        "keywords": [
            "cancel", "cancellation", "stop", "unsubscribe", "terminate",
            "end subscription", "cancel order", "cancel my",
        ],
        "examples": [
            "I want to cancel my order",
            "Please cancel my subscription",
            "How do I cancel?",
        ],
    },
    "product_inquiry": {
        "description": "Customer asks about product details, availability, or compatibility.",
        "keywords": [
            "available", "in stock", "compatible", "specs", "features",
            "warranty", "product", "item", "model", "version", "does it",
            "will it work", "question about",
        ],
        "examples": [
            "Is this product available in blue?",
            "What is the warranty on this item?",
            "Does this work with iPhone?",
        ],
    },
    "complaint_other": {
        "description": "General complaints, negative experiences, or unclassifiable messages.",
        "keywords": [
            "disappointed", "terrible", "awful", "worst", "horrible",
            "unacceptable", "rude", "angry", "upset", "complaint",
            "never again", "disgusting", "outrageous",
        ],
        "examples": [
            "This is the worst service I've ever experienced",
            "I'm very disappointed with your company",
            "Your customer service is terrible",
        ],
    },
}


def assign_intent(text: str) -> tuple[str, bool]:
    """
    Assign intent using keyword matching heuristics.

    Returns (intent_label, needs_review).
    needs_review=True when multiple intents match (ambiguous) or none match.
    """
    text_lower = text.lower()
    matched = []

    for intent, defn in INTENT_TAXONOMY.items():
        for kw in defn["keywords"]:
            if kw in text_lower:
                matched.append(intent)
                break

    if len(matched) == 1:
        return matched[0], False          # Clear match
    elif len(matched) == 0:
        return "complaint_other", True    # No match → fallback + needs review
    else:
        # Multiple matches — pick the first in taxonomy order (deterministic)
        return matched[0], True           # Needs review


def assign_escalation_label(row: pd.Series) -> str:
    """
    Assign a ground-truth escalation label for the golden set.

    Rules (same logic as escalation.py but applied to ground-truth intents):
      - ESCALATE if intent is in HIGH_RISK_INTENTS
      - ESCALATE if text contains escalation keywords
      - ESCALATE if needs_review == True (ambiguous)
      - AUTO-HANDLE otherwise
    """
    text = str(row.get("message", "")).lower()
    intent = row.get("intent", "")
    needs_review = row.get("needs_review", False)

    if intent in HIGH_RISK_INTENTS:
        return "ESCALATE"

    for kw in ESCALATION_KEYWORDS:
        import re
        if re.search(r"\b" + re.escape(kw) + r"\b", text):
            return "ESCALATE"

    if needs_review:
        return "ESCALATE"

    return "AUTO-HANDLE"


def create_golden_set(
    nrows: Optional[int] = None,
    target_size: int = GOLDEN_SET_SIZE,
    brand: str = SELECTED_BRAND,
) -> pd.DataFrame:
    """
    Create and save the golden evaluation set.

    Parameters
    ----------
    nrows       : limit dataset rows loaded (for fast testing)
    target_size : target number of golden examples
    brand       : selected brand

    Returns
    -------
    golden_df : the golden evaluation DataFrame
    """
    from typing import Optional

    logger.info("Loading brand conversations for brand '%s'…", brand)
    df = load_brand_conversations(brand=brand, nrows=nrows)

    logger.info("Preprocessing…")
    df = preprocess_dataframe(df, text_col="customer_text", response_col="brand_response")

    if len(df) == 0:
        raise ValueError(
            f"No conversations found for brand '{brand}'. "
            "Check the dataset and brand selection."
        )

    logger.info("Assigning intent labels using keyword heuristics…")
    df[["intent", "needs_review"]] = df["cleaned_text"].apply(
        lambda t: pd.Series(assign_intent(t))
    )

    logger.info("Intent distribution before sampling:\n%s", df["intent"].value_counts())

    # Stratified sampling: sample up to target_size rows, proportional to class frequency
    intent_counts = df["intent"].value_counts()
    per_class = max(1, target_size // len(intent_counts))

    sampled_parts = []
    for intent, count in intent_counts.items():
        n = min(per_class, count)
        subset = df[df["intent"] == intent].sample(
            n=n, random_state=RANDOM_SEED
        )
        sampled_parts.append(subset)

    golden_df = pd.concat(sampled_parts).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    # Limit to target_size
    golden_df = golden_df.iloc[:target_size].copy()

    # Add ID column
    golden_df.insert(0, "id", range(1, len(golden_df) + 1))

    # Rename for output clarity
    golden_df = golden_df.rename(columns={"cleaned_text": "message"})

    # Assign escalation labels
    golden_df["expected_action"] = golden_df.apply(assign_escalation_label, axis=1)

    # Select and order columns
    output_cols = [
        "id", "message", "intent", "expected_action",
        "needs_review", "brand_response",
    ]
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in golden_df.columns]
    golden_df = golden_df[output_cols]

    # Save
    ensure_dir(PROCESSED_DATA_DIR)
    golden_df.to_csv(GOLDEN_EVAL_PATH, index=False)
    logger.info("Golden evaluation set saved to %s (%d rows).", GOLDEN_EVAL_PATH, len(golden_df))

    # Print summary
    print("\n=== GOLDEN EVALUATION SET SUMMARY ===")
    print(f"Total examples : {len(golden_df)}")
    print(f"\nIntent distribution:\n{golden_df['intent'].value_counts()}")
    print(f"\nEscalation distribution:\n{golden_df['expected_action'].value_counts()}")
    print(f"\nExamples needing human review: {golden_df['needs_review'].sum()}")
    print(f"\n⚠️  IMPORTANT: Please open {GOLDEN_EVAL_PATH} and manually verify")
    print("   rows where needs_review=True before using this as ground truth.")

    return golden_df


def save_intent_definitions() -> None:
    """Save the intent taxonomy to data/processed/intent_definitions.json."""
    ensure_dir(PROCESSED_DATA_DIR)
    save_json(INTENT_TAXONOMY, INTENT_DEFINITIONS_PATH)
    logger.info("Intent definitions saved to %s", INTENT_DEFINITIONS_PATH)


if __name__ == "__main__":
    import argparse
    setup_logging()
    parser = argparse.ArgumentParser(description="Create golden evaluation set")
    parser.add_argument("--nrows", type=int, default=None, help="Limit dataset rows for testing")
    parser.add_argument("--size", type=int, default=GOLDEN_SET_SIZE, help="Target golden set size")
    parser.add_argument("--brand", type=str, default=SELECTED_BRAND, help="Brand name")
    args = parser.parse_args()

    save_intent_definitions()
    create_golden_set(nrows=args.nrows, target_size=args.size, brand=args.brand)
