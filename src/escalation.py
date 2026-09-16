"""
escalation.py
-------------
Transparent, rule-based escalation engine.

The engine decides whether a customer message should be:
  - AUTO-HANDLE : the agent generates a response
  - ESCALATE    : a human agent must intervene

Rules are applied in order; the first triggered rule wins.
All thresholds are configurable in config.py — they are engineering choices,
not statistically optimised values.

Decision output:
    {
        "decision":  "AUTO-HANDLE" | "ESCALATE",
        "reason":    "<human-readable explanation>",
        "rule":      "<rule identifier>",
    }
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from src.config import (
    CONFIDENCE_ESCALATION_THRESHOLD,
    SIMILARITY_ESCALATION_THRESHOLD,
    HIGH_RISK_INTENTS,
    ESCALATION_KEYWORDS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual rule functions
# ---------------------------------------------------------------------------

def _rule_low_confidence(
    confidence: float, threshold: float
) -> Optional[dict]:
    """Escalate if the classifier confidence is below the threshold."""
    if confidence < threshold:
        return {
            "decision": "ESCALATE",
            "reason": (
                f"Classifier confidence ({confidence:.2f}) is below the "
                f"threshold ({threshold:.2f}). The intent is ambiguous."
            ),
            "rule": "low_confidence",
        }
    return None


def _rule_high_risk_intent(intent: str) -> Optional[dict]:
    """Escalate if the intent is flagged as high-risk regardless of confidence."""
    if intent in HIGH_RISK_INTENTS:
        return {
            "decision": "ESCALATE",
            "reason": (
                f"Intent '{intent}' is classified as high-risk and requires "
                "human review to avoid financial or account-related errors."
            ),
            "rule": "high_risk_intent",
        }
    return None


def _rule_sensitive_keywords(text: str, keywords: list[str]) -> Optional[dict]:
    """Escalate if the message contains sensitive/urgent language."""
    text_lower = text.lower()
    found = [kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)]
    if found:
        return {
            "decision": "ESCALATE",
            "reason": (
                f"Message contains sensitive keyword(s) {found} that suggest "
                "urgency or legal/security risk."
            ),
            "rule": "sensitive_keywords",
        }
    return None


def _rule_low_retrieval_quality(
    best_similarity: float, threshold: float
) -> Optional[dict]:
    """Escalate if the best retrieved example is too dissimilar."""
    if best_similarity < threshold:
        return {
            "decision": "ESCALATE",
            "reason": (
                f"Best retrieval similarity ({best_similarity:.2f}) is below "
                f"threshold ({threshold:.2f}). No sufficiently similar historical "
                "example was found to ground a safe automated response."
            ),
            "rule": "low_retrieval_quality",
        }
    return None


def _rule_empty_retrieval() -> dict:
    """Escalate if no historical examples were retrieved at all."""
    return {
        "decision": "ESCALATE",
        "reason": "No historical examples were retrieved for this query.",
        "rule": "empty_retrieval",
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def decide_escalation(
    text: str,
    intent: str,
    confidence: float,
    historical_examples: list[dict],
    *,
    confidence_threshold: float = CONFIDENCE_ESCALATION_THRESHOLD,
    similarity_threshold: float = SIMILARITY_ESCALATION_THRESHOLD,
    escalation_keywords: list[str] = ESCALATION_KEYWORDS,
    high_risk_intents: set = HIGH_RISK_INTENTS,
) -> dict:
    """
    Apply escalation rules in priority order and return the first match.

    Priority:
      1. Sensitive keywords (safety-first)
      2. High-risk intent
      3. Low classifier confidence
      4. Empty retrieval results
      5. Low retrieval similarity

    If no rule triggers, the decision is AUTO-HANDLE.

    Parameters
    ----------
    text                : cleaned customer message
    intent              : predicted intent string
    confidence          : classifier confidence score (0–1)
    historical_examples : list of retrieved example dicts from Retriever
    confidence_threshold: below this → ESCALATE
    similarity_threshold: below best similarity → ESCALATE

    Returns
    -------
    dict with keys: decision, reason, rule
    """
    # Rule 1: Sensitive keywords
    result = _rule_sensitive_keywords(text, escalation_keywords)
    if result:
        logger.info("Escalation rule triggered: %s", result["rule"])
        return result

    # Rule 2: High-risk intent
    result = _rule_high_risk_intent(intent)
    if result:
        logger.info("Escalation rule triggered: %s", result["rule"])
        return result

    # Rule 3: Low confidence
    result = _rule_low_confidence(confidence, confidence_threshold)
    if result:
        logger.info("Escalation rule triggered: %s", result["rule"])
        return result

    # Rule 4: No retrieved examples
    if not historical_examples:
        result = _rule_empty_retrieval()
        logger.info("Escalation rule triggered: %s", result["rule"])
        return result

    # Rule 5: Low retrieval quality
    best_similarity = max(
        ex.get("similarity_score", 0.0) for ex in historical_examples
    )
    result = _rule_low_retrieval_quality(best_similarity, similarity_threshold)
    if result:
        logger.info("Escalation rule triggered: %s", result["rule"])
        return result

    # No rule triggered → auto-handle
    return {
        "decision": "AUTO-HANDLE",
        "reason": (
            f"Intent '{intent}' classified with confidence {confidence:.2f} "
            f"(≥{confidence_threshold}). "
            f"Best retrieval similarity: {best_similarity:.2f} "
            f"(≥{similarity_threshold}). "
            "No sensitive keywords or high-risk signals detected."
        ),
        "rule": "none",
    }
