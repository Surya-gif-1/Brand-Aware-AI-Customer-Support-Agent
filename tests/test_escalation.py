"""
test_escalation.py — pytest tests for src/escalation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.escalation import decide_escalation


GOOD_EXAMPLES = [
    {"customer_text": "where is my order", "brand_response": "Please DM us", "similarity_score": 0.75},
    {"customer_text": "my package is late",  "brand_response": "We'll look into it", "similarity_score": 0.70},
    {"customer_text": "how do I return this","brand_response": "Go to Your Orders",  "similarity_score": 0.68},
]


def test_auto_handle_high_confidence():
    result = decide_escalation(
        text="my order hasn't arrived",
        intent="order_delivery",
        confidence=0.85,
        historical_examples=GOOD_EXAMPLES,
    )
    assert result["decision"] == "AUTO-HANDLE"
    assert "intent" in result["reason"] or "confidence" in result["reason"]

def test_escalate_low_confidence():
    result = decide_escalation(
        text="something unclear",
        intent="order_delivery",
        confidence=0.30,    # below threshold
        historical_examples=GOOD_EXAMPLES,
    )
    assert result["decision"] == "ESCALATE"
    assert result["rule"] == "low_confidence"

def test_escalate_sensitive_keyword():
    result = decide_escalation(
        text="my account was hacked I think there is fraud",
        intent="account_login",
        confidence=0.90,
        historical_examples=GOOD_EXAMPLES,
    )
    assert result["decision"] == "ESCALATE"
    assert result["rule"] == "sensitive_keywords"

def test_escalate_high_risk_intent():
    result = decide_escalation(
        text="I was charged the wrong amount",
        intent="payment_billing",   # HIGH_RISK_INTENTS
        confidence=0.90,
        historical_examples=GOOD_EXAMPLES,
    )
    assert result["decision"] == "ESCALATE"
    assert result["rule"] == "high_risk_intent"

def test_escalate_empty_retrieval():
    result = decide_escalation(
        text="my order is late",
        intent="order_delivery",
        confidence=0.80,
        historical_examples=[],   # no examples
    )
    assert result["decision"] == "ESCALATE"
    assert result["rule"] == "empty_retrieval"

def test_escalate_low_similarity():
    low_sim_examples = [{"customer_text": "x", "brand_response": "y", "similarity_score": 0.20}]
    result = decide_escalation(
        text="my order is late",
        intent="order_delivery",
        confidence=0.80,
        historical_examples=low_sim_examples,
    )
    assert result["decision"] == "ESCALATE"
    assert result["rule"] == "low_retrieval_quality"

def test_result_has_required_keys():
    result = decide_escalation(
        text="my order is late",
        intent="order_delivery",
        confidence=0.85,
        historical_examples=GOOD_EXAMPLES,
    )
    assert "decision" in result
    assert "reason" in result
    assert "rule" in result
    assert result["decision"] in ("AUTO-HANDLE", "ESCALATE")
