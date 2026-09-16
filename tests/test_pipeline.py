"""
test_pipeline.py — pytest integration tests for src/pipeline.py
LLM calls are mocked so these tests run without an API key.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.pipeline import SupportPipeline
from src.intent_classifier import EmbeddingClassifier, TfidfLRClassifier
from src.retriever import Retriever


# ── Test data ─────────────────────────────────────────────────────────────────
SAMPLE_DF = pd.DataFrame({
    "cleaned_text": [
        "my order hasn't arrived yet", "I want a refund for damaged item",
        "can't log in password reset", "why was I charged twice",
        "app keeps crashing on phone", "please cancel my subscription",
        "is this product available in red", "terrible service very angry",
        "where is my package shipment", "how do I return this product",
        "my account is locked out", "unauthorized charge on my card",
        "website error 500 when checkout", "cancel my prime membership",
        "product warranty question help", "I am very disappointed today",
    ],
    "brand_response": [
        "Please DM us your order number.",
        "Go to Your Orders and select Return.",
        "Visit amazon.com/forgot-password.",
        "Duplicate charges reverse in 24-48h.",
        "Try clearing cache or reinstalling.",
        "Cancel from Account Settings.",
        "Check product page for availability.",
        "We're sorry about your experience.",
        "Check tracking from Your Orders.",
        "Initiate return from Your Orders.",
        "DM us for account recovery.",
        "We're investigating this charge.",
        "Our team is working on the issue.",
        "Cancel from Prime Membership page.",
        "Devices have 1-year warranty.",
        "We sincerely apologize.",
    ],
    "intent": [
        "order_delivery", "refund_return", "account_login", "payment_billing",
        "technical_issue", "cancellation", "product_inquiry", "complaint_other",
        "order_delivery", "refund_return", "account_login", "payment_billing",
        "technical_issue", "cancellation", "product_inquiry", "complaint_other",
    ],
    "customer_tweet_id": list(range(1, 17)),
})


@patch("src.response_generator.generate_response")
def test_pipeline_build_and_run(mock_gen):
    """Test that pipeline builds and returns expected keys."""
    mock_gen.return_value = {
        "response": "Thank you for contacting us.",
        "provider": "mock",
        "status": "success",
        "error": None,
    }
    pipeline = SupportPipeline.build(SAMPLE_DF, label_col="intent", train_ratio=0.75)

    result = pipeline.run("My order hasn't arrived yet", generate=True)

    required_keys = {
        "message", "cleaned_message", "intent", "confidence",
        "decision", "decision_reason", "decision_rule",
        "historical_examples", "response", "response_status", "brand",
    }
    assert required_keys.issubset(set(result.keys()))
    assert isinstance(result["intent"], str)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["decision"] in ("AUTO-HANDLE", "ESCALATE")


def test_pipeline_empty_message():
    pipeline = SupportPipeline.build(SAMPLE_DF, label_col="intent", train_ratio=0.75)
    result = pipeline.run("")
    assert result["decision"] == "ESCALATE"
    assert result["intent"] == "unknown"


@patch("src.response_generator.generate_response")
def test_pipeline_batch(mock_gen):
    mock_gen.return_value = {
        "response": "Thank you.", "provider": "mock",
        "status": "success", "error": None,
    }
    pipeline = SupportPipeline.build(SAMPLE_DF, label_col="intent", train_ratio=0.75)
    msgs = ["order not arrived", "want refund", "can't login"]
    results = pipeline.run_batch(msgs, generate=False)
    assert len(results) == 3
    assert all("intent" in r for r in results)
