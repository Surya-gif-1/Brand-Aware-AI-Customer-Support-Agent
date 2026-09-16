"""
test_classifier.py — pytest tests for src/intent_classifier.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent_classifier import MajorityClassifier, TfidfLRClassifier


TRAIN_TEXTS = [
    "my order hasn't arrived",
    "I want a refund",
    "I can't log in",
    "why was I charged twice",
    "the app keeps crashing",
    "I want to cancel my subscription",
    "is this product available",
    "terrible service I'm very angry",
    "where is my package",
    "how do I return this item",
    "password reset isn't working",
    "my credit card was declined",
    "the website shows an error",
    "please cancel my order",
    "what are the product specs",
    "this is unacceptable service",
]
TRAIN_LABELS = [
    "order_delivery", "refund_return", "account_login", "payment_billing",
    "technical_issue", "cancellation", "product_inquiry", "complaint_other",
    "order_delivery", "refund_return", "account_login", "payment_billing",
    "technical_issue", "cancellation", "product_inquiry", "complaint_other",
]


def test_majority_classifier_predict():
    clf = MajorityClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    preds = clf.predict(["some random text", "another text"])
    assert len(preds) == 2
    assert all(isinstance(p, str) for p in preds)

def test_majority_classifier_all_same():
    clf = MajorityClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    preds = clf.predict(["a", "b", "c"])
    assert len(set(preds)) == 1   # all same class

def test_majority_predict_proba():
    clf = MajorityClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    result = clf.predict_proba(["test"])
    assert isinstance(result, list)
    assert "intent" in result[0]
    assert "confidence" in result[0]
    assert result[0]["confidence"] == 1.0

def test_tfidf_lr_predict():
    clf = TfidfLRClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    preds = clf.predict(["my package is delayed"])
    assert len(preds) == 1
    assert isinstance(preds[0], str)

def test_tfidf_lr_predict_proba():
    clf = TfidfLRClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    result = clf.predict_proba(["I want my money back"])
    assert len(result) == 1
    assert "intent" in result[0]
    assert 0.0 <= result[0]["confidence"] <= 1.0

def test_tfidf_lr_not_fitted_raises():
    import pytest
    clf = TfidfLRClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(["test"])

def test_majority_not_fitted_raises():
    import pytest
    clf = MajorityClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(["test"])
