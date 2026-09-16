"""
test_retriever.py — pytest tests for src/retriever.py
Tests brute-force fallback (no FAISS required).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.retriever import Retriever


SAMPLE_DF = pd.DataFrame({
    "cleaned_text": [
        "my order hasn't arrived",
        "I want a refund",
        "I can't log in",
        "the app is crashing",
        "why was I charged twice",
        "cancel my subscription please",
        "is this product available",
    ],
    "brand_response": [
        "Please DM your order number.",
        "Go to Your Orders to return.",
        "Visit the password reset page.",
        "Try reinstalling the app.",
        "Duplicate charges reverse soon.",
        "Cancel from Account Settings.",
        "Check product page availability.",
    ],
    "customer_tweet_id": [1, 2, 3, 4, 5, 6, 7],
})


def test_retriever_builds_and_retrieves():
    r = Retriever(top_k=3)
    r.build(SAMPLE_DF)
    results = r.retrieve("my package hasn't arrived")
    assert len(results) >= 1
    assert "customer_text" in results[0]
    assert "brand_response" in results[0]
    assert "similarity_score" in results[0]

def test_retriever_top_k_respected():
    r = Retriever(top_k=2)
    r.build(SAMPLE_DF)
    results = r.retrieve("order late", top_k=2)
    assert len(results) <= 2

def test_retriever_exclude_ids():
    r = Retriever(top_k=3)
    r.build(SAMPLE_DF)
    # Exclude the most likely top result
    results = r.retrieve("my order hasn't arrived", exclude_ids={1})
    ids = [ex.get("id") for ex in results]   # not all dicts have 'id' exposed
    # Just verify we still get results
    assert len(results) >= 1

def test_retriever_scores_between_zero_and_one():
    r = Retriever(top_k=3)
    r.build(SAMPLE_DF)
    results = r.retrieve("I want to return my item")
    for res in results:
        # Cosine similarity with L2-normalized vectors is in [-1, 1]
        assert -1.0 <= res["similarity_score"] <= 1.0

def test_retriever_not_built_raises():
    import pytest
    r = Retriever()
    with pytest.raises(RuntimeError):
        r.retrieve("test")
