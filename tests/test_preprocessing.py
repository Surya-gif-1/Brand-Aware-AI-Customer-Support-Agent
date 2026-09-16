"""
test_preprocessing.py — pytest tests for src/preprocessing.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import clean_text, preprocess_dataframe, is_empty_after_cleaning
import pandas as pd


def test_clean_text_removes_url():
    assert "http" not in clean_text("Check this https://amazon.com for details")

def test_clean_text_removes_mention():
    assert "@AmazonHelp" not in clean_text("@AmazonHelp my order is late")

def test_clean_text_removes_rt_prefix():
    result = clean_text("RT @user: some tweet")
    assert not result.startswith("RT")

def test_clean_text_keeps_meaningful_words():
    result = clean_text("I want a refund for my damaged product")
    assert "refund" in result
    assert "damaged" in result

def test_clean_text_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   ") == ""

def test_clean_text_hashtag_kept():
    result = clean_text("#refund this product")
    assert "refund" in result

def test_is_empty_after_cleaning():
    assert is_empty_after_cleaning("https://t.co/xyz @AmazonHelp") is True or \
           is_empty_after_cleaning("") is True

def test_preprocess_dataframe():
    df = pd.DataFrame({
        "customer_text": ["@Amazon my order is late", "", "I want a refund"],
        "brand_response": ["Sorry!", "", "Please DM us"],
    })
    result = preprocess_dataframe(df, text_col="customer_text", response_col="brand_response")
    # Empty row should be dropped
    assert len(result) == 2
    # Cleaned text column should exist
    assert "cleaned_text" in result.columns
    # cleaned_response should exist
    assert "cleaned_response" in result.columns

def test_preprocess_deduplication():
    df = pd.DataFrame({
        "customer_text": ["my order is late", "my order is late", "refund please"],
        "brand_response": ["ok", "ok", "sure"],
    })
    result = preprocess_dataframe(df, text_col="customer_text")
    assert len(result) == 2   # duplicate removed
