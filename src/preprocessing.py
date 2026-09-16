"""
preprocessing.py
----------------
Text cleaning and feature engineering for customer support tweets.

Design choices:
  - Preserve as much meaning as possible (do NOT strip negations, brand names, etc.)
  - Remove Twitter-specific noise: @mentions, URLs, RT prefix
  - Normalise whitespace and encoding
  - Keep original text for response display / evaluation
"""

from __future__ import annotations

import re
import unicodedata
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns (compiled once at import time)
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")        # keep the word, drop the #
_RT_RE = re.compile(r"^RT\s*:?\s*", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ASCII_CHARS_RE = re.compile(r"[^\x00-\x7F]+")


def clean_text(text: str, *, keep_hashtag_words: bool = True) -> str:
    """
    Clean a single tweet.

    Steps (in order):
      1. Unicode normalise (NFKD → ASCII)
      2. Remove RT prefix
      3. Remove URLs
      4. Remove @mentions
      5. Optionally keep hashtag words (strip the # symbol)
      6. Collapse whitespace
      7. Strip leading/trailing whitespace

    Parameters
    ----------
    text               : raw tweet string
    keep_hashtag_words : if True, '#refund' → 'refund'; if False, remove entirely

    Returns
    -------
    Cleaned string.  Returns empty string if input is None or empty.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode normalise
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # 2. RT prefix
    text = _RT_RE.sub("", text)

    # 3. URLs
    text = _URL_RE.sub("", text)

    # 4. @mentions
    text = _MENTION_RE.sub("", text)

    # 5. Hashtags
    if keep_hashtag_words:
        text = _HASHTAG_RE.sub(r"\1", text)   # '#refund' → 'refund'
    else:
        text = re.sub(r"#\w+", "", text)

    # 6. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text)

    # 7. Strip
    return text.strip()


def is_empty_after_cleaning(text: str) -> bool:
    """Return True if cleaned text is empty or consists only of punctuation/numbers."""
    cleaned = clean_text(text)
    return len(cleaned.strip(".,!?;:- ")) == 0


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "customer_text",
    response_col: Optional[str] = "brand_response",
) -> pd.DataFrame:
    """
    Apply cleaning to the text columns of a DataFrame.

    Adds columns:
      - cleaned_text  : cleaned customer tweet
      - cleaned_response (if response_col present): cleaned brand response

    Drops rows where cleaned_text is empty.
    Drops exact duplicate (cleaned_text, brand_author_id) pairs if column exists.

    Parameters
    ----------
    df           : conversation pairs DataFrame
    text_col     : name of the customer text column
    response_col : name of the brand response column (optional)

    Returns
    -------
    Cleaned DataFrame (original columns preserved).
    """
    df = df.copy()

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
        )

    logger.info("Preprocessing %d rows…", len(df))

    # Clean customer text
    df["cleaned_text"] = df[text_col].fillna("").apply(clean_text)

    # Drop rows where cleaned text is empty
    before = len(df)
    df = df[df["cleaned_text"].str.len() > 0].copy()
    logger.info("Dropped %d rows with empty cleaned text.", before - len(df))

    # Clean brand response
    if response_col and response_col in df.columns:
        df["cleaned_response"] = df[response_col].fillna("").apply(clean_text)

    # Drop exact duplicates on cleaned text
    before = len(df)
    df = df.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
    logger.info("Dropped %d duplicate rows.", before - len(df))

    logger.info("Preprocessing complete. %d rows remaining.", len(df))
    return df


def extract_features(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Add lightweight text features that may help classification or escalation.

    Features added:
      - char_count     : length of the cleaned text
      - word_count     : number of words
      - has_question   : whether the text contains a question mark
      - exclamation_count : number of exclamation marks (frustration signal)
    """
    df = df.copy()
    texts = df[text_col].fillna("")

    df["char_count"] = texts.str.len()
    df["word_count"] = texts.str.split().str.len()
    df["has_question"] = texts.str.contains(r"\?").astype(int)
    df["exclamation_count"] = texts.str.count(r"!")

    return df
