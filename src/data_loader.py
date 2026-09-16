"""
data_loader.py
--------------
Loads and validates the Customer Support on Twitter (TWCS) dataset.

Supports:
  - Full dataset (twcs.csv in data/raw/)
  - Packaged sample dataset (data/sample/)

Schema detection:
  - Inspects columns present in the CSV
  - Maps them to canonical names (tweet_id, author_id, text, in_response_to_tweet_id)
  - Raises a clear SchemaError if required columns are missing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    TWCS_CSV_PATH,
    SAMPLE_CSV_PATH,
    SELECTED_BRAND,
    MIN_BRAND_CONVERSATIONS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column aliases – maps possible raw column names → canonical names
# ---------------------------------------------------------------------------
_COLUMN_ALIASES: dict[str, list[str]] = {
    "tweet_id":                 ["tweet_id", "id", "TweetId", "tweet_Id"],
    "author_id":                ["author_id", "AuthorId", "author", "handle"],
    "text":                     ["text", "content", "tweet_text", "message", "body"],
    "in_response_to_tweet_id":  [
        "in_response_to_tweet_id",
        "in_response_to",
        "response_to",
        "reply_to",
        "inbound",
    ],
    "created_at":               ["created_at", "CreatedAt", "timestamp", "time"],
    "inbound":                  ["inbound"],
}

_REQUIRED_CANONICAL = {"tweet_id", "author_id", "text", "in_response_to_tweet_id"}


class SchemaError(ValueError):
    """Raised when the dataset does not have the expected columns."""


def _detect_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect column aliases and rename them to canonical names.

    Raises SchemaError if any required canonical column cannot be mapped.
    """
    raw_cols_lower = {c.lower(): c for c in df.columns}
    rename_map: dict[str, str] = {}

    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue  # already present
        for alias in aliases:
            if alias.lower() in raw_cols_lower:
                rename_map[raw_cols_lower[alias.lower()]] = canonical
                break

    df = df.rename(columns=rename_map)

    missing = _REQUIRED_CANONICAL - set(df.columns)
    if missing:
        raise SchemaError(
            f"Dataset is missing required columns: {missing}.\n"
            f"Detected columns: {list(df.columns)}\n"
            "Please check data/README.md for the expected schema."
        )

    logger.info("Schema detection succeeded. Columns: %s", list(df.columns))
    return df


def load_raw_dataset(path: Optional[Path] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the TWCS CSV from *path* (defaults to config.TWCS_CSV_PATH).

    Parameters
    ----------
    path   : CSV file to load; falls back to the sample dataset if the full
             file is not found.
    nrows  : If set, load only this many rows (useful for quick testing).

    Returns
    -------
    DataFrame with canonical column names.
    """
    if path is None:
        path = TWCS_CSV_PATH

    # Check if the nested CSV exists (in case the user accidentally extracted twcs.csv as a folder)
    nested_path = Path(path) / "twcs" / "twcs.csv"
    if nested_path.is_file():
        path = nested_path

    if not Path(path).is_file():
        logger.warning(
            "Full dataset file not found at %s. Falling back to sample dataset at %s.",
            path,
            SAMPLE_CSV_PATH,
        )
        path = SAMPLE_CSV_PATH
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Neither the full dataset ({TWCS_CSV_PATH}) "
                f"nor the sample dataset ({SAMPLE_CSV_PATH}) could be found.\n"
                "Please place twcs.csv in data/raw/."
            )

    logger.info("Loading dataset from %s (nrows=%s)…", path, nrows)
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df = _detect_and_rename(df)
    logger.info("Loaded %d rows from %s.", len(df), path)
    return df


def build_conversation_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join tweets with their replies to form (customer_text, brand_response) pairs.

    Returns a DataFrame with columns:
        customer_tweet_id, customer_text, author_id (customer),
        brand_tweet_id, brand_response, brand_author_id,
        created_at (if present)
    """
    df = df.copy()

    # The TWCS dataset uses:
    #   inbound == True  → customer tweet
    #   inbound == False → brand response
    # If 'inbound' column is absent we infer from in_response_to_tweet_id
    if "inbound" in df.columns:
        # Coerce to bool safely
        df["inbound"] = df["inbound"].map(
            lambda v: str(v).strip().lower() in {"true", "1", "yes"}
        )
        customer_df = df[df["inbound"]].copy()
        brand_df = df[~df["inbound"]].copy()
    else:
        # Inbound heuristic: customer tweets have in_response_to_tweet_id that
        # points to a brand tweet, and brand tweets reply to customer tweets.
        # Simplest safe approach: treat rows whose author_id matches the brand as brand.
        customer_df = df[df["author_id"].str.lower() != SELECTED_BRAND.lower()].copy()
        brand_df = df[df["author_id"].str.lower() == SELECTED_BRAND.lower()].copy()

    # Brand responses reference the customer tweet via in_response_to_tweet_id
    brand_df = brand_df.rename(
        columns={
            "tweet_id": "brand_tweet_id",
            "author_id": "brand_author_id",
            "text": "brand_response",
        }
    )
    customer_df = customer_df.rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "author_id": "customer_author_id",
            "text": "customer_text",
        }
    )

    merged = brand_df.merge(
        customer_df[["customer_tweet_id", "customer_text", "customer_author_id"]],
        left_on="in_response_to_tweet_id",
        right_on="customer_tweet_id",
        how="inner",
    )

    logger.info(
        "Built %d (customer, brand) conversation pairs.", len(merged)
    )
    return merged


def load_brand_conversations(
    brand: str = SELECTED_BRAND,
    path: Optional[Path] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the full dataset and return only conversations for *brand*.

    Returns DataFrame of (customer_text, brand_response) pairs.
    """
    df = load_raw_dataset(path=path, nrows=nrows)
    pairs = build_conversation_pairs(df)

    brand_pairs = pairs[
        pairs["brand_author_id"].str.lower() == brand.lower()
    ].copy()

    if len(brand_pairs) < MIN_BRAND_CONVERSATIONS:
        logger.warning(
            "Brand '%s' has only %d conversation pairs (minimum is %d). "
            "Consider choosing a different brand.",
            brand,
            len(brand_pairs),
            MIN_BRAND_CONVERSATIONS,
        )

    logger.info(
        "Loaded %d conversation pairs for brand '%s'.", len(brand_pairs), brand
    )
    return brand_pairs.reset_index(drop=True)


def list_top_brands(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """Return top_n brands by number of responses (for brand selection)."""
    if "inbound" in df.columns:
        brand_df = df[~df["inbound"]]
    else:
        brand_df = df
    return brand_df["author_id"].value_counts().head(top_n)
