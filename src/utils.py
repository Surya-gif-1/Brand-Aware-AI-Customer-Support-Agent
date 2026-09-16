"""
utils.py
--------
Shared utility functions used across the project.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean formatter."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Save data as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logging.getLogger(__name__).info("Saved JSON to %s", path)


def load_json(path: Path) -> Any:
    """Load JSON from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def truncate(text: str, max_chars: int = 120) -> str:
    """Truncate text for display purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def check_api_keys() -> dict:
    """
    Check which LLM API keys are available.

    Returns dict: {provider: available (bool)}
    """
    from src.config import GROQ_API_KEY, GOOGLE_API_KEY
    return {
        "groq": bool(GROQ_API_KEY),
        "gemini": bool(GOOGLE_API_KEY),
    }


def print_banner(title: str, width: int = 60) -> None:
    """Print a simple console banner."""
    line = "─" * width
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")
