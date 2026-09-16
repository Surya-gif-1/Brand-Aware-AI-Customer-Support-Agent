"""
config.py
---------
Central configuration for the Hiver AI Support Agent.
All tuneable parameters and paths are defined here so other modules
never need to hard-code paths or thresholds.

LLM provider priority:
  1. Groq  (GROQ_API_KEY)    – fast, free tier available
  2. Gemini (GOOGLE_API_KEY) – fallback
  3. None                    – graceful degradation (no response generation)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level up from src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = _PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = _PROJECT_ROOT / "models"
RESULTS_DIR = _PROJECT_ROOT / "evaluation" / "results"

# Primary dataset file (user must place the full TWCS dataset here)
TWCS_CSV_PATH = RAW_DATA_DIR / "twcs.csv"

# Sample dataset (always present – shipped with the repo)
SAMPLE_CSV_PATH = SAMPLE_DATA_DIR / "sample_conversations.csv"

# Processed artefacts
INTENT_DEFINITIONS_PATH = PROCESSED_DATA_DIR / "intent_definitions.json"
GOLDEN_EVAL_PATH = PROCESSED_DATA_DIR / "golden_eval.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_conversations.csv"

# Model / index artefacts
FAISS_INDEX_PATH = MODELS_DIR / "retrieval.faiss"
FAISS_METADATA_PATH = MODELS_DIR / "retrieval_metadata.pkl"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_lr_model.pkl"
EMBEDDING_CLASSIFIER_PATH = MODELS_DIR / "embedding_classifier.pkl"

# ---------------------------------------------------------------------------
# Brand selection
# ---------------------------------------------------------------------------
# Change this to switch to a different brand.
# The value is matched case-insensitively against the 'author_id' column.
SELECTED_BRAND: str = "AmazonHelp"   # populated after dataset inspection

# Minimum number of labelled conversations required to use a brand
MIN_BRAND_CONVERSATIONS: int = 500

# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------
# Defined per-brand in data/processed/intent_definitions.json at runtime.
# This list is the fallback taxonomy used before the JSON is created.
DEFAULT_INTENTS = [
    "order_delivery",
    "refund_return",
    "account_login",
    "payment_billing",
    "technical_issue",
    "cancellation",
    "product_inquiry",
    "complaint_other",
]

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# Using a small, fast, well-tested model that runs on CPU without CUDA
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int = 3                        # number of historical examples to retrieve
MIN_SIMILARITY_SCORE: float = 0.30   # below this → low-confidence retrieval

# ---------------------------------------------------------------------------
# Escalation thresholds
# ---------------------------------------------------------------------------
CONFIDENCE_ESCALATION_THRESHOLD: float = 0.55  # escalate if intent confidence < this
SIMILARITY_ESCALATION_THRESHOLD: float = 0.35  # escalate if best retrieval < this

# Intents that always trigger escalation regardless of confidence
HIGH_RISK_INTENTS = {"complaint_other", "payment_billing"}

# Keywords that raise a soft escalation flag
ESCALATION_KEYWORDS = [
    "fraud", "scam", "stolen", "unauthorized", "legal", "lawsuit",
    "police", "threatening", "dangerous", "urgently", "emergency",
    "disabled", "hacked", "breach",
]

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Provider selection: 'groq' | 'gemini' | 'none'
def _select_llm_provider() -> str:
    if GROQ_API_KEY:
        return "groq"
    if GOOGLE_API_KEY:
        return "gemini"
    return "none"

LLM_PROVIDER: str = _select_llm_provider()

# Groq settings
GROQ_MODEL_NAME: str = "llama-3.1-8b-instant"   # fast, free-tier Groq model

# Gemini settings (fallback)
GEMINI_MODEL_NAME: str = "gemini-1.5-flash"

LLM_MAX_OUTPUT_TOKENS: int = 300
LLM_TEMPERATURE: float = 0.3

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
GOLDEN_SET_SIZE: int = 200          # target number of labelled examples
TRAIN_TEST_SPLIT_RATIO: float = 0.8 # fraction of golden set used for training baselines
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
