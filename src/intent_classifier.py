"""
intent_classifier.py
--------------------
Three classifiers in one module, listed in increasing sophistication:

  1. MajorityClassifier  — trivial baseline (always predicts most-frequent class)
  2. TfidfLRClassifier   — TF-IDF + Logistic Regression (simple ML baseline)
  3. EmbeddingClassifier — sentence-transformer embeddings + LR (final system)

All classifiers share the same interface:
    classifier.fit(texts, labels)
    classifier.predict(texts) → list[str]
    classifier.predict_proba(texts) → list[dict]   # {"intent": str, "confidence": float}
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.config import (
    EMBEDDING_MODEL_NAME,
    TFIDF_MODEL_PATH,
    EMBEDDING_CLASSIFIER_PATH,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sentence_transformer():
    """Lazy-load SentenceTransformer to avoid slow startup when not needed."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for EmbeddingClassifier. "
            "Run: pip install sentence-transformers"
        )


# ---------------------------------------------------------------------------
# 1. Trivial Majority Classifier
# ---------------------------------------------------------------------------

class MajorityClassifier:
    """
    Trivial baseline: predicts the most frequent class for every input.

    This is useful to understand the performance floor implied by class
    imbalance in the golden evaluation set.
    """

    def __init__(self):
        self.majority_class_: Optional[str] = None
        self.class_distribution_: dict = {}

    def fit(self, texts: list[str], labels: list[str]) -> "MajorityClassifier":
        from collections import Counter
        counts = Counter(labels)
        self.majority_class_ = counts.most_common(1)[0][0]
        self.class_distribution_ = dict(counts)
        logger.info("MajorityClassifier fitted. Majority class: '%s'", self.majority_class_)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        if self.majority_class_ is None:
            raise RuntimeError("MajorityClassifier has not been fitted. Call fit() first.")
        return [self.majority_class_] * len(texts)

    def predict_proba(self, texts: list[str]) -> list[dict]:
        predictions = self.predict(texts)
        return [{"intent": p, "confidence": 1.0} for p in predictions]

    def __repr__(self) -> str:
        return f"MajorityClassifier(majority_class='{self.majority_class_}')"


# ---------------------------------------------------------------------------
# 2. TF-IDF + Logistic Regression Classifier
# ---------------------------------------------------------------------------

class TfidfLRClassifier:
    """
    Simple ML baseline: TF-IDF features + Logistic Regression.

    Deliberately kept simple to serve as a meaningful upper bound on what
    keyword-based approaches can achieve.
    """

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple = (1, 2),
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=2,
        )
        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=42,
        )
        self.label_encoder = LabelEncoder()
        self._is_fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfLRClassifier":
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)
        self.classifier.fit(X, y)
        self._is_fitted = True
        logger.info(
            "TfidfLRClassifier fitted on %d examples, %d classes.",
            len(texts),
            len(self.label_encoder.classes_),
        )
        return self

    def predict(self, texts: list[str]) -> list[str]:
        if not self._is_fitted:
            raise RuntimeError("TfidfLRClassifier has not been fitted. Call fit() first.")
        X = self.vectorizer.transform(texts)
        y_pred = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred).tolist()

    def predict_proba(self, texts: list[str]) -> list[dict]:
        if not self._is_fitted:
            raise RuntimeError("TfidfLRClassifier has not been fitted. Call fit() first.")
        X = self.vectorizer.transform(texts)
        proba = self.classifier.predict_proba(X)
        results = []
        for p in proba:
            best_idx = int(np.argmax(p))
            results.append({
                "intent": self.label_encoder.classes_[best_idx],
                "confidence": float(p[best_idx]),
            })
        return results

    def save(self, path: Path = TFIDF_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("TfidfLRClassifier saved to %s", path)

    @classmethod
    def load(cls, path: Path = TFIDF_MODEL_PATH) -> "TfidfLRClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("TfidfLRClassifier loaded from %s", path)
        return obj

    def __repr__(self) -> str:
        return (
            f"TfidfLRClassifier(fitted={self._is_fitted}, "
            f"classes={list(getattr(self.label_encoder, 'classes_', []))})"
        )


# ---------------------------------------------------------------------------
# 3. Sentence-Embedding + Logistic Regression Classifier  (final system)
# ---------------------------------------------------------------------------

class EmbeddingClassifier:
    """
    Final intent classifier using sentence-transformer embeddings + LogisticRegression.

    Why this architecture:
    - Sentence transformers capture semantic similarity better than TF-IDF.
    - A simple LR head is lightweight, interpretable, and fast to retrain.
    - Confidence scores from LR are not calibrated probabilities but provide
      a useful relative signal for the escalation engine.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, C: float = 1.0):
        self.model_name = model_name
        self._encoder = None          # lazy-loaded
        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.label_encoder = LabelEncoder()
        self._is_fitted = False

    @property
    def encoder(self):
        if self._encoder is None:
            logger.info("Loading sentence-transformer model '%s'…", self.model_name)
            self._encoder = _load_sentence_transformer()
        return self._encoder

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts, show_progress_bar=False, batch_size=64)

    def fit(self, texts: list[str], labels: list[str]) -> "EmbeddingClassifier":
        logger.info("Computing embeddings for %d training examples…", len(texts))
        X = self._embed(texts)
        y = self.label_encoder.fit_transform(labels)
        self.classifier.fit(X, y)
        self._is_fitted = True
        logger.info(
            "EmbeddingClassifier fitted. Classes: %s",
            list(self.label_encoder.classes_),
        )
        return self

    def predict(self, texts: list[str]) -> list[str]:
        if not self._is_fitted:
            raise RuntimeError("EmbeddingClassifier has not been fitted. Call fit() first.")
        X = self._embed(texts)
        y_pred = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred).tolist()

    def predict_proba(self, texts: list[str]) -> list[dict]:
        if not self._is_fitted:
            raise RuntimeError("EmbeddingClassifier has not been fitted. Call fit() first.")
        X = self._embed(texts)
        proba = self.classifier.predict_proba(X)
        results = []
        for p in proba:
            best_idx = int(np.argmax(p))
            results.append({
                "intent": self.label_encoder.classes_[best_idx],
                "confidence": float(p[best_idx]),
            })
        return results

    def predict_single(self, text: str) -> dict:
        """Convenience method for single-text inference."""
        return self.predict_proba([text])[0]

    def save(self, path: Path = EMBEDDING_CLASSIFIER_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Do NOT pickle the SentenceTransformer model — it's large and has its own cache.
        # Save only the LR head and label encoder.
        payload = {
            "model_name": self.model_name,
            "classifier": self.classifier,
            "label_encoder": self.label_encoder,
            "is_fitted": self._is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("EmbeddingClassifier saved to %s", path)

    @classmethod
    def load(cls, path: Path = EMBEDDING_CLASSIFIER_PATH) -> "EmbeddingClassifier":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(model_name=payload["model_name"])
        obj.classifier = payload["classifier"]
        obj.label_encoder = payload["label_encoder"]
        obj._is_fitted = payload["is_fitted"]
        logger.info("EmbeddingClassifier loaded from %s", path)
        return obj

    def __repr__(self) -> str:
        return (
            f"EmbeddingClassifier(model='{self.model_name}', "
            f"fitted={self._is_fitted}, "
            f"classes={list(getattr(self.label_encoder, 'classes_', []))})"
        )
