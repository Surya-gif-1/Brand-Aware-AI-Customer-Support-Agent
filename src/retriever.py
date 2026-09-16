"""
retriever.py
------------
FAISS-based semantic retrieval of historically similar customer-support conversations.

Architecture:
    1. Build a FAISS index from brand-filtered conversation pairs.
    2. At query time, embed the customer message and find Top-K nearest neighbours.
    3. Return the matched (customer_text, brand_response, similarity_score) triples.

Anti-leakage:
    - An exclude_ids set is supported so evaluation examples are never retrieved
      from themselves (no self-retrieval).
    - The retriever is built exclusively from the training split of conversations.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    TOP_K,
    MIN_SIMILARITY_SCORE,
)

logger = logging.getLogger(__name__)


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. Run: pip install sentence-transformers"
        )


def _try_import_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


class Retriever:
    """
    Semantic retrieval engine over historical brand conversations.

    Supports FAISS for fast similarity search with a fallback to brute-force
    cosine similarity when FAISS is not installed.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        top_k: int = TOP_K,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._encoder = None
        self._index = None          # FAISS index or None
        self._embeddings: Optional[np.ndarray] = None   # fallback
        self._metadata: list[dict] = []   # parallel list of {customer_text, brand_response, id}
        self._is_built = False
        self._use_faiss = _try_import_faiss() is not None

    @property
    def encoder(self):
        if self._encoder is None:
            logger.info("Loading sentence-transformer '%s'…", self.model_name)
            self._encoder = _load_sentence_transformer(self.model_name)
        return self._encoder

    def _embed(self, texts: list[str]) -> np.ndarray:
        embs = self.encoder.encode(texts, show_progress_bar=len(texts) > 100, batch_size=64)
        # Normalise to unit length for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (embs / norms).astype("float32")

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> "Retriever":
        """
        Build the retrieval index from a DataFrame of conversation pairs.

        Required columns: cleaned_text, brand_response (or customer_text + brand_response)
        Optional column : customer_tweet_id (used for exclude_ids)
        """
        text_col = "cleaned_text" if "cleaned_text" in df.columns else "customer_text"
        resp_col = "brand_response"

        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in retriever DataFrame.")
        if resp_col not in df.columns:
            raise ValueError(f"Column '{resp_col}' not found in retriever DataFrame.")

        texts = df[text_col].fillna("").tolist()
        responses = df[resp_col].fillna("").tolist()
        ids = (
            df["customer_tweet_id"].tolist()
            if "customer_tweet_id" in df.columns
            else list(range(len(texts)))
        )

        logger.info("Building retrieval index for %d documents…", len(texts))
        embeddings = self._embed(texts)

        # Store metadata
        self._metadata = [
            {"id": ids[i], "customer_text": texts[i], "brand_response": responses[i]}
            for i in range(len(texts))
        ]

        if self._use_faiss:
            faiss = _try_import_faiss()
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)   # Inner product = cosine for unit vectors
            self._index.add(embeddings)
            logger.info("FAISS index built with %d vectors (dim=%d).", len(texts), dim)
        else:
            logger.warning("FAISS not available. Using brute-force cosine search (slower).")
            self._embeddings = embeddings

        self._is_built = True
        return self

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        exclude_ids: Optional[set] = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most similar historical conversations.

        Parameters
        ----------
        query      : cleaned customer message
        top_k      : override the default TOP_K
        exclude_ids: set of ids to exclude (prevents self-retrieval in evaluation)

        Returns
        -------
        List of dicts with keys:
            customer_text, brand_response, similarity_score
        Ordered by descending similarity.
        """
        if not self._is_built:
            raise RuntimeError("Retriever is not built. Call build() or load() first.")

        k = top_k or self.top_k
        exclude_ids = exclude_ids or set()

        query_emb = self._embed([query])   # shape (1, dim)

        if self._use_faiss and self._index is not None:
            # Retrieve extra candidates to allow filtering excluded ids
            search_k = min(k + len(exclude_ids) + 5, len(self._metadata))
            scores, indices = self._index.search(query_emb, search_k)
            scores = scores[0].tolist()
            indices = indices[0].tolist()
        else:
            # Brute force
            sims = (self._embeddings @ query_emb.T).flatten()
            top_indices = np.argsort(sims)[::-1][: k + len(exclude_ids) + 5]
            scores = sims[top_indices].tolist()
            indices = top_indices.tolist()

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            if meta["id"] in exclude_ids:
                continue
            results.append({
                "customer_text": meta["customer_text"],
                "brand_response": meta["brand_response"],
                "similarity_score": float(score),
            })
            if len(results) >= k:
                break

        return results

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: Optional[int] = None,
        exclude_ids_list: Optional[list[set]] = None,
    ) -> list[list[dict]]:
        """Batch retrieval for evaluation."""
        if exclude_ids_list is None:
            exclude_ids_list = [set()] * len(queries)
        return [
            self.retrieve(q, top_k, excl)
            for q, excl in zip(queries, exclude_ids_list)
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = FAISS_METADATA_PATH,
    ) -> None:
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if self._use_faiss and self._index is not None:
            faiss = _try_import_faiss()
            faiss.write_index(self._index, str(index_path))
        else:
            np.save(str(index_path).replace(".faiss", "_emb.npy"), self._embeddings)

        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "metadata": self._metadata,
                    "model_name": self.model_name,
                    "top_k": self.top_k,
                    "use_faiss": self._use_faiss,
                },
                f,
            )
        logger.info("Retriever saved to %s / %s", index_path, meta_path)

    @classmethod
    def load(
        cls,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = FAISS_METADATA_PATH,
    ) -> "Retriever":
        meta_path = Path(meta_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Retriever metadata not found at {meta_path}.")

        with open(meta_path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(model_name=payload["model_name"], top_k=payload["top_k"])
        obj._metadata = payload["metadata"]
        obj._use_faiss = payload.get("use_faiss", True)

        index_path = Path(index_path)
        if obj._use_faiss and index_path.exists():
            faiss = _try_import_faiss()
            obj._index = faiss.read_index(str(index_path))
        else:
            npy_path = str(index_path).replace(".faiss", "_emb.npy")
            if Path(npy_path).exists():
                obj._embeddings = np.load(npy_path)

        obj._is_built = True
        logger.info("Retriever loaded from %s.", meta_path)
        return obj

    def __repr__(self) -> str:
        return (
            f"Retriever(model='{self.model_name}', "
            f"top_k={self.top_k}, "
            f"built={self._is_built}, "
            f"n_docs={len(self._metadata)})"
        )
