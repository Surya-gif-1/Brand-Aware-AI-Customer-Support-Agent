"""
pipeline.py
-----------
End-to-end pipeline that orchestrates all components.

Usage:
    from src.pipeline import SupportPipeline

    pipeline = SupportPipeline.load()          # load pre-built artefacts
    result   = pipeline.run("My order hasn't arrived yet")
    print(result)

The pipeline can also be built from scratch:
    pipeline = SupportPipeline.build(conversations_df)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    SELECTED_BRAND,
    EMBEDDING_CLASSIFIER_PATH,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    CONFIDENCE_ESCALATION_THRESHOLD,
    SIMILARITY_ESCALATION_THRESHOLD,
    RANDOM_SEED,
    TOP_K,
)
from src.preprocessing import clean_text, preprocess_dataframe
from src.intent_classifier import EmbeddingClassifier, TfidfLRClassifier, MajorityClassifier
from src.retriever import Retriever
from src.escalation import decide_escalation
from src.response_generator import generate_response

logger = logging.getLogger(__name__)


class SupportPipeline:
    """
    Full brand-aware customer support pipeline.

    Components:
        1. Preprocessing    — clean incoming text
        2. Intent classifier — EmbeddingClassifier (final system)
        3. Retriever        — FAISS semantic search
        4. Escalation engine — rule-based decision
        5. Response generator — LLM-grounded generation
    """

    def __init__(
        self,
        classifier: EmbeddingClassifier,
        retriever: Retriever,
        brand: str = SELECTED_BRAND,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.brand = brand

    # ------------------------------------------------------------------
    # Factory: build from conversations DataFrame
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        conversations_df: pd.DataFrame,
        brand: str = SELECTED_BRAND,
        label_col: str = "intent",
        train_ratio: float = 0.8,
        seed: int = RANDOM_SEED,
    ) -> "SupportPipeline":
        """
        Build and fit all pipeline components from a labelled conversations DataFrame.

        The DataFrame must contain:
            cleaned_text, brand_response, intent  (from golden eval or labelled set)

        The retriever is built ONLY on the training split to avoid leakage.
        """
        if label_col not in conversations_df.columns:
            raise ValueError(
                f"Column '{label_col}' not found. "
                "The pipeline.build() method requires a labelled DataFrame."
            )

        # Train/test split — stratified when possible, random otherwise
        from sklearn.model_selection import train_test_split
        n_classes = conversations_df[label_col].nunique()
        min_per_class = conversations_df[label_col].value_counts().min()
        test_size = 1.0 - train_ratio
        n_test = max(1, int(len(conversations_df) * test_size))
        can_stratify = (n_test >= n_classes) and (min_per_class >= 2)
        try:
            train_df, _ = train_test_split(
                conversations_df,
                train_size=train_ratio,
                stratify=conversations_df[label_col] if can_stratify else None,
                random_state=seed,
            )
        except ValueError:
            # Last resort: just shuffle without stratification
            train_df = conversations_df.sample(frac=train_ratio, random_state=seed)

        texts = train_df["cleaned_text"].tolist()
        labels = train_df[label_col].tolist()

        # Fit classifier
        logger.info("Fitting EmbeddingClassifier…")
        classifier = EmbeddingClassifier()
        classifier.fit(texts, labels)

        # Build retriever on training split only
        logger.info("Building retriever on training split (%d docs)…", len(train_df))
        retriever = Retriever(top_k=TOP_K)
        retriever.build(train_df)

        return cls(classifier=classifier, retriever=retriever, brand=brand)

    # ------------------------------------------------------------------
    # Factory: load from saved artefacts
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        classifier_path: Path = EMBEDDING_CLASSIFIER_PATH,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = FAISS_METADATA_PATH,
        brand: str = SELECTED_BRAND,
    ) -> "SupportPipeline":
        """Load a previously built pipeline from disk."""
        classifier = EmbeddingClassifier.load(classifier_path)
        retriever = Retriever.load(index_path, meta_path)
        return cls(classifier=classifier, retriever=retriever, brand=brand)

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------

    def save(
        self,
        classifier_path: Path = EMBEDDING_CLASSIFIER_PATH,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = FAISS_METADATA_PATH,
    ) -> None:
        self.classifier.save(classifier_path)
        self.retriever.save(index_path, meta_path)
        logger.info("Pipeline artefacts saved.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(
        self,
        message: str,
        top_k: int = TOP_K,
        exclude_ids: Optional[set] = None,
        generate: bool = True,
    ) -> dict:
        """
        Run the full pipeline on a single customer message.

        Parameters
        ----------
        message     : raw customer text (will be cleaned internally)
        top_k       : number of historical examples to retrieve
        exclude_ids : set of IDs to exclude from retrieval (anti-leakage)
        generate    : if False, skip LLM response generation

        Returns
        -------
        Structured result dict.
        """
        if not message or not message.strip():
            return {
                "message": message,
                "cleaned_message": "",
                "intent": "unknown",
                "confidence": 0.0,
                "decision": "ESCALATE",
                "decision_reason": "Empty or blank message received.",
                "decision_rule": "empty_input",
                "historical_examples": [],
                "response": None,
                "response_status": "skipped",
                "brand": self.brand,
            }

        # Step 1: Preprocess
        cleaned = clean_text(message)

        # Step 2: Classify intent
        prediction = self.classifier.predict_single(cleaned)
        intent = prediction["intent"]
        confidence = prediction["confidence"]
        logger.debug("Intent: %s (confidence=%.2f)", intent, confidence)

        # Step 3: Retrieve similar examples
        examples = self.retriever.retrieve(
            cleaned, top_k=top_k, exclude_ids=exclude_ids or set()
        )

        # Step 4: Escalation decision
        escalation = decide_escalation(
            text=cleaned,
            intent=intent,
            confidence=confidence,
            historical_examples=examples,
        )

        # Step 5: Generate response (only for AUTO-HANDLE)
        response_text = None
        response_status = "skipped"
        if generate and escalation["decision"] == "AUTO-HANDLE":
            gen_result = generate_response(
                customer_message=cleaned,
                intent=intent,
                historical_examples=examples,
                brand=self.brand,
            )
            response_text = gen_result["response"]
            response_status = gen_result["status"]
        elif escalation["decision"] == "ESCALATE":
            response_text = (
                "This message has been flagged for human review. "
                "A support agent will contact you shortly."
            )
            response_status = "escalated"

        return {
            "message": message,
            "cleaned_message": cleaned,
            "intent": intent,
            "confidence": round(confidence, 4),
            "decision": escalation["decision"],
            "decision_reason": escalation["reason"],
            "decision_rule": escalation["rule"],
            "historical_examples": examples,
            "response": response_text,
            "response_status": response_status,
            "brand": self.brand,
        }

    def run_batch(self, messages: list[str], **kwargs) -> list[dict]:
        """Run pipeline on a list of messages."""
        return [self.run(msg, **kwargs) for msg in messages]
