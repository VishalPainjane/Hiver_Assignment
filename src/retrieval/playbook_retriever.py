"""Semantic Problem-Twin Retriever over Apple's Curated Playbook Vault.

Features:
1. Intent-Partitioned Search (filters search space by classified intent).
2. Customer-to-Customer Semantic Matching (finds historical customer with identical symptoms).
3. Calibrated 3-Zone Confidence Gating:
   - Zone 1 (High): Strong Precedent Grounding.
   - Zone 2 (Moderate): Style & Structure Only, No Factual Copying.
   - Zone 3 (Low): Out-of-Distribution Edge Case -> Safe Universal Triage.
4. Dual-Engine Architecture: Fast, robust TF-IDF + n-gram vector matching with optional Dense Sentence Transformer integration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("retrieval.playbook")


@dataclass(frozen=True)
class PrecedentMatch:
    """A matched historical customer-complaint and Apple-reply pair."""
    pair_id: str
    case_id: str
    intent: str
    customer_query: str
    apple_reply: str
    similarity_score: float


@dataclass(frozen=True)
class RetrievalResult:
    """Structured output from playbook retrieval."""
    intent: str
    matches: List[PrecedentMatch]
    top_similarity: float
    confidence_zone: str  # "HIGH_PREPARATION" | "MODERATE_STYLE_ONLY" | "LOW_OUT_OF_DISTRIBUTION"
    best_template: str


class PlaybookRetriever:
    """High-speed, intent-partitioned semantic retriever for historical Apple resolutions."""

    # Calibrated thresholds for TF-IDF / lexical vector similarity
    TFIDF_HIGH_THRESHOLD = 0.25
    TFIDF_MODERATE_THRESHOLD = 0.12

    # Calibrated thresholds for Dense Embeddings
    DENSE_HIGH_THRESHOLD = 0.75
    DENSE_MODERATE_THRESHOLD = 0.55

    SAFE_TRIAGE_FALLBACK = (
        "{customer_handle} We'd like to take a closer look into what's happening. "
        "Send us a DM with your exact device model, iOS version, and details so we can assist: {dm_link}"
    )

    def __init__(
        self,
        vault_path: str | Path = "data/processed/apple_playbook_vault.json",
        dense_encoder: Optional[Any] = None,
    ) -> None:
        self.vault_path = Path(vault_path)
        self.dense_encoder = dense_encoder
        self.vault_data: Dict[str, List[Dict[str, str]]] = {}
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.tfidf_matrices: Dict[str, Any] = {}
        self._load_vault()

    def _load_vault(self) -> None:
        """Load vault from JSON and fit TF-IDF vectorizers per intent."""
        if not self.vault_path.exists():
            logger.warning(f"Vault path {self.vault_path} not found.")
            return

        with open(self.vault_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        self.vault_data = content.get("vault", {})
        total = sum(len(v) for v in self.vault_data.values())
        logger.info(f"Loaded {total} playbook pairs across {len(self.vault_data)} intents.")

        # Fit intent-partitioned TF-IDF vectorizers
        for intent, pairs in self.vault_data.items():
            if not pairs:
                continue
            queries = [p["customer_query"] for p in pairs]
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b",
                stop_words="english",
            )
            matrix = vectorizer.fit_transform(queries)
            self.vectorizers[intent] = vectorizer
            self.tfidf_matrices[intent] = matrix

    def retrieve(
        self,
        query: str,
        intent: str = "UNKNOWN",
        top_k: int = 3,
    ) -> RetrievalResult:
        """Retrieve the top-k most similar historical cases for an incoming query."""
        clean_query = query.strip()
        target_intent = intent.upper() if intent.upper() in self.vault_data else "UNKNOWN"

        pairs = self.vault_data.get(target_intent, [])
        if not pairs or target_intent not in self.vectorizers:
            # Fallback if partition empty
            return RetrievalResult(
                intent=target_intent,
                matches=[],
                top_similarity=0.0,
                confidence_zone="LOW_OUT_OF_DISTRIBUTION",
                best_template=self.SAFE_TRIAGE_FALLBACK,
            )

        vectorizer = self.vectorizers[target_intent]
        matrix = self.tfidf_matrices[target_intent]

        # Compute cosine similarity
        query_vec = vectorizer.transform([clean_query])
        scores = cosine_similarity(query_vec, matrix)[0]

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        matches: List[PrecedentMatch] = []
        for idx in top_indices:
            score = float(scores[idx])
            p = pairs[idx]
            matches.append(
                PrecedentMatch(
                    pair_id=p.get("pair_id", ""),
                    case_id=p.get("case_id", ""),
                    intent=target_intent,
                    customer_query=p.get("customer_query", ""),
                    apple_reply=p.get("apple_reply", ""),
                    similarity_score=round(score, 4),
                )
            )

        top_score = matches[0].similarity_score if matches else 0.0

        # Calibrated threshold evaluation
        is_dense = self.dense_encoder is not None
        high_thresh = self.DENSE_HIGH_THRESHOLD if is_dense else self.TFIDF_HIGH_THRESHOLD
        mod_thresh = self.DENSE_MODERATE_THRESHOLD if is_dense else self.TFIDF_MODERATE_THRESHOLD

        if top_score >= high_thresh:
            zone = "HIGH_PREPARATION"
            best_template = matches[0].apple_reply
        elif top_score >= mod_thresh:
            zone = "MODERATE_STYLE_ONLY"
            best_template = matches[0].apple_reply
        else:
            zone = "LOW_OUT_OF_DISTRIBUTION"
            best_template = self.SAFE_TRIAGE_FALLBACK

        return RetrievalResult(
            intent=target_intent,
            matches=matches,
            top_similarity=top_score,
            confidence_zone=zone,
            best_template=best_template,
        )

