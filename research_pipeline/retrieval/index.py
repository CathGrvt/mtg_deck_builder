from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from research_pipeline.models import DocumentChunk, RetrievedChunk


class HybridRetrievalIndex:
    """
    Hybrid lexical + semantic retrieval over DocumentChunk objects.
    Semantic retrieval uses sentence-transformers when available.
    """

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        lexical_weight: float = 0.6,
        semantic_weight: float = 0.4,
        enable_semantic: bool = True,
        semantic_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.chunks: List[DocumentChunk] = list(chunks)
        self.lexical_weight = max(0.0, float(lexical_weight))
        self.semantic_weight = max(0.0, float(semantic_weight))

        if not self.chunks:
            raise ValueError("HybridRetrievalIndex requires at least one chunk.")

        texts = [chunk.text for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        self.lexical_matrix = self.vectorizer.fit_transform(texts)

        self.semantic_model = None
        self.semantic_matrix: Optional[np.ndarray] = None
        if enable_semantic:
            try:
                from sentence_transformers import SentenceTransformer

                self.semantic_model = SentenceTransformer(semantic_model_name)
                self.semantic_matrix = self.semantic_model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception:
                self.semantic_model = None
                self.semantic_matrix = None

        self.chunk_lookup: Dict[str, DocumentChunk] = {
            chunk.chunk_id: chunk for chunk in self.chunks
        }

    def search(self, query: str, top_k: int = 8) -> List[RetrievedChunk]:
        if not query.strip():
            return []

        top_k = max(1, int(top_k))

        lexical_query = self.vectorizer.transform([query])
        lexical_scores = cosine_similarity(lexical_query, self.lexical_matrix).flatten()

        semantic_scores = None
        if self.semantic_model is not None and self.semantic_matrix is not None:
            try:
                query_emb = self.semantic_model.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                semantic_scores = (query_emb @ self.semantic_matrix.T).flatten()
            except Exception:
                semantic_scores = None

        combined_scores = self._combine_scores(lexical_scores, semantic_scores)

        if combined_scores.size == 0:
            return []

        candidate_idx = np.argsort(combined_scores)[::-1][:top_k]
        results: List[RetrievedChunk] = []
        for idx in candidate_idx:
            chunk = self.chunks[int(idx)]
            score = float(combined_scores[int(idx)])
            results.append(RetrievedChunk.from_chunk(chunk, score=score))
        return results

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self.chunk_lookup.get(chunk_id)

    def _combine_scores(
        self,
        lexical_scores: np.ndarray,
        semantic_scores: Optional[np.ndarray],
    ) -> np.ndarray:
        lex = self._normalize(lexical_scores)
        if semantic_scores is None:
            return lex

        sem = self._normalize(semantic_scores)
        total_weight = self.lexical_weight + self.semantic_weight
        if total_weight <= 0.0:
            return 0.5 * lex + 0.5 * sem
        return (self.lexical_weight * lex + self.semantic_weight * sem) / total_weight

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_value = float(scores.min())
        max_value = float(scores.max())
        if max_value - min_value < 1e-12:
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - min_value) / (max_value - min_value)
