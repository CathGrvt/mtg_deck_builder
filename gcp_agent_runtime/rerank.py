from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from gcp_agent_runtime.contracts import RetrievedEvidence, RetrievalBundle


def _tokens(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9']+", str(text).lower())
        if len(token) >= 3
    }


@dataclass
class RerankSettings:
    top_n: int = 12
    source_prior: Dict[str, float] | None = None
    overlap_weight: float = 0.35
    base_score_weight: float = 0.65

    def resolved_source_prior(self) -> Dict[str, float]:
        if self.source_prior is not None:
            return self.source_prior
        return {
            "decklist": 1.0,
            "card_db": 0.95,
            "meta_json": 0.9,
            "vertex_rag": 1.0,
        }


class RerankAgent:
    def __init__(self, settings: RerankSettings | None = None):
        self.settings = settings or RerankSettings()

    def rerank(self, bundle: RetrievalBundle, query_text: str) -> RetrievalBundle:
        query_tokens = _tokens(query_text)
        source_prior = self.settings.resolved_source_prior()

        rescored: List[RetrievedEvidence] = []
        rerank_scores: Dict[str, float] = {}

        for item in bundle.chunks:
            text_tokens = _tokens(f"{item.title} {item.text}")
            overlap = 0.0
            if query_tokens and text_tokens:
                overlap = float(len(query_tokens & text_tokens)) / float(len(query_tokens | text_tokens))
            prior = source_prior.get(item.source, 0.85)
            combined = (
                self.settings.base_score_weight * float(item.score)
                + self.settings.overlap_weight * overlap
            ) * prior

            rescored.append(
                RetrievedEvidence(
                    doc_id=item.doc_id,
                    chunk_id=item.chunk_id,
                    source=item.source,
                    title=item.title,
                    text=item.text,
                    score=float(combined),
                    metadata=dict(item.metadata),
                )
            )
            rerank_scores[item.chunk_id] = float(combined)

        rescored.sort(key=lambda entry: entry.score, reverse=True)
        top_n = max(1, int(self.settings.top_n))
        bundle.chunks = rescored[:top_n]
        bundle.rerank_scores = rerank_scores
        bundle.provenance = dict(bundle.provenance)
        bundle.provenance["reranked_top_n"] = top_n
        return bundle

    def merge_and_rerank(
        self,
        bundles: Sequence[RetrievalBundle],
        query_text: str,
    ) -> RetrievalBundle:
        if not bundles:
            raise ValueError("merge_and_rerank requires at least one retrieval bundle.")

        plan = bundles[0].plan
        merged: Dict[str, RetrievedEvidence] = {}
        provenance = {"merged_bundle_count": len(bundles)}

        for bundle in bundles:
            for item in bundle.chunks:
                current = merged.get(item.chunk_id)
                if current is None or item.score > current.score:
                    merged[item.chunk_id] = item

        combined = RetrievalBundle(
            plan=plan,
            chunks=sorted(merged.values(), key=lambda value: value.score, reverse=True),
            rerank_scores={},
            provenance=provenance,
        )
        return self.rerank(combined, query_text=query_text)
