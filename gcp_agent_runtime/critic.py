from __future__ import annotations

from dataclasses import dataclass
from typing import List

from gcp_agent_runtime.contracts import DeckRecommendationRequest, RetrievalBundle


@dataclass
class CriticOutcome:
    needs_second_pass: bool
    gaps: List[str]
    reason: str
    predicted_confidence: float


@dataclass
class CriticSettings:
    min_unique_docs: int = 4
    min_chunks: int = 8
    confidence_floor: float = 0.45


class CriticAgent:
    def __init__(self, settings: CriticSettings | None = None):
        self.settings = settings or CriticSettings()

    def evaluate(
        self,
        request: DeckRecommendationRequest,
        bundle: RetrievalBundle,
    ) -> CriticOutcome:
        unique_docs = len({item.doc_id for item in bundle.chunks})
        chunk_count = len(bundle.chunks)
        gaps: List[str] = []

        corpus_text = " ".join(item.text.lower() for item in bundle.chunks)
        for name in request.must_include[:8]:
            lowered = name.lower().strip()
            if lowered and lowered not in corpus_text:
                gaps.append(f"Need supporting evidence for include card: {name}")

        doc_ratio = min(1.0, unique_docs / float(max(1, self.settings.min_unique_docs)))
        chunk_ratio = min(1.0, chunk_count / float(max(1, self.settings.min_chunks)))
        predicted_confidence = round(max(0.0, min(1.0, 0.5 * doc_ratio + 0.5 * chunk_ratio)), 4)

        needs_second_pass = (
            unique_docs < self.settings.min_unique_docs
            or chunk_count < self.settings.min_chunks
            or bool(gaps)
        )
        reason = (
            "coverage_sufficient"
            if not needs_second_pass
            else f"coverage_thin docs={unique_docs}, chunks={chunk_count}, gaps={len(gaps)}"
        )

        return CriticOutcome(
            needs_second_pass=needs_second_pass,
            gaps=gaps[:6],
            reason=reason,
            predicted_confidence=max(self.settings.confidence_floor, predicted_confidence)
            if not needs_second_pass
            else predicted_confidence,
        )
