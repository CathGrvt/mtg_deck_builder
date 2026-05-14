from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


@dataclass
class DeckRecommendationRequest:
    session_id: str
    user_query: str
    format: str = "commander"
    colors: List[str] = field(default_factory=list)
    archetype_hint: Optional[str] = None
    must_include: List[str] = field(default_factory=list)
    must_exclude: List[str] = field(default_factory=list)
    mode: str = "deck_recommendation"

    def validate(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id is required.")
        if not self.user_query.strip():
            raise ValueError("user_query is required.")
        if not self.mode.strip():
            raise ValueError("mode is required.")
        if self.mode.strip().lower() not in {"deck_recommendation", "research_copilot"}:
            raise ValueError("mode must be either 'deck_recommendation' or 'research_copilot'.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_query": self.user_query,
            "format": self.format,
            "colors": list(self.colors),
            "archetype_hint": self.archetype_hint,
            "must_include": list(self.must_include),
            "must_exclude": list(self.must_exclude),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DeckRecommendationRequest":
        request = cls(
            session_id=str(payload.get("session_id", "")).strip(),
            user_query=str(payload.get("user_query", "")).strip(),
            format=str(payload.get("format", "commander")).strip() or "commander",
            colors=_string_list(payload.get("colors", [])),
            archetype_hint=str(payload.get("archetype_hint", "")).strip() or None,
            must_include=_string_list(payload.get("must_include", [])),
            must_exclude=_string_list(payload.get("must_exclude", [])),
            mode=str(payload.get("mode", "deck_recommendation")).strip() or "deck_recommendation",
        )
        request.validate()
        return request


@dataclass
class RetrievalPlan:
    rewritten_queries: List[str]
    corpus_targets: List[str] = field(default_factory=lambda: ["decklist", "card_db", "meta_json"])
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    top_k_per_query: int = 8
    max_chunks: int = 40

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rewritten_queries": list(self.rewritten_queries),
            "corpus_targets": list(self.corpus_targets),
            "metadata_filters": dict(self.metadata_filters),
            "top_k_per_query": int(self.top_k_per_query),
            "max_chunks": int(self.max_chunks),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RetrievalPlan":
        return cls(
            rewritten_queries=_string_list(payload.get("rewritten_queries", [])),
            corpus_targets=_string_list(payload.get("corpus_targets", [])) or ["decklist", "card_db", "meta_json"],
            metadata_filters=dict(payload.get("metadata_filters", {})),
            top_k_per_query=max(1, int(payload.get("top_k_per_query", 8))),
            max_chunks=max(1, int(payload.get("max_chunks", 40))),
        )


@dataclass
class RetrievedEvidence:
    doc_id: str
    chunk_id: str
    source: str
    title: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
            "text": self.text,
            "score": float(self.score),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RetrievedEvidence":
        return cls(
            doc_id=str(payload.get("doc_id", "")),
            chunk_id=str(payload.get("chunk_id", "")),
            source=str(payload.get("source", "unknown")),
            title=str(payload.get("title", "")),
            text=str(payload.get("text", "")),
            score=float(payload.get("score", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class RetrievalBundle:
    plan: RetrievalPlan
    chunks: List[RetrievedEvidence]
    rerank_scores: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "rerank_scores": {key: float(value) for key, value in self.rerank_scores.items()},
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RetrievalBundle":
        return cls(
            plan=RetrievalPlan.from_dict(dict(payload.get("plan", {}))),
            chunks=[
                RetrievedEvidence.from_dict(item)
                for item in payload.get("chunks", [])
                if isinstance(item, dict)
            ],
            rerank_scores={
                str(key): float(value)
                for key, value in dict(payload.get("rerank_scores", {})).items()
            },
            provenance=dict(payload.get("provenance", {})),
        )


@dataclass
class DeckCitation:
    doc_id: str
    chunk_id: str
    source: str
    title: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
        }

    @classmethod
    def from_evidence(cls, item: RetrievedEvidence) -> "DeckCitation":
        return cls(
            doc_id=item.doc_id,
            chunk_id=item.chunk_id,
            source=item.source,
            title=item.title,
        )


@dataclass
class SafetyVerdict:
    status: str
    reasons: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    blocked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "risk_score": float(self.risk_score),
            "blocked": bool(self.blocked),
        }


@dataclass
class DeckRecommendationResponse:
    summary: str
    recommended_decklist: List[str]
    key_claims: List[str]
    citations: List[DeckCitation]
    confidence: float
    safety_verdict: SafetyVerdict
    trace_id: str
    latency_ms: int
    model_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "recommended_decklist": list(self.recommended_decklist),
            "key_claims": list(self.key_claims),
            "citations": [item.to_dict() for item in self.citations],
            "confidence": float(self.confidence),
            "safety_verdict": self.safety_verdict.to_dict(),
            "trace_id": self.trace_id,
            "latency_ms": int(self.latency_ms),
            "model_used": self.model_used,
        }

    @classmethod
    def blocked(
        cls,
        reason: str,
        trace_id: str,
        latency_ms: int,
    ) -> "DeckRecommendationResponse":
        return cls(
            summary="Request blocked by safety guardrails.",
            recommended_decklist=[],
            key_claims=[],
            citations=[],
            confidence=0.0,
            safety_verdict=SafetyVerdict(
                status="blocked",
                reasons=[reason],
                risk_score=1.0,
                blocked=True,
            ),
            trace_id=trace_id,
            latency_ms=latency_ms,
            model_used="",
        )
