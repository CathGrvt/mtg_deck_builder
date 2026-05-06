from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_id: str
    source: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DocumentChunk":
        return cls(
            doc_id=str(payload.get("doc_id", "")),
            chunk_id=str(payload.get("chunk_id", "")),
            source=str(payload.get("source", "unknown")),
            title=str(payload.get("title", "")),
            text=str(payload.get("text", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class RetrievedChunk:
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
    def from_chunk(cls, chunk: DocumentChunk, score: float) -> "RetrievedChunk":
        return cls(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            title=chunk.title,
            text=chunk.text,
            score=float(score),
            metadata=dict(chunk.metadata),
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RetrievedChunk":
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
class Citation:
    doc_id: str
    chunk_id: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Citation":
        return cls(
            doc_id=str(payload.get("doc_id", "")),
            chunk_id=str(payload.get("chunk_id", "")),
        )


@dataclass
class Claim:
    claim: str
    citations: List[Citation]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "citations": [item.to_dict() for item in self.citations],
            "confidence": float(self.confidence),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Claim":
        raw_citations = payload.get("citations", [])
        citations = [
            Citation.from_dict(item)
            for item in raw_citations
            if isinstance(item, dict)
        ]
        return cls(
            claim=str(payload.get("claim", "")),
            citations=citations,
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass
class StructuredReport:
    topic: str
    summary: str
    claims: List[Claim]
    open_questions: List[str]
    validation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "claims": [claim.to_dict() for claim in self.claims],
            "open_questions": list(self.open_questions),
            "validation": dict(self.validation),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StructuredReport":
        raw_claims = payload.get("claims", [])
        claims = [Claim.from_dict(item) for item in raw_claims if isinstance(item, dict)]
        return cls(
            topic=str(payload.get("topic", "")),
            summary=str(payload.get("summary", "")),
            claims=claims,
            open_questions=[str(item) for item in payload.get("open_questions", [])],
            validation=dict(payload.get("validation", {})),
        )
