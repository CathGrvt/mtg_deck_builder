from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from research_pipeline.grounding import max_overlap_against_claim
from research_pipeline.models import RetrievedChunk, StructuredReport


DEFAULT_SUPPORT_THRESHOLD = 0.08


def _build_chunk_lookup(retrieved_chunks: Sequence[RetrievedChunk]) -> Dict[Tuple[str, str], RetrievedChunk]:
    return {
        (chunk.doc_id, chunk.chunk_id): chunk
        for chunk in retrieved_chunks
    }


def compute_report_validation(
    report: StructuredReport,
    retrieved_chunks: Sequence[RetrievedChunk],
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> Dict[str, Any]:
    chunk_lookup = _build_chunk_lookup(retrieved_chunks)

    claim_support_scores: List[float] = []
    total_citations = 0
    valid_citations = 0
    supported_claims = 0
    claim_diagnostics: List[Dict[str, Any]] = []

    for claim in report.claims:
        cited_texts: List[str] = []
        citation_hits = 0

        for citation in claim.citations:
            total_citations += 1
            chunk = chunk_lookup.get((citation.doc_id, citation.chunk_id))
            if chunk is None:
                continue
            valid_citations += 1
            citation_hits += 1
            cited_texts.append(chunk.text)

        support_score = max_overlap_against_claim(claim.claim, cited_texts)
        claim_support_scores.append(support_score)
        is_supported = citation_hits > 0 and support_score >= support_threshold
        if is_supported:
            supported_claims += 1

        claim_diagnostics.append(
            {
                "claim": claim.claim,
                "citation_hits": citation_hits,
                "support_score": round(float(support_score), 4),
                "supported": bool(is_supported),
            }
        )

    claim_count = len(report.claims)

    groundedness = float(supported_claims / claim_count) if claim_count else 0.0
    citation_precision = float(valid_citations / total_citations) if total_citations else 0.0
    citation_recall = float(supported_claims / claim_count) if claim_count else 0.0
    faithfulness = float(mean(claim_support_scores)) if claim_support_scores else 0.0

    return {
        "support_threshold": float(support_threshold),
        "total_claims": claim_count,
        "claim_count": claim_count,
        "supported_claims": supported_claims,
        "groundedness": round(groundedness, 4),
        "faithfulness": round(faithfulness, 4),
        "citation_precision": round(citation_precision, 4),
        "citation_recall": round(citation_recall, 4),
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "claim_support_scores": [round(score, 4) for score in claim_support_scores],
        "claim_diagnostics": claim_diagnostics,
    }


def coerce_retrieved_chunks(retrieved_chunks_payload: Iterable[Dict[str, Any]]) -> List[RetrievedChunk]:
    return [
        RetrievedChunk.from_dict(item)
        for item in retrieved_chunks_payload
        if isinstance(item, dict)
    ]
