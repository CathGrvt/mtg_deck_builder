from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from research_pipeline.grounding import best_overlap_against_claim, build_token_idf, topic_alignment
from research_pipeline.models import RetrievedChunk, StructuredReport


DEFAULT_SUPPORT_THRESHOLD = 0.08
DEFAULT_TOPIC_RELEVANCE_THRESHOLD = 0.05


def _build_chunk_lookup(retrieved_chunks: Sequence[RetrievedChunk]) -> Dict[Tuple[str, str], RetrievedChunk]:
    return {
        (chunk.doc_id, chunk.chunk_id): chunk
        for chunk in retrieved_chunks
    }


def compute_report_validation(
    report: StructuredReport,
    retrieved_chunks: Sequence[RetrievedChunk],
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    topic_relevance_threshold: float = DEFAULT_TOPIC_RELEVANCE_THRESHOLD,
) -> Dict[str, Any]:
    chunk_lookup = _build_chunk_lookup(retrieved_chunks)
    topic_idf = build_token_idf(
        [report.topic] + [chunk.text for chunk in retrieved_chunks],
        min_token_length=4,
    )

    claim_support_scores: List[float] = []
    claim_faithfulness_scores: List[float] = []
    claim_topic_relevance_scores: List[float] = []
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

        support_score, support_span = best_overlap_against_claim(claim.claim, cited_texts)
        topic_score, topic_overlap_terms = topic_alignment(
            claim.claim,
            report.topic,
            token_idf=topic_idf,
        )
        faithfulness_score = float(support_score * topic_score)

        claim_support_scores.append(support_score)
        claim_topic_relevance_scores.append(topic_score)
        claim_faithfulness_scores.append(faithfulness_score)

        is_supported = (
            citation_hits > 0
            and support_score >= support_threshold
            and topic_score >= topic_relevance_threshold
        )
        if is_supported:
            supported_claims += 1

        claim_diagnostics.append(
            {
                "claim": claim.claim,
                "citation_hits": citation_hits,
                "support_score": round(float(support_score), 4),
                "topic_relevance_score": round(float(topic_score), 4),
                "faithfulness_score": round(float(faithfulness_score), 4),
                "supported": bool(is_supported),
                "support_span": support_span,
                "topic_overlap_terms": topic_overlap_terms,
            }
        )

    claim_count = len(report.claims)

    groundedness = float(supported_claims / claim_count) if claim_count else 0.0
    citation_precision = float(valid_citations / total_citations) if total_citations else 0.0
    citation_recall = float(supported_claims / claim_count) if claim_count else 0.0
    faithfulness = float(mean(claim_faithfulness_scores)) if claim_faithfulness_scores else 0.0
    topic_relevance = float(mean(claim_topic_relevance_scores)) if claim_topic_relevance_scores else 0.0

    return {
        "support_threshold": float(support_threshold),
        "topic_relevance_threshold": float(topic_relevance_threshold),
        "total_claims": claim_count,
        "claim_count": claim_count,
        "supported_claims": supported_claims,
        "groundedness": round(groundedness, 4),
        "faithfulness": round(faithfulness, 4),
        "topic_relevance": round(topic_relevance, 4),
        "citation_precision": round(citation_precision, 4),
        "citation_recall": round(citation_recall, 4),
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "claim_support_scores": [round(score, 4) for score in claim_support_scores],
        "claim_topic_relevance_scores": [round(score, 4) for score in claim_topic_relevance_scores],
        "claim_faithfulness_scores": [round(score, 4) for score in claim_faithfulness_scores],
        "claim_diagnostics": claim_diagnostics,
    }


def coerce_retrieved_chunks(retrieved_chunks_payload: Iterable[Dict[str, Any]]) -> List[RetrievedChunk]:
    return [
        RetrievedChunk.from_dict(item)
        for item in retrieved_chunks_payload
        if isinstance(item, dict)
    ]
