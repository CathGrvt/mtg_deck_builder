from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

from research_pipeline.grounding import max_overlap_against_claim
from research_pipeline.models import RetrievedChunk, StructuredReport


def evaluate_report(
    report_payload: Dict[str, Any],
    retrieved_chunks_payload: Iterable[Dict[str, Any]],
    support_threshold: float = 0.08,
) -> Dict[str, Any]:
    report = StructuredReport.from_dict(report_payload)
    retrieved_chunks = [
        RetrievedChunk.from_dict(item)
        for item in retrieved_chunks_payload
        if isinstance(item, dict)
    ]

    chunk_lookup = {
        (chunk.doc_id, chunk.chunk_id): chunk
        for chunk in retrieved_chunks
    }

    claim_support_scores: List[float] = []
    claim_is_grounded: List[bool] = []
    total_citations = 0
    valid_citations = 0

    for claim in report.claims:
        cited_texts: List[str] = []
        for citation in claim.citations:
            total_citations += 1
            chunk = chunk_lookup.get((citation.doc_id, citation.chunk_id))
            if chunk is None:
                continue
            valid_citations += 1
            cited_texts.append(chunk.text)

        support_score = max_overlap_against_claim(claim.claim, cited_texts)
        claim_support_scores.append(support_score)
        claim_is_grounded.append(bool(cited_texts) and support_score >= support_threshold)

    claim_count = len(report.claims)
    supported_claims = sum(1 for item in claim_is_grounded if item)

    groundedness = float(supported_claims / claim_count) if claim_count else 0.0
    citation_precision = float(valid_citations / total_citations) if total_citations else 0.0
    citation_recall = float(supported_claims / claim_count) if claim_count else 0.0
    faithfulness = float(mean(claim_support_scores)) if claim_support_scores else 0.0

    return {
        "claim_count": claim_count,
        "supported_claims": supported_claims,
        "groundedness": round(groundedness, 4),
        "faithfulness": round(faithfulness, 4),
        "citation_precision": round(citation_precision, 4),
        "citation_recall": round(citation_recall, 4),
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "support_threshold": support_threshold,
        "claim_support_scores": [round(score, 4) for score in claim_support_scores],
    }


def classify_failure(metrics: Dict[str, Any]) -> Tuple[str, str]:
    claim_count = int(metrics.get("claim_count", 0))
    groundedness = float(metrics.get("groundedness", 0.0))
    faithfulness = float(metrics.get("faithfulness", 0.0))
    citation_precision = float(metrics.get("citation_precision", 0.0))
    total_citations = int(metrics.get("total_citations", 0))

    if claim_count == 0:
        return "retrieval_miss", "No claims produced; retrieval or synthesis likely failed."
    if total_citations == 0:
        return "bad_citation", "Claims are uncited."
    if citation_precision < 0.6:
        return "bad_citation", "Many citations do not match retrieved context."
    if groundedness < 0.5:
        return "retrieval_miss", "Claims are weakly supported by retrieved chunks."
    if faithfulness < 0.1:
        return "hallucinated_claim", "Claim wording has low overlap with cited evidence."
    return "ok", "No major failure category triggered."


def maybe_compute_ragas(
    question: str,
    answer: str,
    contexts: List[str],
) -> Dict[str, Any]:
    """
    Optional ragas integration. Returns empty dict when ragas is unavailable.
    """
    del question, answer, contexts
    try:
        import ragas  # noqa: F401
    except Exception:
        return {}

    # Kept intentionally minimal to avoid hard coupling to a moving API surface.
    # Use this hook if ragas is installed and a project-specific config is provided.
    return {"ragas": "installed_but_not_configured"}
