from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from research_pipeline.models import StructuredReport
from research_pipeline.validation import (
    DEFAULT_SUPPORT_THRESHOLD,
    coerce_retrieved_chunks,
    compute_report_validation,
)


def evaluate_report(
    report_payload: Dict[str, Any],
    retrieved_chunks_payload: Iterable[Dict[str, Any]],
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> Dict[str, Any]:
    report = StructuredReport.from_dict(report_payload)
    retrieved_chunks = coerce_retrieved_chunks(retrieved_chunks_payload)
    return compute_report_validation(
        report=report,
        retrieved_chunks=retrieved_chunks,
        support_threshold=support_threshold,
    )


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
