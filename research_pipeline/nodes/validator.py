from __future__ import annotations

from typing import Any, Dict, List, Tuple

from research_pipeline.grounding import max_overlap_against_claim
from research_pipeline.models import RetrievedChunk, StructuredReport
from research_pipeline.trace import TraceLogger


SUPPORT_THRESHOLD = 0.08


def run_validator_node(
    state: Dict[str, Any],
    trace: TraceLogger | None = None,
) -> Dict[str, Any]:
    report = StructuredReport.from_dict(dict(state.get("report", {})))

    retrieved_chunks: List[RetrievedChunk] = [
        RetrievedChunk.from_dict(item)
        for item in state.get("retrieved_chunks", [])
        if isinstance(item, dict)
    ]

    chunk_lookup: Dict[Tuple[str, str], RetrievedChunk] = {
        (chunk.doc_id, chunk.chunk_id): chunk for chunk in retrieved_chunks
    }

    started = trace.log_node_start(
        "validator",
        {
            "claim_count": len(report.claims),
            "retrieved_chunks": len(retrieved_chunks),
        },
    ) if trace else None

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

        support = max_overlap_against_claim(claim.claim, cited_texts)
        is_supported = citation_hits > 0 and support >= SUPPORT_THRESHOLD
        if is_supported:
            supported_claims += 1

        claim_diagnostics.append(
            {
                "claim": claim.claim,
                "citation_hits": citation_hits,
                "support_score": round(float(support), 4),
                "supported": is_supported,
            }
        )

    citation_precision = float(valid_citations / total_citations) if total_citations else 0.0
    groundedness = float(supported_claims / len(report.claims)) if report.claims else 0.0

    report.validation = {
        "support_threshold": SUPPORT_THRESHOLD,
        "total_claims": len(report.claims),
        "supported_claims": supported_claims,
        "groundedness": round(groundedness, 4),
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "citation_precision": round(citation_precision, 4),
        "claim_diagnostics": claim_diagnostics,
    }

    if trace and started is not None:
        trace.log_node_end(
            "validator",
            started,
            {
                "groundedness": report.validation["groundedness"],
                "citation_precision": report.validation["citation_precision"],
            },
        )

    return {
        "report": report.to_dict(),
    }
