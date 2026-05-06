from __future__ import annotations

from typing import Any, Dict, List

from research_pipeline.models import RetrievedChunk, StructuredReport
from research_pipeline.trace import TraceLogger
from research_pipeline.validation import DEFAULT_SUPPORT_THRESHOLD, compute_report_validation

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

    started = trace.log_node_start(
        "validator",
        {
            "claim_count": len(report.claims),
            "retrieved_chunks": len(retrieved_chunks),
        },
    ) if trace else None

    report.validation = compute_report_validation(
        report=report,
        retrieved_chunks=retrieved_chunks,
        support_threshold=DEFAULT_SUPPORT_THRESHOLD,
    )

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
