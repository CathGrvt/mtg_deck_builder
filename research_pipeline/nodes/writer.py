from __future__ import annotations

from typing import Any, Dict, List

from research_pipeline.llm import AgentLLM
from research_pipeline.models import RetrievedChunk
from research_pipeline.trace import TraceLogger


def run_writer_node(
    state: Dict[str, Any],
    llm: AgentLLM,
    trace: TraceLogger | None = None,
) -> Dict[str, Any]:
    topic = str(state.get("topic", "")).strip()
    gaps = [str(item) for item in state.get("gaps", [])]
    retrieved_chunks: List[RetrievedChunk] = [
        RetrievedChunk.from_dict(item)
        for item in state.get("retrieved_chunks", [])
        if isinstance(item, dict)
    ]

    started = trace.log_node_start(
        "writer",
        {
            "topic": topic,
            "retrieved_chunks": len(retrieved_chunks),
        },
    ) if trace else None

    report = llm.write_report(
        topic=topic,
        retrieved_chunks=retrieved_chunks,
        gaps=gaps,
    )

    if trace and started is not None:
        trace.log_node_end(
            "writer",
            started,
            {
                "claim_count": len(report.claims),
                "open_questions": len(report.open_questions),
            },
        )

    return {
        "report": report.to_dict(),
    }
