from __future__ import annotations

from typing import Any, Dict, List

from research_pipeline.llm import AgentLLM
from research_pipeline.models import RetrievedChunk
from research_pipeline.trace import TraceLogger


def run_critic_node(
    state: Dict[str, Any],
    llm: AgentLLM,
    max_iterations: int,
    trace: TraceLogger | None = None,
) -> Dict[str, Any]:
    topic = str(state.get("topic", "")).strip()
    subquestions = [str(item) for item in state.get("subquestions", [])]
    retrieved_chunks: List[RetrievedChunk] = [
        RetrievedChunk.from_dict(item)
        for item in state.get("retrieved_chunks", [])
        if isinstance(item, dict)
    ]

    next_iteration = int(state.get("iteration", 0)) + 1

    started = trace.log_node_start(
        "critic",
        {
            "iteration": next_iteration,
            "retrieved_chunks": len(retrieved_chunks),
        },
    ) if trace else None

    critique = llm.critique(
        topic=topic,
        subquestions=subquestions,
        retrieved_chunks=retrieved_chunks,
        iteration=next_iteration,
        max_iterations=max_iterations,
    )

    needs_more = bool(critique.get("needs_more_research", False))
    gaps = [str(item).strip() for item in critique.get("gaps", []) if str(item).strip()]
    if next_iteration >= max_iterations:
        needs_more = False

    if trace and started is not None:
        trace.log_node_end(
            "critic",
            started,
            {
                "needs_more_research": needs_more,
                "gaps": gaps,
                "reason": str(critique.get("reason", "")),
            },
        )

    return {
        "iteration": next_iteration,
        "critic_needs_more_research": needs_more,
        "gaps": gaps,
        "critic_reason": str(critique.get("reason", "")),
        "active_queries": gaps if needs_more and gaps else subquestions,
    }
