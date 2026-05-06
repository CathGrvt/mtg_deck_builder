from __future__ import annotations

from typing import Any, Dict

from research_pipeline.llm import AgentLLM
from research_pipeline.trace import TraceLogger


def run_planner_node(
    state: Dict[str, Any],
    llm: AgentLLM,
    max_questions: int,
    trace: TraceLogger | None = None,
) -> Dict[str, Any]:
    topic = str(state.get("topic", "")).strip()
    gaps = [str(item) for item in state.get("gaps", [])]

    started = trace.log_node_start("planner", {"topic": topic, "gaps": gaps}) if trace else None

    subquestions = llm.plan_subquestions(
        topic=topic,
        previous_gaps=gaps,
        max_questions=max_questions,
    )

    payload = {
        "subquestions": subquestions,
        "active_queries": subquestions,
    }

    if trace and started is not None:
        trace.log_node_end("planner", started, {"subquestion_count": len(subquestions)})

    return payload
