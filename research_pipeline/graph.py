from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from research_pipeline.llm import AgentLLM, build_default_llm
from research_pipeline.models import StructuredReport
from research_pipeline.nodes.critic import run_critic_node
from research_pipeline.nodes.planner import run_planner_node
from research_pipeline.nodes.retriever import run_retriever_node
from research_pipeline.nodes.validator import run_validator_node
from research_pipeline.nodes.writer import run_writer_node
from research_pipeline.retrieval.corpus import build_domain_corpus
from research_pipeline.retrieval.index import HybridRetrievalIndex
from research_pipeline.trace import TraceLogger


class PipelineState(TypedDict, total=False):
    topic: str
    iteration: int
    max_iterations: int
    subquestions: List[str]
    active_queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    query_hits: Dict[str, List[str]]
    gaps: List[str]
    critic_reason: str
    critic_needs_more_research: bool
    report: Dict[str, Any]


class ResearchPipeline:
    def __init__(
        self,
        index: HybridRetrievalIndex,
        llm: Optional[AgentLLM] = None,
        max_iterations: int = 3,
        max_questions: int = 5,
        top_k_per_query: int = 5,
        use_langgraph: bool = True,
        trace_path: Optional[str] = None,
    ):
        self.index = index
        self.llm = llm or build_default_llm()
        self.max_iterations = max(1, int(max_iterations))
        self.max_questions = max(1, int(max_questions))
        self.top_k_per_query = max(1, int(top_k_per_query))
        self.trace = TraceLogger(trace_path) if trace_path else None

        self._compiled_graph = None
        if use_langgraph:
            self._compiled_graph = self._build_langgraph_graph()

    def run(self, topic: str) -> Dict[str, Any]:
        initial_state: PipelineState = {
            "topic": topic,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "gaps": [],
            "subquestions": [],
            "active_queries": [],
            "retrieved_chunks": [],
            "query_hits": {},
            "critic_needs_more_research": False,
        }

        if self.trace:
            self.trace.log(
                "pipeline_start",
                "pipeline",
                {
                    "topic": topic,
                    "max_iterations": self.max_iterations,
                    "top_k_per_query": self.top_k_per_query,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        if self._compiled_graph is not None:
            final_state = self._compiled_graph.invoke(initial_state)
        else:
            final_state = self._run_manual(initial_state)

        report = StructuredReport.from_dict(dict(final_state.get("report", {})))

        output = {
            "topic": topic,
            "state": final_state,
            "report": report.to_dict(),
        }

        if self.trace:
            self.trace.log(
                "pipeline_end",
                "pipeline",
                {
                    "topic": topic,
                    "iterations": final_state.get("iteration", 0),
                    "claim_count": len(report.claims),
                    "groundedness": report.validation.get("groundedness"),
                },
            )

        return output

    def _run_manual(self, state: PipelineState) -> PipelineState:
        state = dict(state)
        state.update(
            run_planner_node(
                state=state,
                llm=self.llm,
                max_questions=self.max_questions,
                trace=self.trace,
            )
        )
        state.update(
            run_retriever_node(
                state=state,
                index=self.index,
                top_k_per_query=self.top_k_per_query,
                trace=self.trace,
            )
        )

        while True:
            state.update(
                run_critic_node(
                    state=state,
                    llm=self.llm,
                    max_iterations=self.max_iterations,
                    trace=self.trace,
                )
            )
            if not state.get("critic_needs_more_research", False):
                break
            state.update(
                run_retriever_node(
                    state=state,
                    index=self.index,
                    top_k_per_query=self.top_k_per_query,
                    trace=self.trace,
                )
            )

        state.update(run_writer_node(state=state, llm=self.llm, trace=self.trace))
        state.update(run_validator_node(state=state, trace=self.trace))

        return state

    def _build_langgraph_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        graph = StateGraph(PipelineState)

        graph.add_node(
            "planner",
            lambda state: run_planner_node(
                state=state,
                llm=self.llm,
                max_questions=self.max_questions,
                trace=self.trace,
            ),
        )
        graph.add_node(
            "retriever",
            lambda state: run_retriever_node(
                state=state,
                index=self.index,
                top_k_per_query=self.top_k_per_query,
                trace=self.trace,
            ),
        )
        graph.add_node(
            "critic",
            lambda state: run_critic_node(
                state=state,
                llm=self.llm,
                max_iterations=self.max_iterations,
                trace=self.trace,
            ),
        )
        graph.add_node(
            "writer",
            lambda state: run_writer_node(
                state=state,
                llm=self.llm,
                trace=self.trace,
            ),
        )
        graph.add_node(
            "validator",
            lambda state: run_validator_node(
                state=state,
                trace=self.trace,
            ),
        )

        graph.set_entry_point("planner")
        graph.add_edge("planner", "retriever")
        graph.add_edge("retriever", "critic")

        def route_critic(state: PipelineState) -> str:
            if state.get("critic_needs_more_research", False):
                return "retriever"
            return "writer"

        graph.add_conditional_edges(
            "critic",
            route_critic,
            {
                "retriever": "retriever",
                "writer": "writer",
            },
        )

        graph.add_edge("writer", "validator")
        graph.add_edge("validator", END)

        return graph.compile()


def build_pipeline_from_local_data(
    cards_csv: str,
    decks_dir: str,
    meta_json_paths: Optional[List[str]] = None,
    max_iterations: int = 3,
    max_questions: int = 5,
    top_k_per_query: int = 5,
    enable_semantic: bool = True,
    lexical_weight: float = 0.6,
    semantic_weight: float = 0.4,
    use_langgraph: bool = True,
    trace_path: Optional[str] = None,
) -> ResearchPipeline:
    chunks = build_domain_corpus(
        cards_csv=cards_csv,
        decks_dir=decks_dir,
        meta_json_paths=meta_json_paths,
    )
    if not chunks:
        raise ValueError(
            "No corpus chunks were built. Check cards_csv, decks_dir, and meta_json_paths."
        )

    index = HybridRetrievalIndex(
        chunks=chunks,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
        enable_semantic=enable_semantic,
    )

    return ResearchPipeline(
        index=index,
        max_iterations=max_iterations,
        max_questions=max_questions,
        top_k_per_query=top_k_per_query,
        use_langgraph=use_langgraph,
        trace_path=trace_path,
    )
