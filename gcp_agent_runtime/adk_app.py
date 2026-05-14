from __future__ import annotations

from typing import Any, Dict, Optional

from gcp_agent_runtime.adapter import AdapterSettings, CloudRunAgentAdapter
from gcp_agent_runtime.coordinator import RootCoordinatorAgent
from gcp_agent_runtime.model_routing import ModelRoutingConfig


def _import_adk():
    try:
        from google.adk.agents import LlmAgent
        from google.adk.tools import FunctionTool
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "google-adk is not installed. Install requirements-gcp.txt for ADK deployment."
        ) from exc
    return LlmAgent, FunctionTool


def build_adk_root_agent(
    coordinator: Optional[RootCoordinatorAgent] = None,
    routing_config: Optional[ModelRoutingConfig] = None,
):
    """
    Build ADK multi-agent runtime with explicit decomposition:
    RootCoordinatorAgent, QueryRewriteAgent, RetrieverAgent, RerankAgent,
    CriticAgent, DeckPlanAgent, SafetyGateAgent.
    """
    del routing_config  # coordinator already owns routing; kept for explicit API stability.
    LlmAgent, FunctionTool = _import_adk()
    resolved = coordinator or RootCoordinatorAgent()
    # Force local execution inside Agent Engine tool handlers to avoid recursive proxy loops.
    adapter = CloudRunAgentAdapter(
        coordinator=resolved,
        settings=AdapterSettings(
            backend_mode="local",
            vertex_fallback_to_local=True,
            vertex_proxy_research=False,
            vertex_proxy_chat=False,
        ),
    )

    def run_deck_recommendation(
        session_id: str,
        user_query: str,
        format: str = "commander",
        colors: list[str] | None = None,
        archetype_hint: str | None = None,
        must_include: list[str] | None = None,
        must_exclude: list[str] | None = None,
        mode: str = "deck_recommendation",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "user_query": user_query,
            "format": format,
            "colors": colors or [],
            "archetype_hint": archetype_hint,
            "must_include": must_include or [],
            "must_exclude": must_exclude or [],
            "mode": mode,
        }
        return adapter.handle_request(payload)

    def run_research(
        session_id: str,
        topic: str,
        max_iterations: int = 3,
        max_questions: int = 5,
        top_k_per_query: int = 5,
        enable_semantic: bool = True,
        use_langgraph: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "topic": topic,
            "max_iterations": int(max_iterations),
            "max_questions": int(max_questions),
            "top_k_per_query": int(top_k_per_query),
            "enable_semantic": bool(enable_semantic),
            "use_langgraph": bool(use_langgraph),
        }
        return adapter.handle_research(payload)

    def run_chat_response(
        session_id: str,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int = 6,
        enable_clarification: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "question": question,
            "history": list(history or []),
            "top_k": int(top_k),
            "enable_clarification": bool(enable_clarification),
        }
        return adapter.handle_chat(payload)

    recommendation_tool = FunctionTool(run_deck_recommendation)
    research_tool = FunctionTool(run_research)
    chat_tool = FunctionTool(run_chat_response)

    query_rewrite_agent = LlmAgent(
        name="QueryRewriteAgent",
        model="gemini-2.5-flash",
        description="Generates alternative retrieval queries and metadata intents.",
        instruction=(
            "Rewrite user requests into retrieval-oriented subqueries while preserving intent, "
            "format constraints, colors, and include/exclude cards."
        ),
    )

    retriever_agent = LlmAgent(
        name="RetrieverAgent",
        model="gemini-2.5-flash",
        description="Runs managed RAG retrieval over cards, decks, and meta corpora.",
        instruction=(
            "Select and retrieve high-signal evidence from configured corpora with metadata filters."
        ),
    )

    rerank_agent = LlmAgent(
        name="RerankAgent",
        model="gemini-2.5-flash",
        description="Reranks and compresses retrieved evidence into citation-ready context.",
        instruction=(
            "Prioritize relevance, remove duplicates, and preserve citation integrity."
        ),
    )

    critic_agent = LlmAgent(
        name="CriticAgent",
        model="gemini-2.5-flash",
        description="Assesses evidence sufficiency and requests second-pass retrieval when needed.",
        instruction=(
            "Check whether evidence supports requested deck constraints. Trigger extra retrieval if coverage is thin."
        ),
    )

    deck_plan_agent = LlmAgent(
        name="DeckPlanAgent",
        model="gemini-2.5-flash",
        description="Builds deck recommendations and rationale from grounded evidence.",
        instruction=(
            "Produce a concrete deck recommendation grounded in provided evidence with explicit citations."
        ),
    )

    safety_agent = LlmAgent(
        name="SafetyGateAgent",
        model="gemini-2.5-flash",
        description="Applies policy checks and blocks unsafe or misaligned behavior.",
        instruction=(
            "Enforce policy and intent alignment before returning final responses."
        ),
    )

    root = LlmAgent(
        name="RootCoordinatorAgent",
        model="gemini-2.5-flash",
        description=(
            "Routes deck recommendation, research, and chat requests through specialized tool paths."
        ),
        instruction=(
            "Coordinate specialized subagents and call the matching tool: "
            "run_deck_recommendation for deck outputs, run_research for structured research reports, "
            "and run_chat_response for conversational answers. "
            "Escalate complex reasoning to gemini-2.5-pro only when constraints are conflicting or confidence is low."
        ),
        tools=[recommendation_tool, research_tool, chat_tool],
        sub_agents=[
            query_rewrite_agent,
            retriever_agent,
            rerank_agent,
            critic_agent,
            deck_plan_agent,
            safety_agent,
        ],
    )
    return root


def build_agent_engine_app(root_agent=None):
    try:
        from vertexai import agent_engines
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "google-cloud-aiplatform is not installed. Install requirements-gcp.txt for Agent Engine deployment."
        ) from exc

    resolved_root = root_agent or build_adk_root_agent()
    return agent_engines.AdkApp(
        agent=resolved_root,
        enable_tracing=True,
    )
