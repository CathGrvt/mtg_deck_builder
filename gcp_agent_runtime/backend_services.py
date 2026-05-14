from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from gcp_agent_runtime.contracts import SafetyVerdict
from gcp_agent_runtime.llm_provider import LLMProviderRuntime, build_rule_based_chat_answer
from gcp_agent_runtime.retrieval import LocalHybridRetrieverClient, LocalRetrieverConfig
from gcp_agent_runtime.safety import SafetyGateAgent
from research_pipeline.graph import ResearchPipeline
from research_pipeline.models import RetrievedChunk


def _bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _merge_safety(primary: SafetyVerdict, secondary: SafetyVerdict) -> SafetyVerdict:
    status = primary.status
    blocked = primary.blocked or secondary.blocked
    if blocked:
        status = "blocked"
    elif primary.status == "review" or secondary.status == "review":
        status = "review"
    return SafetyVerdict(
        status=status,
        reasons=list(primary.reasons) + list(secondary.reasons),
        risk_score=max(float(primary.risk_score), float(secondary.risk_score)),
        blocked=blocked,
    )


def _history_items(raw_history: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_history, list):
        return []
    rows: List[Dict[str, str]] = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        role = "assistant" if str(item.get("role")) == "assistant" else "user"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        rows.append({"role": role, "content": content})
    return rows


class ResearchBackendService:
    def __init__(
        self,
        retriever_client: Optional[LocalHybridRetrieverClient] = None,
        safety_gate: Optional[SafetyGateAgent] = None,
        llm_runtime: Optional[LLMProviderRuntime] = None,
    ):
        self.retriever_client = retriever_client or LocalHybridRetrieverClient(LocalRetrieverConfig.from_env())
        self.safety_gate = safety_gate or SafetyGateAgent()
        self.llm_runtime = llm_runtime or LLMProviderRuntime()

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id is required.")

        topic = str(payload.get("topic", "")).strip()
        if not topic:
            raise ValueError("topic is required.")

        max_iterations = _bounded_int(payload.get("max_iterations"), default=3, minimum=1, maximum=8)
        max_questions = _bounded_int(payload.get("max_questions"), default=5, minimum=1, maximum=12)
        top_k_per_query = _bounded_int(payload.get("top_k_per_query"), default=5, minimum=1, maximum=20)
        enable_semantic = _bool_value(
            payload.get("enable_semantic"),
            default=bool(self.retriever_client.config.enable_semantic),
        )
        use_langgraph = _bool_value(payload.get("use_langgraph"), default=True)

        started = time.perf_counter()
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"

        pre_safety = self.safety_gate.evaluate_request(text=topic, mode="research_copilot")
        if pre_safety.blocked:
            latency_ms = int(round((time.perf_counter() - started) * 1000))
            return {
                "report": {
                    "topic": topic,
                    "summary": "Request blocked by safety guardrails.",
                    "claims": [],
                    "open_questions": [],
                    "validation": {},
                },
                "validation": {},
                "latency_ms": latency_ms,
                "trace_id": trace_id,
                "model_used": self.llm_runtime.model_for_reporting(),
                "corpus_stats": {"chunks": 0, "sources": 0},
                "safety_verdict": pre_safety.to_dict(),
            }

        retriever_client = self.retriever_client
        if enable_semantic != bool(self.retriever_client.config.enable_semantic):
            base_cfg = self.retriever_client.config
            retriever_client = LocalHybridRetrieverClient(
                LocalRetrieverConfig(
                    cards_csv=base_cfg.cards_csv,
                    decks_dir=base_cfg.decks_dir,
                    meta_json_paths=list(base_cfg.meta_json_paths) if base_cfg.meta_json_paths else None,
                    rag_corpus_uri=base_cfg.rag_corpus_uri,
                    enable_semantic=enable_semantic,
                    lexical_weight=base_cfg.lexical_weight,
                    semantic_weight=base_cfg.semantic_weight,
                )
            )
        index = retriever_client.get_index()
        llm = self.llm_runtime.build_research_llm()

        pipeline = ResearchPipeline(
            index=index,
            llm=llm,
            max_iterations=max_iterations,
            max_questions=max_questions,
            top_k_per_query=top_k_per_query,
            use_langgraph=use_langgraph,
            trace_path=None,
        )
        output = pipeline.run(topic)
        report = dict(output.get("report", {}))

        claims = report.get("claims", []) if isinstance(report.get("claims", []), list) else []
        post_safety = self.safety_gate.evaluate_output(
            text="\n".join(
                [str(report.get("summary", ""))] + [str(item.get("claim", "")) for item in claims if isinstance(item, dict)]
            )
        )
        merged_safety = _merge_safety(pre_safety, post_safety)

        if merged_safety.blocked:
            report = {
                "topic": topic,
                "summary": "Request blocked by safety guardrails.",
                "claims": [],
                "open_questions": [],
                "validation": {},
            }

        latency_ms = int(round((time.perf_counter() - started) * 1000))
        return {
            "report": report,
            "validation": dict(report.get("validation", {})),
            "latency_ms": latency_ms,
            "trace_id": trace_id,
            "model_used": self.llm_runtime.model_for_reporting(),
            "corpus_stats": {
                "chunks": len(index.chunks),
                "sources": len({item.doc_id for item in index.chunks}),
            },
            "safety_verdict": merged_safety.to_dict(),
        }


class ChatBackendService:
    def __init__(
        self,
        retriever_client: Optional[LocalHybridRetrieverClient] = None,
        safety_gate: Optional[SafetyGateAgent] = None,
        llm_runtime: Optional[LLMProviderRuntime] = None,
    ):
        self.retriever_client = retriever_client or LocalHybridRetrieverClient(LocalRetrieverConfig.from_env())
        self.safety_gate = safety_gate or SafetyGateAgent()
        self.llm_runtime = llm_runtime or LLMProviderRuntime()

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id is required.")

        question = str(payload.get("question", "")).strip()
        if not question:
            raise ValueError("question is required.")

        top_k = _bounded_int(payload.get("top_k"), default=6, minimum=1, maximum=20)
        history = _history_items(payload.get("history", []))

        started = time.perf_counter()
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"

        pre_safety = self.safety_gate.evaluate_request(text=question, mode="research_copilot")
        if pre_safety.blocked:
            latency_ms = int(round((time.perf_counter() - started) * 1000))
            return {
                "answer": "Request blocked by safety guardrails.",
                "evidence": [],
                "latency_ms": latency_ms,
                "trace_id": trace_id,
                "model_used": self.llm_runtime.model_for_reporting(),
                "safety_verdict": pre_safety.to_dict(),
            }

        index = self.retriever_client.get_index()
        retrieved = index.search(question, top_k=top_k)
        evidence = [item.to_dict() for item in retrieved]

        if not retrieved:
            answer = (
                "I could not find relevant corpus evidence for that question. "
                "Try rephrasing with a card, archetype, or format keyword."
            )
        else:
            llm_answer = self.llm_runtime.try_chat_answer(
                question=question,
                history=history,
                evidence=retrieved,
            )
            answer = llm_answer.strip() if llm_answer and llm_answer.strip() else ""
            if not answer:
                answer = build_rule_based_chat_answer(question=question, retrieved=retrieved)

        post_safety = self.safety_gate.evaluate_output(text=answer)
        merged_safety = _merge_safety(pre_safety, post_safety)
        if merged_safety.blocked:
            answer = "Request blocked by safety guardrails."
            evidence = []

        latency_ms = int(round((time.perf_counter() - started) * 1000))
        return {
            "answer": answer,
            "evidence": evidence,
            "latency_ms": latency_ms,
            "trace_id": trace_id,
            "model_used": self.llm_runtime.model_for_reporting(),
            "safety_verdict": merged_safety.to_dict(),
        }
