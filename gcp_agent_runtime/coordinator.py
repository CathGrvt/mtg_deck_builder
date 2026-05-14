from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

from gcp_agent_runtime.contracts import (
    DeckRecommendationRequest,
    DeckRecommendationResponse,
    SafetyVerdict,
)
from gcp_agent_runtime.critic import CriticAgent
from gcp_agent_runtime.deck_plan import DeckPlanAgent
from gcp_agent_runtime.model_routing import ModelLifecycleGuard, ModelRoutingConfig
from gcp_agent_runtime.query_rewrite import QueryRewriteAgent
from gcp_agent_runtime.rerank import RerankAgent
from gcp_agent_runtime.retrieval import LocalHybridRetrieverClient, LocalRetrieverConfig, RetrieverAgent, RetrieverClient
from gcp_agent_runtime.safety import SafetyGateAgent


@dataclass
class CoordinatorConfig:
    enable_second_pass: bool = True


class RootCoordinatorAgent:
    def __init__(
        self,
        retriever_client: Optional[RetrieverClient] = None,
        query_rewriter: Optional[QueryRewriteAgent] = None,
        retriever: Optional[RetrieverAgent] = None,
        reranker: Optional[RerankAgent] = None,
        critic: Optional[CriticAgent] = None,
        safety_gate: Optional[SafetyGateAgent] = None,
        deck_planner: Optional[DeckPlanAgent] = None,
        routing_config: Optional[ModelRoutingConfig] = None,
        config: Optional[CoordinatorConfig] = None,
    ):
        self.config = config or CoordinatorConfig()
        self.query_rewriter = query_rewriter or QueryRewriteAgent()
        client = retriever_client or LocalHybridRetrieverClient(config=LocalRetrieverConfig.from_env())
        self.retriever = retriever or RetrieverAgent(client=client)
        self.reranker = reranker or RerankAgent()
        self.critic = critic or CriticAgent()
        self.safety_gate = safety_gate or SafetyGateAgent()
        routing_guard = ModelLifecycleGuard(config=routing_config or ModelRoutingConfig())
        self.deck_planner = deck_planner or DeckPlanAgent(routing_guard=routing_guard)

    @staticmethod
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

    def run(self, request: DeckRecommendationRequest) -> DeckRecommendationResponse:
        started = time.perf_counter()
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"

        pre_safety = self.safety_gate.evaluate_request(text=request.user_query, mode=request.mode)
        if pre_safety.blocked:
            latency_ms = int(round((time.perf_counter() - started) * 1000))
            return DeckRecommendationResponse.blocked(
                reason="; ".join(pre_safety.reasons) or "request blocked by policy",
                trace_id=trace_id,
                latency_ms=latency_ms,
            )

        first_plan = self.query_rewriter.build_retrieval_plan(request=request)
        first_bundle = self.retriever.retrieve(first_plan)
        first_bundle = self.reranker.rerank(first_bundle, query_text=request.user_query)

        critic_outcome = self.critic.evaluate(request=request, bundle=first_bundle)
        final_bundle = first_bundle

        if self.config.enable_second_pass and critic_outcome.needs_second_pass:
            second_plan = self.query_rewriter.build_retrieval_plan(
                request=request,
                additional_queries=critic_outcome.gaps,
            )
            second_bundle = self.retriever.retrieve(second_plan)
            second_bundle = self.reranker.rerank(second_bundle, query_text=request.user_query)
            final_bundle = self.reranker.merge_and_rerank(
                bundles=[first_bundle, second_bundle],
                query_text=request.user_query,
            )
            critic_outcome = self.critic.evaluate(request=request, bundle=final_bundle)

        plan_result = self.deck_planner.plan_deck(
            request=request,
            bundle=final_bundle,
            predicted_confidence=critic_outcome.predicted_confidence,
        )

        post_safety = self.safety_gate.evaluate_output(
            text="\n".join([plan_result.summary] + list(plan_result.key_claims))
        )
        merged_safety = self._merge_safety(pre_safety, post_safety)
        if merged_safety.blocked:
            latency_ms = int(round((time.perf_counter() - started) * 1000))
            return DeckRecommendationResponse.blocked(
                reason="; ".join(merged_safety.reasons) or "output blocked by policy",
                trace_id=trace_id,
                latency_ms=latency_ms,
            )

        latency_ms = int(round((time.perf_counter() - started) * 1000))
        return DeckRecommendationResponse(
            summary=plan_result.summary,
            recommended_decklist=plan_result.recommended_decklist,
            key_claims=plan_result.key_claims,
            citations=plan_result.citations,
            confidence=plan_result.confidence,
            safety_verdict=merged_safety,
            trace_id=trace_id,
            latency_ms=latency_ms,
            model_used=plan_result.model_selection.model_id,
        )
