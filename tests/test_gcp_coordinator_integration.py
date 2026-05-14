import unittest

from gcp_agent_runtime.contracts import DeckCitation, DeckRecommendationRequest, RetrievedEvidence
from gcp_agent_runtime.coordinator import RootCoordinatorAgent
from gcp_agent_runtime.deck_plan import DeckPlanResult
from gcp_agent_runtime.model_routing import ModelSelection
from gcp_agent_runtime.retrieval import RetrieverClient


class FakeRetrieverClient(RetrieverClient):
    def __init__(self):
        self.call_count = 0

    def retrieve(self, plan):
        self.call_count += 1
        if self.call_count == 1:
            return [
                RetrievedEvidence(
                    doc_id="deck::boros",
                    chunk_id="deck::boros::chunk-000",
                    source="decklist",
                    title="Boros Midrange",
                    text="Boros decks rely on efficient removal and resilient threats.",
                    score=0.8,
                    metadata={"query_count": len(plan.rewritten_queries)},
                )
            ]
        return [
            RetrievedEvidence(
                doc_id="deck::boros",
                chunk_id="deck::boros::chunk-000",
                source="decklist",
                title="Boros Midrange",
                text="Boros decks rely on efficient removal and resilient threats.",
                score=0.85,
                metadata={},
            ),
            RetrievedEvidence(
                doc_id="card::teferis-protection",
                chunk_id="card::teferis-protection::chunk-000",
                source="card_db",
                title="Teferi's Protection",
                text="Protects your board and life total from sweepers and combat damage.",
                score=0.75,
                metadata={},
            ),
            RetrievedEvidence(
                doc_id="meta::boros",
                chunk_id="meta::boros::chunk-000",
                source="meta_json",
                title="Boros Meta Share",
                text="Boros archetypes gain win-rate from instant speed interaction packages.",
                score=0.72,
                metadata={},
            ),
        ]


class StubDeckPlanAgent:
    def plan_deck(self, request, bundle, predicted_confidence):
        del request
        citations = [DeckCitation.from_evidence(item) for item in bundle.chunks[:2]]
        return DeckPlanResult(
            summary="Stub plan generated.",
            recommended_decklist=["Lightning Bolt", "Mountain", "Plains"],
            key_claims=["Efficient removal improves tempo.", "Protection effects improve resilience."],
            citations=citations,
            confidence=max(0.4, predicted_confidence),
            model_selection=ModelSelection(
                model_id="gemini-2.5-flash",
                reason="test",
                escalated=False,
                lifecycle_fallback=False,
            ),
        )


class CoordinatorIntegrationTests(unittest.TestCase):
    def test_end_to_end_flow_triggers_second_pass_and_returns_schema(self):
        client = FakeRetrieverClient()
        coordinator = RootCoordinatorAgent(
            retriever_client=client,
            deck_planner=StubDeckPlanAgent(),
        )
        request = DeckRecommendationRequest(
            session_id="session-a",
            user_query="Recommend a Boros deck that survives board wipes.",
            format="commander",
            colors=["W", "R"],
            archetype_hint="midrange",
            must_include=["Teferi's Protection"],
            must_exclude=[],
            mode="deck_recommendation",
        )

        response = coordinator.run(request)

        self.assertGreaterEqual(client.call_count, 2)
        self.assertTrue(response.trace_id.startswith("trace-"))
        self.assertGreaterEqual(response.latency_ms, 0)
        self.assertGreater(len(response.recommended_decklist), 0)
        self.assertGreater(len(response.citations), 0)
        self.assertIn("summary", response.to_dict())


if __name__ == "__main__":
    unittest.main()
