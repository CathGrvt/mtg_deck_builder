import unittest

from gcp_agent_runtime.adapter import CloudRunAgentAdapter
from gcp_agent_runtime.contracts import (
    DeckRecommendationResponse,
    SafetyVerdict,
)


class StubCoordinator:
    def run(self, request):
        del request
        return DeckRecommendationResponse(
            summary="ok",
            recommended_decklist=["Lightning Bolt", "Mountain"],
            key_claims=["Claim A"],
            citations=[],
            confidence=0.7,
            safety_verdict=SafetyVerdict(status="allow", reasons=[], risk_score=0.0, blocked=False),
            trace_id="trace-123",
            latency_ms=45,
            model_used="gemini-2.5-flash",
        )


class AdapterSchemaTests(unittest.TestCase):
    def test_adapter_returns_contract_shape(self):
        adapter = CloudRunAgentAdapter(coordinator=StubCoordinator())
        payload = {
            "session_id": "s-1",
            "user_query": "recommend deck",
            "format": "commander",
            "colors": ["W", "R"],
            "archetype_hint": "midrange",
            "must_include": [],
            "must_exclude": [],
            "mode": "deck_recommendation",
        }
        response = adapter.handle_request(payload)

        self.assertIn("summary", response)
        self.assertIn("recommended_decklist", response)
        self.assertIn("key_claims", response)
        self.assertIn("citations", response)
        self.assertIn("confidence", response)
        self.assertIn("safety_verdict", response)
        self.assertIn("trace_id", response)
        self.assertIn("latency_ms", response)

    def test_adapter_rejects_invalid_mode(self):
        adapter = CloudRunAgentAdapter(coordinator=StubCoordinator())
        payload = {
            "session_id": "s-1",
            "user_query": "recommend deck",
            "mode": "unsupported",
        }
        with self.assertRaises(ValueError):
            adapter.handle_request(payload)


if __name__ == "__main__":
    unittest.main()
