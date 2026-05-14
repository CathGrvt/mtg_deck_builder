import unittest

from gcp_agent_runtime.adapter import AdapterSettings, CloudRunAgentAdapter
from gcp_agent_runtime.contracts import DeckRecommendationResponse, SafetyVerdict


class StubCoordinator:
    def __init__(self):
        self.calls = 0

    def run(self, request):
        del request
        self.calls += 1
        return DeckRecommendationResponse(
            summary="local",
            recommended_decklist=["Mountain"],
            key_claims=["Local claim"],
            citations=[],
            confidence=0.5,
            safety_verdict=SafetyVerdict(status="allow", reasons=[], risk_score=0.0, blocked=False),
            trace_id="trace-local",
            latency_ms=10,
            model_used="gemini-2.5-flash",
        )


class StubVertexClient:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result or {"summary": "vertex", "recommended_decklist": ["Island"]}
        self.error = error
        self.calls = 0

    def recommend(self, payload):
        del payload
        self.calls += 1
        if self.error is not None:
            raise self.error
        return dict(self.result)


class StubResearchService:
    def run(self, payload):
        del payload
        return {
            "report": {"topic": "x", "summary": "ok", "claims": [], "open_questions": [], "validation": {}},
            "validation": {},
            "latency_ms": 5,
            "trace_id": "trace-r",
            "model_used": "rule-based",
            "corpus_stats": {"chunks": 1, "sources": 1},
        }


class StubChatService:
    def run(self, payload):
        del payload
        return {
            "answer": "ok",
            "evidence": [],
            "latency_ms": 6,
            "trace_id": "trace-c",
            "model_used": "rule-based",
        }


class AdapterModeTests(unittest.TestCase):
    def _payload(self):
        return {
            "session_id": "s-1",
            "user_query": "recommend deck",
            "format": "commander",
            "colors": ["W", "R"],
            "archetype_hint": "midrange",
            "must_include": [],
            "must_exclude": [],
            "mode": "deck_recommendation",
        }

    def test_local_mode_uses_local_coordinator(self):
        coordinator = StubCoordinator()
        vertex = StubVertexClient()
        adapter = CloudRunAgentAdapter(
            coordinator=coordinator,
            settings=AdapterSettings(backend_mode="local", vertex_fallback_to_local=True),
            vertex_client=vertex,
            research_service=StubResearchService(),
            chat_service=StubChatService(),
        )

        response = adapter.handle_request(self._payload())
        self.assertEqual(response["summary"], "local")
        self.assertEqual(coordinator.calls, 1)
        self.assertEqual(vertex.calls, 0)

    def test_vertex_mode_uses_vertex_client(self):
        coordinator = StubCoordinator()
        vertex = StubVertexClient(result={"summary": "vertex", "recommended_decklist": ["Island"]})
        adapter = CloudRunAgentAdapter(
            coordinator=coordinator,
            settings=AdapterSettings(backend_mode="vertex", vertex_fallback_to_local=True),
            vertex_client=vertex,
            research_service=StubResearchService(),
            chat_service=StubChatService(),
        )

        response = adapter.handle_request(self._payload())
        self.assertEqual(response["summary"], "vertex")
        self.assertEqual(vertex.calls, 1)
        self.assertEqual(coordinator.calls, 0)

    def test_vertex_failure_falls_back_to_local_when_enabled(self):
        coordinator = StubCoordinator()
        vertex = StubVertexClient(error=RuntimeError("vertex failure"))
        adapter = CloudRunAgentAdapter(
            coordinator=coordinator,
            settings=AdapterSettings(backend_mode="vertex", vertex_fallback_to_local=True),
            vertex_client=vertex,
            research_service=StubResearchService(),
            chat_service=StubChatService(),
        )

        response = adapter.handle_request(self._payload())
        self.assertEqual(response["summary"], "local")
        self.assertEqual(vertex.calls, 1)
        self.assertEqual(coordinator.calls, 1)

    def test_vertex_failure_raises_when_fallback_disabled(self):
        adapter = CloudRunAgentAdapter(
            coordinator=StubCoordinator(),
            settings=AdapterSettings(backend_mode="vertex", vertex_fallback_to_local=False),
            vertex_client=StubVertexClient(error=RuntimeError("vertex failure")),
            research_service=StubResearchService(),
            chat_service=StubChatService(),
        )

        with self.assertRaises(RuntimeError):
            adapter.handle_request(self._payload())

    def test_research_and_chat_handlers_return_expected_shape(self):
        adapter = CloudRunAgentAdapter(
            coordinator=StubCoordinator(),
            settings=AdapterSettings(backend_mode="local", vertex_fallback_to_local=True),
            vertex_client=StubVertexClient(),
            research_service=StubResearchService(),
            chat_service=StubChatService(),
        )

        research = adapter.handle_research({"session_id": "s", "topic": "t"})
        self.assertIn("report", research)
        self.assertIn("validation", research)
        self.assertIn("latency_ms", research)
        self.assertIn("trace_id", research)
        self.assertIn("model_used", research)
        self.assertIn("corpus_stats", research)

        chat = adapter.handle_chat({"session_id": "s", "question": "q"})
        self.assertIn("answer", chat)
        self.assertIn("evidence", chat)
        self.assertIn("latency_ms", chat)
        self.assertIn("trace_id", chat)
        self.assertIn("model_used", chat)


if __name__ == "__main__":
    unittest.main()
