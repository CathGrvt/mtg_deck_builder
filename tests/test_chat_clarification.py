import unittest

from gcp_agent_runtime.backend_services import ChatBackendService
from research_pipeline.models import RetrievedChunk


class StubIndex:
    def __init__(self, hits):
        self.hits = list(hits)

    def search(self, query, top_k):
        del query, top_k
        return list(self.hits)


class StubRetrieverClient:
    def __init__(self, hits):
        self._index = StubIndex(hits)

    def get_index(self):
        return self._index


class StubLLMRuntime:
    def __init__(self, answer=""):
        self.answer = answer

    def model_for_reporting(self):
        return "stub-model"

    def try_chat_answer(self, question, history, evidence, temperature=0.2):
        del question, history, evidence, temperature
        return self.answer


class ChatClarificationTests(unittest.TestCase):
    def test_ambiguous_question_triggers_clarification(self):
        service = ChatBackendService(
            retriever_client=StubRetrieverClient(hits=[]),
            llm_runtime=StubLLMRuntime(answer=""),
        )

        response = service.run(
            {
                "session_id": "s-1",
                "question": "help me",
            }
        )

        self.assertTrue(response.get("needs_clarification", False))
        self.assertTrue(str(response.get("clarifying_question", "")).startswith("Quick clarification:"))
        self.assertEqual(response.get("answer"), response.get("clarifying_question"))

    def test_specific_question_with_evidence_returns_answer(self):
        hit = RetrievedChunk(
            doc_id="card::sol-ring",
            chunk_id="card::sol-ring::chunk-000",
            source="card_db",
            title="Sol Ring",
            text="Sol Ring provides fast mana in Commander decks.",
            score=0.8,
            metadata={},
        )
        service = ChatBackendService(
            retriever_client=StubRetrieverClient(hits=[hit]),
            llm_runtime=StubLLMRuntime(answer="Use Sol Ring for acceleration."),
        )

        response = service.run(
            {
                "session_id": "s-1",
                "question": "In commander aggro, is Sol Ring still strong?",
            }
        )

        self.assertFalse(response.get("needs_clarification", True))
        self.assertEqual(response.get("clarifying_question"), "")
        self.assertIn("Sol Ring", response.get("answer", ""))


if __name__ == "__main__":
    unittest.main()
