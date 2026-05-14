import unittest
from unittest.mock import Mock, patch

import requests

from mtg_ui_app.backend_client import (
    build_chat_backend_payload,
    build_research_backend_payload,
    call_backend_json,
)


class UiBackendClientTests(unittest.TestCase):
    def test_build_research_payload_shape(self):
        payload = build_research_backend_payload(
            session_id="s1",
            topic="topic",
            max_iterations=3,
            max_questions=5,
            top_k_per_query=6,
            enable_semantic=True,
            use_langgraph=False,
        )
        self.assertEqual(payload["session_id"], "s1")
        self.assertEqual(payload["topic"], "topic")
        self.assertEqual(payload["top_k_per_query"], 6)
        self.assertTrue(payload["enable_semantic"])
        self.assertFalse(payload["use_langgraph"])

    def test_build_chat_payload_normalizes_history(self):
        payload = build_chat_backend_payload(
            session_id="s1",
            question="hello",
            history=[
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": " "},
                "invalid",
            ],
            top_k=4,
        )
        self.assertEqual(payload["session_id"], "s1")
        self.assertEqual(payload["question"], "hello")
        self.assertEqual(payload["top_k"], 4)
        self.assertEqual(len(payload["history"]), 2)

    def test_call_backend_json_raises_on_http_failure(self):
        with patch("mtg_ui_app.backend_client.requests.post", side_effect=requests.RequestException("boom")):
            with self.assertRaises(requests.RequestException):
                call_backend_json("http://localhost:8080/v1/chat/respond", {"x": 1}, timeout_sec=5)

    def test_call_backend_json_returns_dict(self):
        mock_response = Mock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status.return_value = None
        with patch("mtg_ui_app.backend_client.requests.post", return_value=mock_response):
            payload = call_backend_json("http://localhost:8080/v1/chat/respond", {"x": 1}, timeout_sec=5)
        self.assertEqual(payload["ok"], True)


if __name__ == "__main__":
    unittest.main()
