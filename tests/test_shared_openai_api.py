import unittest
from unittest.mock import Mock, patch

import requests

from mtg_shared.openai_api import (
    chat_completion_json_object,
    safe_chat_completion_content,
)


class SharedOpenAIAPITests(unittest.TestCase):
    def _mock_response(self, payload):
        response = Mock()
        response.raise_for_status = Mock()
        response.json = Mock(return_value=payload)
        return response

    def test_chat_completion_json_object_success(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"scores": {"Sol Ring": 98}}',
                    }
                }
            ]
        }
        with patch("mtg_shared.openai_api.requests.post", return_value=self._mock_response(payload)):
            result = chat_completion_json_object(
                api_key="sk-test",
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hello"}],
            )
        self.assertEqual(result, {"scores": {"Sol Ring": 98}})

    def test_chat_completion_json_object_malformed_json_returns_empty(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "not json",
                    }
                }
            ]
        }
        with patch("mtg_shared.openai_api.requests.post", return_value=self._mock_response(payload)):
            result = chat_completion_json_object(
                api_key="sk-test",
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hello"}],
            )
        self.assertEqual(result, {})

    def test_safe_chat_completion_content_empty_choices_uses_empty_message(self):
        payload = {"choices": []}
        with patch("mtg_shared.openai_api.requests.post", return_value=self._mock_response(payload)):
            result = safe_chat_completion_content(
                api_key="sk-test",
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hello"}],
                empty_message="No response",
                fallback_message="fallback",
            )
        self.assertEqual(result, "No response")

    def test_safe_chat_completion_content_timeout_returns_fallback(self):
        with patch(
            "mtg_shared.openai_api.requests.post",
            side_effect=requests.Timeout("timeout"),
        ):
            result = safe_chat_completion_content(
                api_key="sk-test",
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hello"}],
                fallback_message="fallback",
            )
        self.assertEqual(result, "fallback")


if __name__ == "__main__":
    unittest.main()
