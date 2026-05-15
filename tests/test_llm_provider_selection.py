import os
import unittest
from unittest.mock import patch

from gcp_agent_runtime.llm_provider import LLMProviderConfig, LLMProviderRuntime
from research_pipeline.llm import OpenAIChatLLM, RuleBasedLLM, build_default_llm


class LLMProviderSelectionTests(unittest.TestCase):
    def test_build_default_llm_rule_provider(self):
        llm = build_default_llm(provider="rule")
        self.assertIsInstance(llm, RuleBasedLLM)

    def test_build_default_llm_openai_without_key_falls_back(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "",
                "MTG_OPENAI_API_KEY_SECRET_RESOURCE": "",
                "MTG_OPENAI_API_KEY_SECRET": "",
            },
            clear=False,
        ):
            llm = build_default_llm(provider="openai", api_key_env="OPENAI_API_KEY")
        self.assertIsInstance(llm, RuleBasedLLM)

    def test_build_default_llm_vertex_without_sdk_falls_back(self):
        with patch("research_pipeline.llm._has_vertex_sdk", return_value=False):
            llm = build_default_llm(provider="vertex")
        self.assertIsInstance(llm, RuleBasedLLM)

    def test_build_default_llm_openai_uses_secret_manager_reference(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "",
                "MTG_OPENAI_API_KEY_SECRET_RESOURCE": "mtg-openai-api-key",
                "MTG_OPENAI_API_KEY_SECRET": "",
                "GOOGLE_CLOUD_PROJECT": "proj-test",
            },
            clear=False,
        ):
            with patch("mtg_shared.secrets._access_secret_version", return_value="sk-secret"):
                llm = build_default_llm(provider="openai", api_key_env="OPENAI_API_KEY")
        self.assertIsInstance(llm, OpenAIChatLLM)

    def test_runtime_reports_model_name(self):
        runtime = LLMProviderRuntime(
            LLMProviderConfig(
                provider="vertex",
                openai_model="gpt-4o-mini",
                vertex_model="gemini-2.5-flash",
            )
        )
        self.assertEqual(runtime.model_for_reporting(), "gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
