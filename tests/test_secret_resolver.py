import os
import unittest
from unittest.mock import patch

from research_pipeline.secret_resolver import _SECRET_CACHE, resolve_openai_api_key


class SecretResolverTests(unittest.TestCase):
    def tearDown(self):
        _SECRET_CACHE.clear()

    def test_returns_existing_env_value(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test",
                "MTG_OPENAI_API_KEY_SECRET_RESOURCE": "",
                "MTG_OPENAI_API_KEY_SECRET": "",
            },
            clear=False,
        ):
            value = resolve_openai_api_key(api_key_env="OPENAI_API_KEY")
        self.assertEqual(value, "sk-test")

    def test_resolves_from_secret_manager_reference(self):
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
            with patch(
                "research_pipeline.secret_resolver._access_secret_version",
                return_value="sk-secret",
            ) as mocked:
                first = resolve_openai_api_key(api_key_env="OPENAI_API_KEY")
                second = resolve_openai_api_key(api_key_env="OPENAI_API_KEY")

        self.assertEqual(first, "sk-secret")
        self.assertEqual(second, "sk-secret")
        self.assertEqual(mocked.call_count, 1)
        mocked.assert_called_once_with("projects/proj-test/secrets/mtg-openai-api-key/versions/latest")

    def test_missing_project_for_short_ref_returns_empty(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "",
                "MTG_OPENAI_API_KEY_SECRET_RESOURCE": "mtg-openai-api-key",
                "MTG_OPENAI_API_KEY_SECRET": "",
                "GOOGLE_CLOUD_PROJECT": "",
            },
            clear=False,
        ):
            value = resolve_openai_api_key(api_key_env="OPENAI_API_KEY")
        self.assertEqual(value, "")


if __name__ == "__main__":
    unittest.main()
