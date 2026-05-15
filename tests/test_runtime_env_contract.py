import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from deploy_agent_engine import _collect_agent_runtime_env
from mtg_shared.runtime_env import runtime_env_names


class RuntimeEnvContractTests(unittest.TestCase):
    def test_deploy_collector_matches_agent_engine_contract(self):
        keys = runtime_env_names("agent-engine")
        seeded = {key: f"value-{idx}" for idx, key in enumerate(keys, start=1)}
        with patch.dict(os.environ, seeded, clear=False):
            collected = _collect_agent_runtime_env()
        self.assertEqual(set(collected.keys()), set(keys))

    def test_workflow_env_block_covers_backend_contract(self):
        workflow = Path(".github/workflows/deploy-gcp.yml").read_text(encoding="utf-8")
        env_keys = set(re.findall(r"^\s{6}([A-Z0-9_]+):\s*\$\{\{", workflow, flags=re.MULTILINE))
        missing = [key for key in runtime_env_names("backend") if key not in env_keys]
        self.assertEqual(missing, [], msg=f"Missing backend env keys in workflow env block: {missing}")


if __name__ == "__main__":
    unittest.main()
