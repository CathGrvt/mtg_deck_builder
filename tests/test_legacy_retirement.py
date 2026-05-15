import unittest
from pathlib import Path


RETIRED_SCRIPTS = [
    "analyze_meta_old_try_to_parse.py",
    "analyze_meta_using_keywords.py",
    "semantics_meta_analysis.py",
    "integrated_deck_name_analyzer.py",
    "consolidated_meta_analysis.py",
]

DOC_PATHS = [
    Path("README.md"),
    Path("docs/repository_master_report.md"),
    Path("docs/gcp_adk_vertex_deployment.md"),
    Path(".github/workflows/deploy-gcp.yml"),
]


class LegacyRetirementSmokeTests(unittest.TestCase):
    def test_retired_scripts_are_removed(self):
        for script in RETIRED_SCRIPTS:
            with self.subTest(script=script):
                self.assertFalse(Path(script).exists())

    def test_retired_scripts_not_referenced_in_supported_docs_or_workflows(self):
        for path in DOC_PATHS:
            content = path.read_text(encoding="utf-8")
            for script in RETIRED_SCRIPTS:
                with self.subTest(path=str(path), script=script):
                    self.assertNotIn(script, content)


if __name__ == "__main__":
    unittest.main()
