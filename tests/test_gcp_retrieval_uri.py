import json
import tempfile
import unittest
from pathlib import Path

from gcp_agent_runtime.contracts import RetrievalPlan
from gcp_agent_runtime.retrieval import LocalHybridRetrieverClient, LocalRetrieverConfig


class RetrieverUriTests(unittest.TestCase):
    def test_prebuilt_corpus_uri_is_used(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = Path(tmp_dir) / "rag_corpus.jsonl"
            row = {
                "doc_id": "card::lightning-bolt",
                "chunk_id": "card::lightning-bolt::chunk-000",
                "source": "card_db",
                "title": "Lightning Bolt",
                "text": "Lightning Bolt deals 3 damage to any target.",
                "metadata": {},
            }
            corpus_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            client = LocalHybridRetrieverClient(
                LocalRetrieverConfig(
                    rag_corpus_uri=str(corpus_path),
                    enable_semantic=False,
                )
            )
            plan = RetrievalPlan(
                rewritten_queries=["lightning damage"],
                top_k_per_query=4,
                max_chunks=4,
            )

            hits = client.retrieve(plan)
            self.assertGreaterEqual(len(hits), 1)
            self.assertEqual(hits[0].chunk_id, "card::lightning-bolt::chunk-000")


if __name__ == "__main__":
    unittest.main()
