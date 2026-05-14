import unittest

from gcp_agent_runtime.contracts import RetrievedEvidence, RetrievalBundle, RetrievalPlan
from gcp_agent_runtime.rerank import RerankAgent


class RerankAgentTests(unittest.TestCase):
    def _bundle(self, score_offset: float) -> RetrievalBundle:
        plan = RetrievalPlan(rewritten_queries=["boros removal"], top_k_per_query=4, max_chunks=10)
        chunks = [
            RetrievedEvidence(
                doc_id="deck::boros",
                chunk_id="deck::boros::chunk-000",
                source="decklist",
                title="Boros Midrange",
                text="Lightning Bolt and instant-speed removal define the plan.",
                score=0.7 + score_offset,
                metadata={},
            ),
            RetrievedEvidence(
                doc_id="card::teferis-protection",
                chunk_id="card::teferis-protection::chunk-000",
                source="card_db",
                title="Teferi's Protection",
                text="Protect your board and life total from sweepers.",
                score=0.5 + score_offset,
                metadata={},
            ),
        ]
        return RetrievalBundle(plan=plan, chunks=chunks)

    def test_merge_and_rerank_deduplicates_chunks(self):
        agent = RerankAgent()
        first = self._bundle(0.0)
        second = self._bundle(0.2)

        merged = agent.merge_and_rerank(
            bundles=[first, second],
            query_text="boros removal protection",
        )
        chunk_ids = [item.chunk_id for item in merged.chunks]

        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertIn("deck::boros::chunk-000", chunk_ids)
        self.assertIn("card::teferis-protection::chunk-000", chunk_ids)
        self.assertIn("deck::boros::chunk-000", merged.rerank_scores)


if __name__ == "__main__":
    unittest.main()
