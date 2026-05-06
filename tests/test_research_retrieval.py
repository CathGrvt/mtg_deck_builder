import unittest

from research_pipeline.models import DocumentChunk
from research_pipeline.retrieval.index import HybridRetrievalIndex


class HybridRetrievalTests(unittest.TestCase):
    def test_lexical_search_returns_relevant_chunk(self):
        chunks = [
            DocumentChunk(
                doc_id="deck::aggro",
                chunk_id="deck::aggro::chunk-000",
                source="decklist",
                title="Aggro Deck",
                text="Lightning Bolt and fast red pressure define this deck.",
            ),
            DocumentChunk(
                doc_id="deck::control",
                chunk_id="deck::control::chunk-000",
                source="decklist",
                title="Control Deck",
                text="Counterspell and board wipes slow the game down.",
            ),
        ]

        index = HybridRetrievalIndex(chunks=chunks, enable_semantic=False)
        results = index.search("red lightning removal", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "deck::aggro::chunk-000")
        self.assertGreaterEqual(results[0].score, 0.0)


if __name__ == "__main__":
    unittest.main()
