import unittest

from research_pipeline.llm import RuleBasedLLM
from research_pipeline.models import RetrievedChunk


def _chunk(idx: int, score: float, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=f"doc-{idx}",
        chunk_id=f"chunk-{idx}",
        source="test",
        title=f"Chunk {idx}",
        text=text,
        score=score,
        metadata={},
    )


class RuleWriterCandidateTests(unittest.TestCase):
    def test_writer_scans_beyond_top_8_chunks_for_topical_candidates(self):
        topic = "How should commander decks size graveyard hate slots without overloading reactive cards?"
        chunks = [
            _chunk(idx=1, score=0.99, text="Oracle text: Create a 1/1 white Soldier creature token."),
            _chunk(idx=2, score=0.98, text="Oracle text: Target creature gets +2/+2 until end of turn."),
            _chunk(idx=3, score=0.97, text="Oracle text: Draw two cards, then discard a card."),
            _chunk(idx=4, score=0.96, text="Oracle text: Counter target spell unless its controller pays {2}."),
            _chunk(idx=5, score=0.95, text="Oracle text: Add one mana of any color."),
            _chunk(idx=6, score=0.94, text="Oracle text: Scry 2, then draw a card."),
            _chunk(idx=7, score=0.93, text="Oracle text: Creatures you control get +1/+1 until end of turn."),
            _chunk(idx=8, score=0.92, text="Oracle text: Return target creature to its owner's hand."),
            _chunk(idx=9, score=0.91, text="Oracle text: Exile target artifact."),
            _chunk(
                idx=10,
                score=0.35,
                text=(
                    "Observed list patterns suggest most decks stay near three graveyard hate slots "
                    "to avoid overloading reactive cards."
                ),
            ),
        ]

        report = RuleBasedLLM().write_report(topic=topic, retrieved_chunks=chunks, gaps=[])

        self.assertTrue(report.claims)
        self.assertTrue(
            any("graveyard hate slots" in claim.claim.lower() for claim in report.claims)
        )


if __name__ == "__main__":
    unittest.main()
