import unittest

from research_pipeline.grounding import (
    best_overlap_against_claim,
    build_token_idf,
    combined_overlap_score,
    lexical_overlap_score,
    topic_alignment,
)


class GroundingOverlapTests(unittest.TestCase):
    def test_span_scoring_beats_full_chunk_jaccard_for_faithful_sentence(self):
        claim = "Lightning Bolt deals 3 damage to any target."
        evidence = (
            "Lightning Bolt is a staple interaction spell in many decks. "
            "Lightning Bolt deals 3 damage to any target at instant speed. "
            "It is commonly played in low-curve red shells."
        )

        full_chunk_jaccard = lexical_overlap_score(claim, evidence)
        span_score, span_text = best_overlap_against_claim(claim, [evidence])

        self.assertGreater(span_score, full_chunk_jaccard)
        self.assertIn("deals 3 damage", span_text.lower())

    def test_combined_overlap_prefers_higher_claim_token_coverage(self):
        claim = "Card draw and recursion improve resilience."
        weak = "Resilience matters over long games."
        strong = "Card draw and recursion improve resilience in grindy matchups."

        weak_score = combined_overlap_score(claim, weak)
        strong_score = combined_overlap_score(claim, strong)

        self.assertGreater(strong_score, weak_score)

    def test_topic_alignment_rejects_off_topic_claim(self):
        topic = "What interaction package sizes are common in top-performing commander decklists?"
        on_topic = "Interaction package sizes in top-performing lists cluster around 10 to 12 slots."
        off_topic = "Oracle text: target creature gets +3/+3 until end of turn."

        on_score, on_terms = topic_alignment(on_topic, topic)
        off_score, off_terms = topic_alignment(off_topic, topic)

        self.assertGreater(on_score, off_score)
        self.assertGreater(on_score, 0.0)
        self.assertEqual(off_score, 0.0)
        self.assertTrue(on_terms)
        self.assertFalse(off_terms)

    def test_topic_alignment_idf_deweights_generic_tokens(self):
        topic = "What interaction package sizes are common in commander decklists?"
        on_topic = "Interaction package sizes are usually around ten slots in midrange shells."
        generic = "Commander decklists are common and successful in current data."

        token_idf = build_token_idf(
            [
                topic,
                "Commander decklists are common in current data.",
                "Successful commander decks are common in current metagames.",
                "Common commander decklists include many staple cards.",
                "Interaction package sizes vary by archetype.",
            ],
            min_token_length=4,
        )

        on_topic_score, _ = topic_alignment(on_topic, topic, token_idf=token_idf)
        generic_score, _ = topic_alignment(generic, topic, token_idf=token_idf)

        self.assertGreater(on_topic_score, generic_score)


if __name__ == "__main__":
    unittest.main()
