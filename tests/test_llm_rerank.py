import unittest

import pandas as pd

from ai_deck_generator import DeckSpec, LLMRerankConfig, rerank_weights_with_llm


def _fake_llm_scorer(card_briefs, spec, llm_config, category):
    del spec, llm_config, category
    scores = {}
    for brief in card_briefs:
        name = brief["name"]
        scores[name] = 100.0 if name == "Lightning Bolt" else 10.0
    return scores


class LLMRerankTests(unittest.TestCase):
    def test_rerank_boosts_high_llm_score(self):
        names = ["Lightning Bolt", "Shock", "Mountain"]
        base_weights = [1.0, 1.0, 1.0]
        name_index = {
            "Lightning Bolt": pd.Series({"type_line": "Instant", "oracle_text": "Deal 3 damage."}),
            "Shock": pd.Series({"type_line": "Instant", "oracle_text": "Deal 2 damage."}),
            "Mountain": pd.Series({"type_line": "Basic Land — Mountain", "oracle_text": ""}),
        }
        spec = DeckSpec(format="commander", colors=["R"])
        cfg = LLMRerankConfig(top_k=2, strength=2.0)

        reranked = rerank_weights_with_llm(
            names=names,
            base_weights=base_weights,
            name_index=name_index,
            spec=spec,
            llm_config=cfg,
            category="nonlands",
            scorer=_fake_llm_scorer,
        )

        self.assertGreater(reranked[0], reranked[1])
        self.assertEqual(reranked[2], 1.0)  # outside top_k, unchanged

    def test_rerank_fallback_on_scorer_error(self):
        def raising_scorer(card_briefs, spec, llm_config, category):
            del card_briefs, spec, llm_config, category
            raise RuntimeError("boom")

        names = ["Sol Ring", "Arcane Signet"]
        base_weights = [3.0, 2.0]
        name_index = {
            "Sol Ring": pd.Series({"type_line": "Artifact", "oracle_text": "{T}: Add {C}{C}."}),
            "Arcane Signet": pd.Series({"type_line": "Artifact", "oracle_text": "{T}: Add one mana."}),
        }
        spec = DeckSpec(format="commander")
        cfg = LLMRerankConfig(top_k=2, strength=1.0)

        reranked = rerank_weights_with_llm(
            names=names,
            base_weights=base_weights,
            name_index=name_index,
            spec=spec,
            llm_config=cfg,
            category="nonlands",
            scorer=raising_scorer,
        )

        self.assertEqual(reranked, base_weights)


if __name__ == "__main__":
    unittest.main()
