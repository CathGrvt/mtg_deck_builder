import unittest
from collections import Counter

import numpy as np

from ai_deck_generator import DeckSpec, build_weights_with_clusters


class SemanticWeightingTests(unittest.TestCase):
    def test_semantic_alignment_boosts_weight(self):
        names = ["Fireball", "Counterspell"]
        training_counts = Counter({"Fireball": 10, "Counterspell": 10})
        cluster_model = {
            "cluster_centers": np.array([[1.0, 1.0]], dtype=np.float32),
            "cluster_colors": np.array([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
            "card_vocab": ["Fireball", "Counterspell"],
            "color_order": ["W", "U", "B", "R", "G"],
            "card_text_embeddings": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "cluster_semantic_centers": np.array([[1.0, 0.0]], dtype=np.float32),
            "semantic_enabled": True,
        }
        spec = DeckSpec(format="commander", colors=["R"])

        weights = build_weights_with_clusters(
            names=names,
            training_counts=training_counts,
            cluster_model=cluster_model,
            spec=spec,
            cluster_strength=0.0,
            semantic_strength=2.0,
        )

        self.assertGreater(weights[0], weights[1])


if __name__ == "__main__":
    unittest.main()
