import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from deck_corpus_builder import build_card_lookup, build_corpus
from train_deck_generator import train_kmeans


class CorpusBuilderTests(unittest.TestCase):
    def test_sparse_corpus_and_quality_stats(self):
        card_db = pd.DataFrame(
            [
                {"name": "Sol Ring", "color_identity": []},
                {"name": "Swords to Plowshares", "color_identity": ["W"]},
                {"name": "Lightning Bolt", "color_identity": ["R"]},
            ]
        )
        decklists = {
            "Deck A": ["Sol Ring", "Swords to Plowshares", "Unknown Card"],
            "Deck B": ["Sol Ring", "Lightning Bolt", "Lightning Bolt"],
        }

        card_lookup = build_card_lookup(card_db)
        X, deck_sizes, deck_colors, card_vocab, deck_names, quality = build_corpus(
            card_lookup=card_lookup,
            decklists=decklists,
            min_card_frequency=2,
        )

        self.assertIsInstance(X, csr_matrix)
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(deck_sizes.tolist(), [3, 3])
        self.assertEqual(deck_names, ["Deck A", "Deck B"])
        self.assertEqual(card_vocab, ["Lightning Bolt", "Sol Ring"])
        self.assertEqual(int(quality["unknown_cards"]), 1)
        self.assertEqual(int(quality["matched_cards_in_db"]), 5)
        self.assertEqual(int(quality["matched_cards_after_vocab"]), 4)
        self.assertAlmostEqual(float(quality["coverage_ratio"]), 4 / 6, places=6)
        self.assertTrue(np.array_equal(deck_colors[0], np.array([1, 0, 0, 0, 0], dtype=np.float32)))
        self.assertTrue(np.array_equal(deck_colors[1], np.array([0, 0, 0, 1, 0], dtype=np.float32)))


class TrainerTests(unittest.TestCase):
    def test_cluster_count_is_capped_by_deck_count(self):
        X = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        deck_colors = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]], dtype=np.float32)

        centers, cluster_colors, cluster_sizes, inertia, labels = train_kmeans(
            X=X,
            deck_colors=deck_colors,
            n_clusters=16,
            seed=42,
        )

        self.assertEqual(centers.shape[0], 2)
        self.assertEqual(cluster_colors.shape, (2, 5))
        self.assertEqual(int(cluster_sizes.sum()), 2)
        self.assertEqual(len(labels), 2)
        self.assertGreaterEqual(inertia, 0.0)


if __name__ == "__main__":
    unittest.main()
