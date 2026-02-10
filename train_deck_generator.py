import argparse
import json
import os
from typing import List

import numpy as np
from sklearn.cluster import MiniBatchKMeans


COLOR_ORDER = ["W", "U", "B", "R", "G"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight unsupervised deck generator model "
        "using K-means clustering over deck vectors."
    )
    parser.add_argument(
        "--corpus-prefix",
        default="data/deck_corpus",
        help=(
            "Prefix of the deck corpus files (default: data/deck_corpus). "
            "Expects '<prefix>.npz', '<prefix>_cards.json', '<prefix>_meta.json'."
        ),
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=16,
        help="Number of clusters for K-means (default: 16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for clustering (default: 42).",
    )
    parser.add_argument(
        "--output-prefix",
        default="models/deck_kmeans",
        help=(
            "Prefix for the trained model (default: models/deck_kmeans). "
            "Creates '<prefix>.npz' and '<prefix>_meta.json'."
        ),
    )
    return parser.parse_args()


def load_corpus(prefix: str):
    npz_path = prefix + ".npz"
    cards_path = prefix + "_cards.json"
    meta_path = prefix + "_meta.json"

    data = np.load(npz_path)
    X = data["X"]
    deck_colors = data["deck_colors"]

    with open(cards_path, "r", encoding="utf-8") as f:
        cards_data = json.load(f)
    card_vocab: List[str] = cards_data.get("card_vocab", [])

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    color_order = meta.get("color_order", COLOR_ORDER)

    return X, deck_colors, card_vocab, color_order


def train_kmeans(
    X: np.ndarray,
    deck_colors: np.ndarray,
    n_clusters: int,
    seed: int,
):
    """
    Train MiniBatchKMeans on deck vectors and compute a simple
    color profile per cluster.
    """
    print(f"Training MiniBatchKMeans with {n_clusters} clusters...")
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(256, max(32, X.shape[0] // 4)),
        n_init="auto",
    )
    model.fit(X)

    labels = model.labels_
    centers = model.cluster_centers_

    n_colors = deck_colors.shape[1]
    cluster_colors = np.zeros((n_clusters, n_colors), dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=np.int32)

    for i, label in enumerate(labels):
        cluster_colors[label] += deck_colors[i]
        counts[label] += 1

    for k in range(n_clusters):
        if counts[k] > 0:
            cluster_colors[k] /= counts[k]

    return centers, cluster_colors


def main() -> None:
    args = parse_args()

    print(f"Loading deck corpus from prefix '{args.corpus_prefix}'...")
    X, deck_colors, card_vocab, color_order = load_corpus(args.corpus_prefix)

    print(
        f"Corpus has {X.shape[0]} decks and {X.shape[1]} cards in vocabulary."
    )

    centers, cluster_colors = train_kmeans(
        X=X,
        deck_colors=deck_colors,
        n_clusters=args.clusters,
        seed=args.seed,
    )

    out_prefix = args.output_prefix
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    npz_path = out_prefix + ".npz"
    meta_path = out_prefix + "_meta.json"

    print(f"Saving trained model to {npz_path}...")
    np.savez_compressed(
        npz_path,
        cluster_centers=centers.astype(np.float32),
        cluster_colors=cluster_colors.astype(np.float32),
    )

    print(f"Saving model metadata to {meta_path}...")
    meta = {
        "card_vocab": card_vocab,
        "color_order": color_order,
        "n_clusters": int(centers.shape[0]),
        "vector_size": int(centers.shape[1]),
        "corpus_prefix": args.corpus_prefix,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Deck generator model training complete.")


if __name__ == "__main__":
    main()

