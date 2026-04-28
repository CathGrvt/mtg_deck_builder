import argparse
import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix


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
    parser.add_argument(
        "--cards",
        default="data/commander_cards.csv",
        help=(
            "Path to card database CSV for oracle-text semantics. "
            "If missing, training falls back to non-semantic clustering."
        ),
    )
    parser.add_argument(
        "--semantic-dim",
        type=int,
        default=64,
        help="Target dimensionality for card text embeddings (default: 64).",
    )
    return parser.parse_args()


def load_corpus(prefix: str):
    npz_path = prefix + ".npz"
    cards_path = prefix + "_cards.json"
    meta_path = prefix + "_meta.json"

    data = np.load(npz_path)
    if {"X_data", "X_indices", "X_indptr", "X_shape"}.issubset(data.files):
        shape = tuple(int(x) for x in data["X_shape"])
        X = csr_matrix(
            (data["X_data"], data["X_indices"], data["X_indptr"]),
            shape=shape,
            dtype=np.float32,
        )
    elif "X" in data.files:
        X = np.asarray(data["X"], dtype=np.float32)
    else:
        raise ValueError(
            "Unsupported corpus format. Expected sparse CSR fields "
            "('X_data', 'X_indices', 'X_indptr', 'X_shape') or dense 'X'."
        )
    deck_colors = np.asarray(data["deck_colors"], dtype=np.float32)

    with open(cards_path, "r", encoding="utf-8") as f:
        cards_data = json.load(f)
    card_vocab: List[str] = cards_data.get("card_vocab", [])

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    color_order = meta.get("color_order", COLOR_ORDER)

    return X, deck_colors, card_vocab, color_order


def load_oracle_text_by_name(cards_csv: str) -> Dict[str, str]:
    """
    Load card oracle text keyed by card name.
    """
    if not cards_csv or not os.path.isfile(cards_csv):
        return {}

    df = pd.read_csv(cards_csv)
    if "name" not in df.columns:
        return {}

    if "oracle_text" not in df.columns:
        df["oracle_text"] = ""

    unique = df.drop_duplicates(subset="name", keep="first")
    oracle_by_name: Dict[str, str] = {}
    for _, row in unique.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        oracle = row.get("oracle_text")
        oracle_by_name[name] = str(oracle).strip() if isinstance(oracle, str) else ""
    return oracle_by_name


def build_card_text_embeddings(
    card_vocab: List[str],
    oracle_by_name: Dict[str, str],
    semantic_dim: int,
) -> Optional[Dict[str, Any]]:
    """
    Build semantic vectors for cards from oracle text using TF-IDF + SVD.
    """
    if not card_vocab:
        return None
    if not oracle_by_name:
        return None

    texts: List[str] = []
    available = 0
    for name in card_vocab:
        oracle = oracle_by_name.get(name, "")
        if oracle:
            available += 1
        texts.append(f"{name}. {oracle}".strip())

    if available == 0:
        return None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=25000,
        min_df=1,
    )
    tfidf = vectorizer.fit_transform(texts)
    if tfidf.shape[1] < 2:
        return None

    target_dim = int(max(2, semantic_dim))
    usable_dim = min(target_dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if usable_dim < 2:
        return None

    svd = TruncatedSVD(n_components=usable_dim, random_state=42)
    dense = svd.fit_transform(tfidf).astype(np.float32)
    dense = normalize(dense, norm="l2")

    return {
        "card_text_embeddings": dense.astype(np.float32),
        "semantic_dim": int(usable_dim),
        "explained_variance_ratio": float(np.sum(svd.explained_variance_ratio_)),
        "cards_with_oracle_text": int(available),
        "total_cards": int(len(card_vocab)),
    }


def build_deck_semantic_vectors(
    X: Union[np.ndarray, csr_matrix],
    card_text_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Aggregate card text embeddings into one semantic vector per deck.
    """
    weighted = X @ card_text_embeddings
    if isinstance(weighted, csr_matrix):
        weighted = weighted.toarray()
    weighted = np.asarray(weighted, dtype=np.float32)

    if isinstance(X, csr_matrix):
        deck_sizes = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    else:
        deck_sizes = np.asarray(X.sum(axis=1), dtype=np.float32).ravel()
    deck_sizes = np.maximum(deck_sizes, 1.0)
    weighted /= deck_sizes[:, None]

    return normalize(weighted, norm="l2").astype(np.float32)


def compute_cluster_semantic_centers(
    deck_semantic_vectors: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Compute semantic centroid per cluster.
    """
    centers = np.zeros((n_clusters, deck_semantic_vectors.shape[1]), dtype=np.float32)
    for cluster_id in range(n_clusters):
        members = deck_semantic_vectors[labels == cluster_id]
        if len(members) == 0:
            continue
        center = members.mean(axis=0)
        norm = float(np.linalg.norm(center))
        if norm > 0.0:
            center = center / norm
        centers[cluster_id] = center.astype(np.float32)
    return centers


def train_kmeans(
    X: Union[np.ndarray, csr_matrix],
    deck_colors: np.ndarray,
    n_clusters: int,
    seed: int,
):
    """
    Train MiniBatchKMeans on deck vectors and compute a simple
    color profile per cluster.
    """
    if X.shape[0] == 0:
        raise ValueError("Corpus has no decks to train on.")

    effective_clusters = max(1, min(n_clusters, X.shape[0]))
    if effective_clusters != n_clusters:
        print(
            f"Requested {n_clusters} clusters but corpus has {X.shape[0]} decks; "
            f"using {effective_clusters} clusters."
        )

    print(f"Training MiniBatchKMeans with {effective_clusters} clusters...")
    model = MiniBatchKMeans(
        n_clusters=effective_clusters,
        random_state=seed,
        batch_size=min(256, max(32, X.shape[0] // 4)),
        n_init="auto",
    )
    model.fit(X)

    labels = model.labels_
    centers = model.cluster_centers_

    n_colors = deck_colors.shape[1]
    cluster_colors = np.zeros((effective_clusters, n_colors), dtype=np.float32)
    cluster_sizes = np.bincount(labels, minlength=effective_clusters).astype(np.int32)

    for k in range(effective_clusters):
        if cluster_sizes[k] > 0:
            cluster_colors[k] = deck_colors[labels == k].mean(axis=0)

    return centers, cluster_colors, cluster_sizes, float(model.inertia_), labels


def main() -> None:
    args = parse_args()

    print(f"Loading deck corpus from prefix '{args.corpus_prefix}'...")
    X, deck_colors, card_vocab, color_order = load_corpus(args.corpus_prefix)

    print(
        f"Corpus has {X.shape[0]} decks and {X.shape[1]} cards in vocabulary."
    )

    centers, cluster_colors, cluster_sizes, inertia, labels = train_kmeans(
        X=X,
        deck_colors=deck_colors,
        n_clusters=args.clusters,
        seed=args.seed,
    )

    oracle_by_name = load_oracle_text_by_name(args.cards)
    semantic_info = build_card_text_embeddings(
        card_vocab=card_vocab,
        oracle_by_name=oracle_by_name,
        semantic_dim=args.semantic_dim,
    )
    card_text_embeddings = None
    cluster_semantic_centers = None
    if semantic_info is not None:
        card_text_embeddings = semantic_info["card_text_embeddings"]
        deck_semantic_vectors = build_deck_semantic_vectors(
            X=X,
            card_text_embeddings=card_text_embeddings,
        )
        cluster_semantic_centers = compute_cluster_semantic_centers(
            deck_semantic_vectors=deck_semantic_vectors,
            labels=labels,
            n_clusters=centers.shape[0],
        )
        print(
            "Built semantic card-text embeddings: "
            f"{semantic_info['cards_with_oracle_text']}/{semantic_info['total_cards']} "
            f"cards, dim={semantic_info['semantic_dim']}, "
            f"explained_var={semantic_info['explained_variance_ratio']:.3f}"
        )
    else:
        print(
            "Semantic card-text embeddings unavailable (missing cards CSV or "
            "insufficient oracle text); continuing with non-semantic model."
        )

    out_prefix = args.output_prefix
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    npz_path = out_prefix + ".npz"
    meta_path = out_prefix + "_meta.json"

    print(f"Saving trained model to {npz_path}...")
    payload = {
        "cluster_centers": centers.astype(np.float32),
        "cluster_colors": cluster_colors.astype(np.float32),
    }
    if card_text_embeddings is not None and cluster_semantic_centers is not None:
        payload["card_text_embeddings"] = card_text_embeddings.astype(np.float32)
        payload["cluster_semantic_centers"] = cluster_semantic_centers.astype(np.float32)
    np.savez_compressed(npz_path, **payload)

    print(f"Saving model metadata to {meta_path}...")
    meta = {
        "card_vocab": card_vocab,
        "color_order": color_order,
        "n_clusters": int(centers.shape[0]),
        "vector_size": int(centers.shape[1]),
        "corpus_prefix": args.corpus_prefix,
        "cluster_sizes": cluster_sizes.tolist(),
        "inertia": inertia,
        "semantic_enabled": bool(card_text_embeddings is not None),
        "semantic_dim": int(card_text_embeddings.shape[1]) if card_text_embeddings is not None else 0,
        "semantic_cards_csv": args.cards,
    }
    if semantic_info is not None:
        meta["semantic_explained_variance_ratio"] = semantic_info["explained_variance_ratio"]
        meta["semantic_cards_with_oracle_text"] = semantic_info["cards_with_oracle_text"]
        meta["semantic_total_cards"] = semantic_info["total_cards"]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Deck generator model training complete.")


if __name__ == "__main__":
    main()
