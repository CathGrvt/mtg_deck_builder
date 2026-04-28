import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from mtg_io import load_card_database, load_decklists_from_directory


COLOR_ORDER = ["W", "U", "B", "R", "G"]


def build_card_vocab(
    decklists: Dict[str, List[str]],
    known_cards: set[str],
    min_card_frequency: int = 1,
    max_vocab_size: int = 0,
) -> Tuple[List[str], Counter, Dict[str, Any]]:
    """
    Build a vocabulary of cards used in decklists that also exist in the
    card database, with optional low-frequency pruning.
    """
    vocab_counter: Counter = Counter()
    matched_cards = 0
    unknown_cards = 0

    for deck in decklists.values():
        for name in deck:
            if name in known_cards:
                vocab_counter[name] += 1
                matched_cards += 1
            else:
                unknown_cards += 1

    if min_card_frequency > 1:
        filtered_counts = Counter(
            {name: count for name, count in vocab_counter.items() if count >= min_card_frequency}
        )
    else:
        filtered_counts = vocab_counter

    sorted_vocab = sorted(filtered_counts.keys(), key=lambda n: (-filtered_counts[n], n))
    if max_vocab_size > 0:
        sorted_vocab = sorted_vocab[:max_vocab_size]

    stats = {
        "matched_cards_in_db": int(matched_cards),
        "unknown_cards": int(unknown_cards),
        "unique_known_cards_seen": int(len(vocab_counter)),
        "unique_vocab_after_filters": int(len(sorted_vocab)),
    }
    return sorted_vocab, filtered_counts, stats


def build_card_lookup(card_db: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Build a compact lookup map keyed by card name for fast feature extraction.
    """
    unique = card_db.drop_duplicates(subset="name")
    by_name: Dict[str, Dict[str, Any]] = {}

    for _, row in unique.iterrows():
        color_identity = row.get("color_identity")
        if not isinstance(color_identity, list):
            color_identity = []
        by_name[row["name"]] = {
            "color_identity": [c for c in color_identity if c in COLOR_ORDER],
        }
    return by_name


def compute_deck_colors(
    deck: List[str],
    card_lookup: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    """
    Compute a simple color-identity vector for a deck using COLOR_ORDER.
    """
    color_index = {c: i for i, c in enumerate(COLOR_ORDER)}
    vec = np.zeros(len(COLOR_ORDER), dtype=np.float32)

    seen_colors = set()
    for name in set(deck):
        card = card_lookup.get(name)
        if card is None:
            continue
        for c in card.get("color_identity", []):
            if c in color_index:
                seen_colors.add(c)

    for c in seen_colors:
        vec[color_index[c]] = 1.0
    return vec


def build_corpus(
    card_lookup: Dict[str, Dict[str, Any]],
    decklists: Dict[str, List[str]],
    min_card_frequency: int = 1,
    max_vocab_size: int = 0,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, List[str], List[str], Dict[str, Any]]:
    """
    Build a scalable deck corpus:
    - X: sparse CSR deck-by-card count matrix
    - deck_sizes: number of cards per deck
    - deck_colors: color identity multi-hot per deck
    - card_vocab: ordered list of card names
    - deck_names: list of deck identifiers
    - quality_stats: corpus quality/coverage summary
    """
    card_vocab, _, vocab_stats = build_card_vocab(
        decklists=decklists,
        known_cards=set(card_lookup.keys()),
        min_card_frequency=min_card_frequency,
        max_vocab_size=max_vocab_size,
    )
    if not card_vocab:
        raise ValueError("No overlapping cards between decklists and card database.")

    name_to_idx = {name: i for i, name in enumerate(card_vocab)}
    deck_names: List[str] = list(decklists.keys())

    num_decks = len(deck_names)
    vocab_size = len(card_vocab)

    deck_sizes = np.zeros(num_decks, dtype=np.int32)
    deck_colors = np.zeros((num_decks, len(COLOR_ORDER)), dtype=np.float32)
    row_idx: List[int] = []
    col_idx: List[int] = []
    values: List[int] = []
    matched_cards_after_vocab = 0

    for i, deck_name in enumerate(deck_names):
        deck = decklists[deck_name]
        counts = Counter(deck)
        deck_sizes[i] = len(deck)

        for card_name, count in counts.items():
            idx = name_to_idx.get(card_name)
            if idx is not None:
                row_idx.append(i)
                col_idx.append(idx)
                values.append(int(count))
                matched_cards_after_vocab += int(count)

        deck_colors[i, :] = compute_deck_colors(deck, card_lookup)

    X = csr_matrix(
        (np.asarray(values, dtype=np.float32), (row_idx, col_idx)),
        shape=(num_decks, vocab_size),
        dtype=np.float32,
    )

    total_cards = int(deck_sizes.sum())
    quality_stats = {
        **vocab_stats,
        "matched_cards_after_vocab": int(matched_cards_after_vocab),
        "filtered_out_known_cards": int(vocab_stats["matched_cards_in_db"] - matched_cards_after_vocab),
        "total_cards_in_decklists": total_cards,
        "coverage_ratio": float((matched_cards_after_vocab / total_cards) if total_cards else 0.0),
        "deck_size_min": int(deck_sizes.min()) if len(deck_sizes) else 0,
        "deck_size_max": int(deck_sizes.max()) if len(deck_sizes) else 0,
        "deck_size_mean": float(deck_sizes.mean()) if len(deck_sizes) else 0.0,
        "matrix_nnz": int(X.nnz),
        "matrix_density": float((X.nnz / (num_decks * vocab_size)) if num_decks and vocab_size else 0.0),
    }

    return X, deck_sizes, deck_colors, card_vocab, deck_names, quality_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a simple deck corpus for training deck generator models."
    )
    parser.add_argument(
        "--cards",
        default="data/commander_cards.csv",
        help="Path to card database CSV (default: data/commander_cards.csv).",
    )
    parser.add_argument(
        "--decks",
        default="current_commander_decks",
        help="Directory containing example decklists (default: current_commander_decks).",
    )
    parser.add_argument(
        "--output-prefix",
        default="data/deck_corpus",
        help=(
            "Prefix for output files (default: data/deck_corpus). "
            "Creates '<prefix>.npz', '<prefix>_cards.json', '<prefix>_meta.json'."
        ),
    )
    parser.add_argument(
        "--min-card-frequency",
        type=int,
        default=1,
        help="Minimum number of occurrences required for a card to remain in vocab (default: 1).",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=0,
        help="Optional cap on vocabulary size by corpus frequency (0 = unlimited).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading card database from {args.cards}...")
    card_db = load_card_database(args.cards)

    print(f"Loading decklists from {args.decks}...")
    decklists = load_decklists_from_directory(
        args.decks,
        include_command_zone=True,
    )
    if not decklists:
        print("Error: no decklists found; aborting corpus build.")
        return

    card_lookup = build_card_lookup(card_db)

    print(f"Building corpus from {len(decklists)} decks...")
    X, deck_sizes, deck_colors, card_vocab, deck_names, quality_stats = build_corpus(
        card_lookup=card_lookup,
        decklists=decklists,
        min_card_frequency=max(1, args.min_card_frequency),
        max_vocab_size=max(0, args.max_vocab_size),
    )

    out_prefix = args.output_prefix
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    npz_path = out_prefix + ".npz"
    print(f"Saving matrix data to {npz_path}...")
    np.savez_compressed(
        npz_path,
        X_data=X.data.astype(np.float32),
        X_indices=X.indices.astype(np.int32),
        X_indptr=X.indptr.astype(np.int32),
        X_shape=np.asarray(X.shape, dtype=np.int32),
        deck_sizes=deck_sizes,
        deck_colors=deck_colors,
    )

    cards_path = out_prefix + "_cards.json"
    meta_path = out_prefix + "_meta.json"

    print(f"Saving card vocabulary to {cards_path}...")
    with open(cards_path, "w", encoding="utf-8") as f:
        json.dump({"card_vocab": card_vocab}, f, indent=2)

    print(f"Saving corpus metadata to {meta_path}...")
    meta = {
        "deck_names": deck_names,
        "color_order": COLOR_ORDER,
        "cards_source": args.cards,
        "decks_source": args.decks,
        "matrix_format": "csr",
        "min_card_frequency": int(max(1, args.min_card_frequency)),
        "max_vocab_size": int(max(0, args.max_vocab_size)),
        "quality": quality_stats,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Deck corpus build complete.")


if __name__ == "__main__":
    main()
