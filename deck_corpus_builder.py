import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from mtg_io import load_card_database, load_decklists_from_directory


COLOR_ORDER = ["W", "U", "B", "R", "G"]


def build_card_vocab(
    card_db: pd.DataFrame, decklists: Dict[str, List[str]]
) -> Tuple[List[str], Counter]:
    """
    Build a vocabulary of cards actually used in the supplied decklists
    and a frequency counter over that corpus.
    """
    db_names = set(card_db["name"])
    vocab_counter: Counter = Counter()

    for deck in decklists.values():
        for name in deck:
            if name in db_names:
                vocab_counter[name] += 1

    # Sort by decreasing frequency, then by name for determinism
    vocab = sorted(vocab_counter.keys(), key=lambda n: (-vocab_counter[n], n))
    return vocab, vocab_counter


def compute_deck_colors(
    deck: List[str], card_db: pd.DataFrame
) -> np.ndarray:
    """
    Compute a simple color-identity vector for a deck using COLOR_ORDER.
    """
    color_index = {c: i for i, c in enumerate(COLOR_ORDER)}
    vec = np.zeros(len(COLOR_ORDER), dtype=np.float32)

    # Build quick lookup by name
    unique = card_db.drop_duplicates(subset="name")
    by_name = {row["name"]: row for _, row in unique.iterrows()}

    seen_colors = set()
    for name in set(deck):
        card = by_name.get(name)
        if card is None:
            continue
        identity = card.get("color_identity")
        if isinstance(identity, list):
            for c in identity:
                if c in color_index:
                    seen_colors.add(c)

    for c in seen_colors:
        vec[color_index[c]] = 1.0
    return vec


def build_corpus(
    card_db: pd.DataFrame, decklists: Dict[str, List[str]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build a basic deck corpus:
    - X: deck-by-card count matrix
    - deck_sizes: number of cards per deck
    - deck_colors: color identity multi-hot per deck
    - card_vocab: ordered list of card names
    - deck_names: list of deck identifiers
    """
    card_vocab, _ = build_card_vocab(card_db, decklists)
    if not card_vocab:
        raise ValueError("No overlapping cards between decklists and card database.")

    name_to_idx = {name: i for i, name in enumerate(card_vocab)}
    deck_names: List[str] = list(decklists.keys())

    num_decks = len(deck_names)
    vocab_size = len(card_vocab)

    X = np.zeros((num_decks, vocab_size), dtype=np.int16)
    deck_sizes = np.zeros(num_decks, dtype=np.int16)
    deck_colors = np.zeros((num_decks, len(COLOR_ORDER)), dtype=np.float32)

    for i, deck_name in enumerate(deck_names):
        deck = decklists[deck_name]
        counts = Counter(deck)
        deck_sizes[i] = len(deck)

        for card_name, count in counts.items():
            idx = name_to_idx.get(card_name)
            if idx is not None:
                X[i, idx] = int(count)

        deck_colors[i, :] = compute_deck_colors(deck, card_db)

    return X, deck_sizes, deck_colors, card_vocab, deck_names


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

    print(f"Building corpus from {len(decklists)} decks...")
    X, deck_sizes, deck_colors, card_vocab, deck_names = build_corpus(card_db, decklists)

    out_prefix = args.output_prefix
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    npz_path = out_prefix + ".npz"
    print(f"Saving matrix data to {npz_path}...")
    np.savez_compressed(
        npz_path,
        X=X,
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
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Deck corpus build complete.")


if __name__ == "__main__":
    main()
