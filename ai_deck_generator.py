import argparse
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from deck_analysis import COMMANDER_DUPLICATE_EXCEPTIONS
from mtg_io import load_card_database, load_decklists_from_directory, safe_parse_list


COLOR_ORDER = ["W", "U", "B", "R", "G"]


@dataclass
class DeckSpec:
    """
    High-level specification for a deck to generate.
    This is intentionally simple and model-agnostic so it can be
    reused by more advanced generators later.
    """

    format: str = "commander"
    colors: List[str] = field(default_factory=list)
    archetype: Optional[str] = None
    target_size: Optional[int] = None
    land_ratio: Optional[float] = None
    include_cards: List[str] = field(default_factory=list)
    exclude_cards: List[str] = field(default_factory=list)

    def effective_size(self) -> int:
        """Return the target deck size, falling back to format defaults."""
        if self.target_size is not None and self.target_size > 0:
            return self.target_size
        fmt = self.format.lower()
        if fmt == "commander":
            return 100
        # Default constructed size (Standard / Pioneer / etc.)
        return 60

    def effective_land_ratio(self) -> float:
        """Return the target land ratio, with format-aware defaults."""
        if self.land_ratio is not None and 0.0 < self.land_ratio < 1.0:
            return self.land_ratio
        fmt = self.format.lower()
        if fmt == "commander":
            return 0.42
        return 0.4


@dataclass
class LLMRerankConfig:
    """
    Optional LLM reranking settings.
    """

    top_k: int = 0
    strength: float = 1.0
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    timeout_sec: int = 45


def parse_colors(raw: str) -> List[str]:
    """
    Parse a color string like "WR", "W,R", or "W U" into a sorted list.
    """
    if not raw:
        return []
    cleaned = raw.replace(",", "").replace(" ", "").upper()
    seen = []
    for ch in cleaned:
        if ch in COLOR_ORDER and ch not in seen:
            seen.append(ch)
    return seen


def build_name_index(card_db: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Build a simple name → row mapping, dropping duplicate names arbitrarily.
    """
    if "name" not in card_db.columns:
        raise ValueError("Card database must contain a 'name' column")
    unique = card_db.drop_duplicates(subset="name")
    return {row["name"]: row for _, row in unique.iterrows()}


def card_matches_colors(card: pd.Series, colors: Sequence[str]) -> bool:
    """
    Check if a card's color identity is compatible with the requested colors.
    Colorless cards are always allowed. If no colors are specified, allow all.
    """
    if not colors:
        return True

    requested = set(colors)
    identity = card.get("color_identity")
    if isinstance(identity, list):
        identity_set = set(identity)
    elif isinstance(identity, str) and identity:
        # Fallback – load_deck_database normally converts to list already
        parsed = safe_parse_list(identity)
        identity_set = set(parsed) if isinstance(parsed, list) else set()
    else:
        identity_set = set()

    # Colorless or fully within requested colors
    return not identity_set or identity_set.issubset(requested)


def is_basic_land(card: pd.Series) -> bool:
    type_line = str(card.get("type_line") or "")
    return "Basic" in type_line and "Land" in type_line


def max_allowed_copies(card_name: str, card: pd.Series, spec: DeckSpec) -> int:
    """
    Approximate deckbuilding constraints.
    - Commander: singleton, except basic lands and known exceptions.
    - Non-Commander: up to 4 copies, basic lands unlimited.
    """
    if is_basic_land(card):
        return 99

    fmt = spec.format.lower()
    if fmt == "commander":
        if card_name in COMMANDER_DUPLICATE_EXCEPTIONS:
            return 99
        return 1

    # Default constructed formats
    return 4


def build_training_counts(decklists: Dict[str, List[str]]) -> Counter:
    """
    Build a simple frequency counter from example decklists.
    """
    counts: Counter = Counter()
    for deck in decklists.values():
        counts.update(deck)
    return counts


def load_cluster_model(
    model_prefix: str,
) -> Optional[Dict[str, Any]]:
    """
    Load a lightweight unsupervised model trained by train_deck_generator.py.

    Expects '<prefix>.npz' and '<prefix>_meta.json'.
    """
    npz_path = model_prefix + ".npz"
    meta_path = model_prefix + "_meta.json"

    if not (os.path.isfile(npz_path) and os.path.isfile(meta_path)):
        print(
            f"Warning: cluster model files '{npz_path}' / "
            f"'{meta_path}' not found; ignoring --cluster-model."
        )
        return None

    try:
        data = np.load(npz_path)
        centers = data["cluster_centers"]
        cluster_colors = data["cluster_colors"]
        card_text_embeddings = data["card_text_embeddings"] if "card_text_embeddings" in data.files else None
        cluster_semantic_centers = (
            data["cluster_semantic_centers"] if "cluster_semantic_centers" in data.files else None
        )
    except Exception as e:
        print(f"Warning: failed to load cluster centers from {npz_path}: {e}")
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        card_vocab = meta.get("card_vocab", [])
        color_order = meta.get("color_order", COLOR_ORDER)
    except Exception as e:
        print(f"Warning: failed to load cluster metadata from {meta_path}: {e}")
        return None

    if not card_vocab or centers.shape[1] != len(card_vocab):
        print(
            "Warning: cluster model metadata is inconsistent with "
            "cluster centers; ignoring model."
        )
        return None

    semantic_enabled = (
        card_text_embeddings is not None
        and cluster_semantic_centers is not None
        and card_text_embeddings.shape[0] == len(card_vocab)
        and cluster_semantic_centers.shape[0] == centers.shape[0]
    )
    if not semantic_enabled:
        card_text_embeddings = None
        cluster_semantic_centers = None

    return {
        "cluster_centers": centers,
        "cluster_colors": cluster_colors,
        "card_vocab": card_vocab,
        "color_order": color_order,
        "card_text_embeddings": card_text_embeddings,
        "cluster_semantic_centers": cluster_semantic_centers,
        "semantic_enabled": semantic_enabled,
    }


def select_best_cluster_for_colors(
    cluster_colors: np.ndarray,
    color_order: Sequence[str],
    requested_colors: Sequence[str],
) -> int:
    """
    Select the cluster whose color profile best matches the requested colors.
    """
    if cluster_colors.size == 0:
        return 0

    if not requested_colors:
        # No preference: choose the most "colorful" cluster (largest norm)
        norms = np.linalg.norm(cluster_colors, axis=1)
        return int(norms.argmax())

    color_index = {c: i for i, c in enumerate(color_order)}
    req_vec = np.zeros(cluster_colors.shape[1], dtype=np.float32)
    for c in requested_colors:
        idx = color_index.get(c)
        if idx is not None:
            req_vec[idx] = 1.0

    scores = cluster_colors @ req_vec
    return int(scores.argmax())


def build_weights_with_clusters(
    names: Sequence[str],
    training_counts: Counter,
    cluster_model: Dict[str, Any],
    spec: "DeckSpec",
    cluster_strength: float,
    semantic_strength: float,
) -> List[float]:
    """
    Build sampling weights that combine meta frequency with a trained
    cluster model. Cards that are more typical of the selected cluster
    are boosted relative to the baseline frequency.
    """
    centers: np.ndarray = cluster_model["cluster_centers"]
    cluster_colors: np.ndarray = cluster_model["cluster_colors"]
    card_vocab: List[str] = cluster_model["card_vocab"]
    color_order: Sequence[str] = cluster_model["color_order"]
    card_text_embeddings: Optional[np.ndarray] = cluster_model.get("card_text_embeddings")
    cluster_semantic_centers: Optional[np.ndarray] = cluster_model.get("cluster_semantic_centers")

    if centers.ndim != 2 or centers.shape[0] == 0:
        return [float(training_counts.get(n, 0) + 1.0) for n in names]

    cluster_id = select_best_cluster_for_colors(
        cluster_colors=cluster_colors,
        color_order=color_order,
        requested_colors=spec.colors,
    )

    cluster_center = centers[cluster_id]
    max_val = float(cluster_center.max()) if cluster_center.size > 0 else 0.0
    if max_val <= 0.0:
        # Degenerate cluster; fall back to pure frequency
        return [float(training_counts.get(n, 0) + 1.0) for n in names]

    vocab_index = {name: i for i, name in enumerate(card_vocab)}

    weights: List[float] = []
    for name in names:
        base = float(training_counts.get(name, 0) + 1.0)
        idx = vocab_index.get(name)
        if idx is None:
            weights.append(base)
            continue
        cluster_raw = float(cluster_center[idx])
        if cluster_raw <= 0.0:
            weights.append(base)
            continue
        cluster_norm = cluster_raw / max_val

        semantic_boost = 0.0
        if (
            semantic_strength > 0.0
            and card_text_embeddings is not None
            and cluster_semantic_centers is not None
            and idx < card_text_embeddings.shape[0]
            and cluster_id < cluster_semantic_centers.shape[0]
        ):
            card_vec = card_text_embeddings[idx]
            cluster_vec = cluster_semantic_centers[cluster_id]
            denom = float(np.linalg.norm(card_vec) * np.linalg.norm(cluster_vec))
            if denom > 0.0:
                cosine_sim = float(np.dot(card_vec, cluster_vec) / denom)
                semantic_boost = max(0.0, cosine_sim)

        cluster_factor = 1.0 + cluster_strength * cluster_norm
        semantic_factor = 1.0 + semantic_strength * semantic_boost
        weights.append(base * cluster_factor * semantic_factor)
    return weights


def weighted_choice(
    names: Sequence[str],
    weights: Sequence[float],
    rng: random.Random,
) -> Optional[str]:
    """
    Pick a single element using the given weights. Returns None if empty.
    """
    if not names:
        return None
    # random.choices is available from Python 3.6+
    return rng.choices(names, weights=weights, k=1)[0]


def _extract_json_scores(text: str) -> Dict[str, float]:
    """
    Parse LLM output and extract a name -> score mapping.
    Expected format:
      {"scores": {"Card Name": 87, ...}}
    """
    if not text:
        return {}

    payload: Any = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    if not isinstance(payload, dict):
        return {}

    raw_scores = payload.get("scores", payload)
    if not isinstance(raw_scores, dict):
        return {}

    scores: Dict[str, float] = {}
    for name, value in raw_scores.items():
        try:
            scores[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return scores


def score_cards_with_llm(
    card_briefs: Sequence[Dict[str, Any]],
    spec: DeckSpec,
    llm_config: LLMRerankConfig,
    category: str,
) -> Dict[str, float]:
    """
    Score a candidate subset with an LLM.
    Returns dict: card_name -> 0..100 score (higher is better fit).
    """
    if not card_briefs:
        return {}

    api_key = os.getenv(llm_config.api_key_env, "")
    if not api_key:
        raise ValueError(
            f"Environment variable '{llm_config.api_key_env}' is not set; "
            "cannot use LLM reranking."
        )

    requested_colors = "".join(spec.colors) if spec.colors else "any"
    archetype = spec.archetype or "unspecified"
    system_prompt = (
        "You are an expert MTG deck construction assistant. "
        "Score each candidate card for how well it fits the requested deck plan. "
        "Output strict JSON only."
    )
    user_payload = {
        "task": "score_cards_for_deck",
        "format": spec.format,
        "colors": requested_colors,
        "archetype_hint": archetype,
        "target_size": spec.effective_size(),
        "category": category,
        "candidates": list(card_briefs),
        "output_schema": {
            "scores": {
                "<exact card name>": "number 0-100"
            }
        },
        "instructions": [
            "Use exact candidate card names as keys.",
            "Score all candidates from 0 to 100.",
            "Prefer cards that synergize with the requested strategy and colors.",
            "Return JSON only with key 'scores'.",
        ],
    }
    user_prompt = json.dumps(user_payload, ensure_ascii=False)

    url = llm_config.base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": llm_config.model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        headers=headers,
        json=body,
        timeout=llm_config.timeout_sec,
    )
    response.raise_for_status()

    data = response.json()
    content = ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            content = str(message.get("content", "")).strip()

    return _extract_json_scores(content)


def rerank_weights_with_llm(
    names: Sequence[str],
    base_weights: Sequence[float],
    name_index: Dict[str, pd.Series],
    spec: DeckSpec,
    llm_config: Optional[LLMRerankConfig],
    category: str,
    scorer: Optional[Callable[[Sequence[Dict[str, Any]], DeckSpec, LLMRerankConfig, str], Dict[str, float]]] = None,
) -> List[float]:
    """
    Rerank top-K candidates with an LLM and apply multiplicative boosts.
    """
    if (
        llm_config is None
        or llm_config.top_k <= 0
        or llm_config.strength <= 0.0
        or not names
    ):
        return [float(w) for w in base_weights]

    top_k = min(int(llm_config.top_k), len(names))
    if top_k <= 1:
        return [float(w) for w in base_weights]

    ranked_indices = sorted(
        range(len(names)),
        key=lambda i: float(base_weights[i]),
        reverse=True,
    )[:top_k]

    card_briefs: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        name = names[idx]
        row = name_index.get(name)
        if row is None:
            continue
        oracle_text = str(row.get("oracle_text") or "")
        if len(oracle_text) > 420:
            oracle_text = oracle_text[:420] + "..."
        card_briefs.append(
            {
                "name": name,
                "type_line": str(row.get("type_line") or ""),
                "mana_cost": str(row.get("mana_cost") or ""),
                "cmc": float(row.get("cmc") or 0.0),
                "oracle_text": oracle_text,
            }
        )

    if not card_briefs:
        return [float(w) for w in base_weights]

    scorer_fn = scorer or score_cards_with_llm
    try:
        llm_scores = scorer_fn(card_briefs, spec, llm_config, category)
    except Exception as e:
        print(f"Warning: LLM rerank failed for {category}: {e}")
        return [float(w) for w in base_weights]

    if not llm_scores:
        print(f"Warning: LLM rerank returned no usable scores for {category}.")
        return [float(w) for w in base_weights]

    reranked = [float(w) for w in base_weights]
    for idx in ranked_indices:
        name = names[idx]
        raw = llm_scores.get(name)
        if raw is None:
            continue
        norm = max(0.0, min(100.0, float(raw))) / 100.0
        reranked[idx] *= (1.0 + llm_config.strength * norm)

    return reranked


def generate_deck_from_meta(
    card_db: pd.DataFrame,
    decklists: Dict[str, List[str]],
    spec: DeckSpec,
    cluster_model: Optional[Dict[str, Any]] = None,
    cluster_strength: float = 2.0,
    semantic_strength: float = 1.0,
    llm_rerank_config: Optional[LLMRerankConfig] = None,
    llm_scorer: Optional[
        Callable[[Sequence[Dict[str, Any]], DeckSpec, LLMRerankConfig, str], Dict[str, float]]
    ] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Baseline meta-driven generator:
    - Uses card database + optional training decklists.
    - Favors cards that appear more often in the meta.
    - Respects colors and very rough deckbuilding rules.
    """
    rng = random.Random(seed)
    np.random.seed(seed if seed is not None else None)

    name_index = build_name_index(card_db)
    training_counts = build_training_counts(decklists) if decklists else Counter()

    # Filter allowed cards by color and exclusions
    allowed_names: List[str] = []
    for name, row in name_index.items():
        if name in spec.exclude_cards:
            continue
        if card_matches_colors(row, spec.colors):
            allowed_names.append(name)

    if not allowed_names:
        raise ValueError("No cards match the requested color identity and filters.")

    # Split into lands and nonlands
    land_names: List[str] = []
    spell_names: List[str] = []
    for name in allowed_names:
        row = name_index[name]
        if bool(row.get("is_land")):
            land_names.append(name)
        else:
            spell_names.append(name)

    if not land_names or not spell_names:
        raise ValueError("Insufficient variety in allowed card pool to build a deck.")

    # Build sampling weights based on meta frequency and optional cluster model
    def build_weights(names: Sequence[str]) -> List[float]:
        if cluster_model is not None:
            return build_weights_with_clusters(
                names=names,
                training_counts=training_counts,
                cluster_model=cluster_model,
                spec=spec,
                cluster_strength=cluster_strength,
                semantic_strength=semantic_strength,
            )
        return [float(training_counts.get(n, 0) + 1.0) for n in names]

    land_weights = build_weights(land_names)
    spell_weights = build_weights(spell_names)
    if llm_rerank_config is not None:
        land_weights = rerank_weights_with_llm(
            names=land_names,
            base_weights=land_weights,
            name_index=name_index,
            spec=spec,
            llm_config=llm_rerank_config,
            category="lands",
            scorer=llm_scorer,
        )
        spell_weights = rerank_weights_with_llm(
            names=spell_names,
            base_weights=spell_weights,
            name_index=name_index,
            spec=spec,
            llm_config=llm_rerank_config,
            category="nonlands",
            scorer=llm_scorer,
        )

    target_size = spec.effective_size()
    target_land = int(round(target_size * spec.effective_land_ratio()))

    deck: List[str] = []
    counts: Counter = Counter()

    # First, honor include_cards as much as possible
    for name in spec.include_cards:
        row = name_index.get(name)
        if row is None:
            print(f"Warning: requested include card '{name}' not found in card database.")
            continue
        if name in spec.exclude_cards:
            print(f"Warning: requested include card '{name}' is also excluded; skipping.")
            continue
        limit = max_allowed_copies(name, row, spec)
        if limit <= 0:
            continue
        if counts[name] < limit:
            deck.append(name)
            counts[name] += 1

    # Count how many of the already included cards are lands
    current_lands = 0
    for name in deck:
        row = name_index.get(name)
        if row is not None and bool(row.get("is_land")):
            current_lands += 1

    remaining_slots = max(0, target_size - len(deck))
    remaining_land_target = max(0, target_land - current_lands)

    # Helper to sample cards of a given category
    def fill_category(
        candidate_names: Sequence[str],
        candidate_weights: Sequence[float],
        remaining_to_add: int,
    ) -> None:
        if remaining_to_add <= 0 or not candidate_names:
            return

        max_attempts = max(remaining_to_add * 10, 50)
        attempts = 0
        while len(deck) < target_size and remaining_to_add > 0 and attempts < max_attempts:
            name = weighted_choice(candidate_names, candidate_weights, rng)
            if name is None:
                break
            row = name_index[name]
            limit = max_allowed_copies(name, row, spec)
            if counts[name] < limit:
                deck.append(name)
                counts[name] += 1
                remaining_to_add -= 1
            attempts += 1

    # Fill lands toward target
    fill_category(land_names, land_weights, remaining_land_target)

    # Fill remaining slots with nonland spells
    remaining_slots = max(0, target_size - len(deck))
    fill_category(spell_names, spell_weights, remaining_slots)

    if len(deck) < target_size:
        print(
            f"Warning: generated deck has {len(deck)} cards "
            f"(target was {target_size}). Card pool or constraints may be too strict."
        )

    return deck


def summarize_deck(deck: List[str], card_db: pd.DataFrame) -> str:
    """
    Produce a human-readable summary of the generated deck:
    - card counts
    - land count vs. nonlands
    """
    if not deck:
        return "Empty deck."

    name_index = build_name_index(card_db)
    counts = Counter(deck)

    land_count = 0
    nonland_count = 0
    for name, count in counts.items():
        row = name_index.get(name)
        if row is not None and bool(row.get("is_land")):
            land_count += count
        else:
            nonland_count += count

    lines = []
    lines.append(f"Total cards: {sum(counts.values())}")
    lines.append(f"  Lands: {land_count}")
    lines.append(f"  Nonlands: {nonland_count}")
    return "\n".join(lines)


def format_decklist(deck: List[str], card_db: pd.DataFrame) -> str:
    """
    Format a decklist as simple '<count> <card name>' lines.
    Lands are listed last for readability.
    """
    if not deck:
        return ""

    name_index = build_name_index(card_db)
    counts = Counter(deck)

    def sort_key(name: str):
        row = name_index.get(name)
        is_land = bool(row.get("is_land")) if row is not None else False
        cmc = float(row.get("cmc")) if row is not None else 0.0
        return (1 if is_land else 0, cmc, name)

    ordered_names = sorted(counts.keys(), key=sort_key)

    lines: List[str] = []
    for name in ordered_names:
        lines.append(f"{counts[name]} {name}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline AI deck generator using meta statistics."
    )
    parser.add_argument(
        "--cards",
        default="data/commander_cards.csv",
        help="Path to card database CSV (default: data/commander_cards.csv)",
    )
    parser.add_argument(
        "--training-decks",
        default="current_commander_decks",
        help=(
            "Directory containing example decklists used as meta training data "
            "(default: current_commander_decks)"
        ),
    )
    parser.add_argument(
        "--format",
        dest="deck_format",
        default="commander",
        help="Format label for the deck (e.g., commander, standard).",
    )
    parser.add_argument(
        "--colors",
        default="",
        help="Desired color identity (e.g., 'WR', 'W,U', or 'W U').",
    )
    parser.add_argument(
        "--archetype",
        default=None,
        help="Optional archetype hint (aggro, midrange, control, tempo, combo).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Target deck size. Defaults to 100 for commander, 60 otherwise.",
    )
    parser.add_argument(
        "--land-ratio",
        type=float,
        default=None,
        help="Target land ratio between 0 and 1 (default: ~0.42 commander, 0.4 other).",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        help="Card names that must be included (quoted if they contain spaces).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Card names that must NOT be included.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--cluster-model",
        default=None,
        help=(
            "Optional prefix of a trained cluster model "
            "(from train_deck_generator.py). Expects '<prefix>.npz' "
            "and '<prefix>_meta.json'."
        ),
    )
    parser.add_argument(
        "--cluster-strength",
        type=float,
        default=2.0,
        help=(
            "How strongly to favor cards that are typical of the selected "
            "cluster (default: 2.0). Ignored if no cluster model is used."
        ),
    )
    parser.add_argument(
        "--semantic-strength",
        type=float,
        default=1.0,
        help=(
            "How strongly to favor cards semantically aligned with the selected "
            "cluster's oracle-text profile (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--llm-rerank-top-k",
        type=int,
        default=0,
        help=(
            "Optional LLM rerank on top-K candidates per category (lands/nonlands). "
            "0 disables (default: 0)."
        ),
    )
    parser.add_argument(
        "--llm-strength",
        type=float,
        default=1.0,
        help="Strength of LLM rerank boost when enabled (default: 1.0).",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="Chat model to use for LLM reranking (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable holding API key for LLM reranking (default: OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--llm-base-url",
        default="https://api.openai.com/v1",
        help="Base URL for chat-completions API (default: https://api.openai.com/v1).",
    )
    parser.add_argument(
        "--llm-timeout-sec",
        type=int,
        default=45,
        help="HTTP timeout in seconds for LLM reranking requests (default: 45).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the generated decklist as a .txt file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading card database from {args.cards}...")
    card_db = load_card_database(args.cards)

    decklists: Dict[str, List[str]] = {}
    if args.training_decks and os.path.isdir(args.training_decks):
        print(f"Loading training decks from {args.training_decks}...")
        try:
            decklists = load_decklists_from_directory(
                args.training_decks,
                include_command_zone=True,
            )
            if not decklists:
                print(
                    "Warning: no training decklists loaded; "
                    "generation will fall back to card database only."
                )
        except Exception as e:
            print(f"Warning: failed to load training decklists: {e}")
    else:
        print(
            "Warning: training deck directory not found; "
            "generation will rely on card database only."
        )

    cluster_model: Optional[Dict[str, Any]] = None
    if args.cluster_model:
        print(f"Loading cluster model from prefix '{args.cluster_model}'...")
        cluster_model = load_cluster_model(args.cluster_model)

    colors = parse_colors(args.colors)
    spec = DeckSpec(
        format=args.deck_format,
        colors=colors,
        archetype=args.archetype,
        target_size=args.size,
        land_ratio=args.land_ratio,
        include_cards=args.include,
        exclude_cards=args.exclude,
    )

    print("=== Deck Specification ===")
    print(f"Format: {spec.format}")
    print(f"Colors: {''.join(spec.colors) if spec.colors else 'Any'}")
    if spec.archetype:
        print(f"Archetype hint: {spec.archetype}")
    print(f"Target size: {spec.effective_size()}")
    print(f"Target land ratio: {spec.effective_land_ratio():.2f}")
    if spec.include_cards:
        print(f"Must include: {', '.join(spec.include_cards)}")
    if spec.exclude_cards:
        print(f"Must exclude: {', '.join(spec.exclude_cards)}")

    if args.cluster_model and cluster_model is not None:
        print(f"Using cluster model with strength {args.cluster_strength:.2f}")
        if cluster_model.get("semantic_enabled"):
            print(f"Using semantic textbox weighting with strength {args.semantic_strength:.2f}")

    llm_rerank_config: Optional[LLMRerankConfig] = None
    if args.llm_rerank_top_k > 0 and args.llm_strength > 0.0:
        if os.getenv(args.llm_api_key_env):
            llm_rerank_config = LLMRerankConfig(
                top_k=max(0, int(args.llm_rerank_top_k)),
                strength=max(0.0, float(args.llm_strength)),
                model=args.llm_model,
                api_key_env=args.llm_api_key_env,
                base_url=args.llm_base_url,
                timeout_sec=max(1, int(args.llm_timeout_sec)),
            )
            print(
                "Using LLM reranking: "
                f"top_k={llm_rerank_config.top_k}, "
                f"strength={llm_rerank_config.strength:.2f}, "
                f"model={llm_rerank_config.model}"
            )
        else:
            print(
                f"Warning: LLM reranking requested but env var "
                f"'{args.llm_api_key_env}' is not set; skipping."
            )

    print("\nGenerating deck...")
    deck = generate_deck_from_meta(
        card_db=card_db,
        decklists=decklists,
        spec=spec,
        cluster_model=cluster_model,
        cluster_strength=args.cluster_strength,
        semantic_strength=args.semantic_strength,
        llm_rerank_config=llm_rerank_config,
        seed=args.seed,
    )

    print("\n=== Generated Deck Summary ===")
    print(summarize_deck(deck, card_db))

    print("\n=== Generated Decklist ===")
    deck_text = format_decklist(deck, card_db)
    print(deck_text)

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(deck_text + "\n")
        print(f"\nDecklist written to {args.output}")


if __name__ == "__main__":
    main()
