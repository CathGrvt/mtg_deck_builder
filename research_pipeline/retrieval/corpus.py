from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from mtg_io import load_decklists_from_directory
from research_pipeline.models import DocumentChunk
from research_pipeline.set_aliases import set_name_for_code


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned or "unknown"


def _chunk_words(text: str, chunk_size: int = 180, overlap: int = 35) -> List[str]:
    words = text.split()
    if not words:
        return []

    if chunk_size <= 0:
        chunk_size = 180
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        segment = words[start : start + chunk_size]
        if not segment:
            break
        chunks.append(" ".join(segment))
        if start + chunk_size >= len(words):
            break
    return chunks


def _deck_text(deck_name: str, cards: Sequence[str]) -> str:
    counts = Counter(cards)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top_lines = [f"{count}x {name}" for name, count in ranked[:30]]
    summary = "; ".join(top_lines)

    full_list = ", ".join(cards[:220])
    return (
        f"Deck {deck_name}. Total cards: {len(cards)}. "
        f"Most frequent cards: {summary}. "
        f"Deck card sequence: {full_list}."
    )


def build_deck_chunks(
    decks_dir: str,
    max_decks: int = 400,
    chunk_size: int = 180,
    overlap: int = 35,
) -> List[DocumentChunk]:
    if not decks_dir or not os.path.isdir(decks_dir):
        return []

    decklists = load_decklists_from_directory(decks_dir, include_command_zone=True)
    chunks: List[DocumentChunk] = []

    for deck_name in sorted(decklists.keys())[:max(0, max_decks)]:
        cards = decklists[deck_name]
        text = _deck_text(deck_name=deck_name, cards=cards)
        doc_id = f"deck::{_slugify(deck_name)}"

        for idx, segment in enumerate(_chunk_words(text, chunk_size=chunk_size, overlap=overlap)):
            chunk_id = f"{doc_id}::chunk-{idx:03d}"
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    source="decklist",
                    title=deck_name,
                    text=segment,
                    metadata={
                        "deck_name": deck_name,
                        "deck_size": len(cards),
                    },
                )
            )

    return chunks


def _row_to_card_text(row: pd.Series) -> str:
    name = str(row.get("name", "")).strip()
    type_line = str(row.get("type_line", "")).strip()
    oracle_text = str(row.get("oracle_text", "")).strip()
    mana_cost = str(row.get("mana_cost", "")).strip()
    set_code = str(row.get("set", "")).strip().lower()
    set_name = set_name_for_code(set_code)
    color_identity = row.get("color_identity", [])
    keywords = row.get("keywords", [])

    color_identity_text = ", ".join(color_identity) if isinstance(color_identity, list) else str(color_identity)
    keywords_text = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
    set_text_parts = []
    if set_code:
        set_text_parts.append(f"Set code: {set_code}.")
    if set_name:
        set_text_parts.append(f"Set name: {set_name}.")
    set_text = " ".join(set_text_parts)

    return (
        f"Card {name}. Type: {type_line}. Mana cost: {mana_cost}. "
        f"Color identity: {color_identity_text}. Keywords: {keywords_text}. "
        f"{set_text} "
        f"Oracle text: {oracle_text}."
    )


def build_card_chunks(
    cards_csv: str,
    max_cards: int = 0,
    chunk_size: int = 180,
    overlap: int = 35,
) -> List[DocumentChunk]:
    if not cards_csv or not os.path.isfile(cards_csv):
        return []

    try:
        from mtg_io import load_card_database

        card_db = load_card_database(cards_csv)
    except Exception:
        card_db = pd.read_csv(cards_csv)

    chunks: List[DocumentChunk] = []
    unique = card_db.drop_duplicates(subset="name")

    rows = unique if int(max_cards) <= 0 else unique.head(max(0, int(max_cards)))
    for _, row in rows.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue

        doc_id = f"card::{_slugify(name)}"
        title = name
        text = _row_to_card_text(row)

        for idx, segment in enumerate(_chunk_words(text, chunk_size=chunk_size, overlap=overlap)):
            chunk_id = f"{doc_id}::chunk-{idx:03d}"
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    source="card_db",
                    title=title,
                    text=segment,
                    metadata={
                        "card_name": name,
                        "type_line": str(row.get("type_line", "")),
                        "set": str(row.get("set", "")).lower(),
                        "set_name": set_name_for_code(str(row.get("set", "")).lower()),
                    },
                )
            )

    return chunks


def build_meta_chunks(meta_json_paths: Optional[Sequence[str]]) -> List[DocumentChunk]:
    paths: List[str] = []
    if meta_json_paths:
        for path in meta_json_paths:
            if os.path.isfile(path):
                paths.append(path)

    chunks: List[DocumentChunk] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        records: Iterable[Dict[str, Any]]
        if isinstance(payload, list):
            records = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            maybe_records = payload.get("decks") or payload.get("meta") or []
            records = [item for item in maybe_records if isinstance(item, dict)]
        else:
            records = []

        for idx, item in enumerate(records):
            archetype = str(item.get("archetype") or item.get("archetype_name") or "Unknown Archetype")
            meta_percentage = item.get("meta_percentage")
            deck_count = item.get("deck_count")
            url = str(item.get("url") or "")
            text = (
                f"Archetype {archetype}. Meta percentage: {meta_percentage}. "
                f"Deck count: {deck_count}. URL: {url}."
            )
            doc_id = f"meta::{_slugify(os.path.basename(path))}::{idx:04d}"
            chunk_id = f"{doc_id}::chunk-000"
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    source="meta_json",
                    title=archetype,
                    text=text,
                    metadata={
                        "meta_percentage": meta_percentage,
                        "deck_count": deck_count,
                        "url": url,
                        "path": path,
                    },
                )
            )

    return chunks


def discover_default_meta_paths() -> List[str]:
    return sorted(glob.glob(os.path.join("json_outputs", "*.json")))


def build_domain_corpus(
    cards_csv: Optional[str] = None,
    decks_dir: Optional[str] = None,
    meta_json_paths: Optional[Sequence[str]] = None,
    max_decks: int = 400,
    max_cards: int = 0,
    chunk_size: int = 180,
    overlap: int = 35,
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []

    if decks_dir:
        chunks.extend(
            build_deck_chunks(
                decks_dir=decks_dir,
                max_decks=max_decks,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    if cards_csv:
        chunks.extend(
            build_card_chunks(
                cards_csv=cards_csv,
                max_cards=max_cards,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    if meta_json_paths is None:
        meta_json_paths = discover_default_meta_paths()
    chunks.extend(build_meta_chunks(meta_json_paths))

    return chunks
