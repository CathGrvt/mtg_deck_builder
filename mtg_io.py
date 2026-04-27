import ast
import os
import re
from typing import Any, Dict, Iterable, List

import pandas as pd


SECTION_ALIASES = {
    "deck": "mainboard",
    "sideboard": "sideboard",
    "commander": "commanders",
    "companion": "companions",
}

CARD_LINE_RE = re.compile(
    r"^(?:(\d+)[x]?\s+)?(.+?)(?:\s+[x]?(\d+))?$",
    re.IGNORECASE,
)


def normalize_card_name(card_name: str) -> str:
    """
    Normalize card name by standardizing single slash to double slash format.
    """
    if not card_name:
        return ""
    if "/" in card_name and "//" not in card_name:
        parts = card_name.split("/")
        if len(parts) == 2:
            return f"{parts[0].strip()} // {parts[1].strip()}"
    return card_name


def safe_parse_list(value: Any) -> List[str]:
    """
    Safely parse list-like values loaded from CSV fields.
    """
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
                if isinstance(parsed, tuple):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except (SyntaxError, ValueError):
                pass

            # Fallback parser for malformed list-ish strings.
            inner = text[1:-1].strip()
            if not inner:
                return []
            items = []
            for item in inner.split(","):
                cleaned = item.strip().strip("'\"")
                if cleaned:
                    items.append(cleaned)
            return items
        return [text]
    return []


def parse_decklist_lines(lines: Iterable[str]) -> Dict[str, List[str]]:
    """
    Parse decklist lines and split them into sections.

    Supports both:
    - Explicit section headers (`Commander`, `Deck`, `Companion`, `Sideboard`)
    - Constructed format blank-line separator between mainboard and sideboard
    """
    sections = {
        "mainboard": [],
        "sideboard": [],
        "commanders": [],
        "companions": [],
    }
    current_section = "mainboard"
    saw_explicit_section = False

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            # If no explicit sections are used, treat first blank line as sideboard separator.
            if (
                not saw_explicit_section
                and current_section == "mainboard"
                and sections["mainboard"]
            ):
                current_section = "sideboard"
            continue

        if line.startswith("#"):
            continue

        section_key = SECTION_ALIASES.get(line.lower())
        if section_key:
            current_section = section_key
            saw_explicit_section = True
            continue

        match = CARD_LINE_RE.match(line)
        if not match:
            continue

        count = int(match.group(1) or match.group(3) or "1")
        card_name = normalize_card_name(match.group(2).strip())
        sections[current_section].extend([card_name] * count)

    return sections


def parse_decklist_file(filepath: str) -> Dict[str, List[str]]:
    with open(filepath, "r", encoding="utf-8") as handle:
        return parse_decklist_lines(handle.readlines())


def load_decklists_from_directory(
    directory: str,
    include_command_zone: bool = True,
) -> Dict[str, List[str]]:
    """
    Load decklists from a directory.

    Returns mainboard cards by default and optionally appends commander/companion cards.
    Sideboards are always excluded.
    """
    decklists: Dict[str, List[str]] = {}

    deck_files = sorted(
        f for f in os.listdir(directory) if f.endswith(".txt") and not f.startswith(".")
    )
    for filename in deck_files:
        filepath = os.path.join(directory, filename)
        sections = parse_decklist_file(filepath)

        cards = list(sections["mainboard"])
        if include_command_zone:
            cards.extend(sections["commanders"])
            cards.extend(sections["companions"])

        if cards:
            deck_name = os.path.splitext(filename)[0]
            decklists[deck_name] = cards

    return decklists


def load_card_database(csv_path: str) -> pd.DataFrame:
    """
    Shared card database loader with consistent coercion rules.
    """
    df = pd.read_csv(csv_path)

    bool_columns = [
        "is_creature",
        "is_land",
        "is_instant_sorcery",
        "is_multicolored",
        "has_etb_effect",
        "is_legendary",
    ]
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].map(
                {"True": True, "False": False, True: True, False: False}
            )

    list_columns = ["colors", "color_identity", "keywords", "produced_mana"]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list)

    numeric_columns = ["cmc", "color_count"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df
