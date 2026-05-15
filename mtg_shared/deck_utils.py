from __future__ import annotations

import ast
import math
from typing import Any, List


def normalize_card_name(card_name: str) -> str:
    if not card_name:
        return ""
    if "/" in card_name and "//" not in card_name:
        parts = card_name.split("/")
        if len(parts) == 2:
            return f"{parts[0].strip()} // {parts[1].strip()}"
    return card_name


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def safe_parse_list(value: Any) -> List[str]:
    if _is_missing(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
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


def parse_card_list(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []

    names: List[str] = []
    for chunk in text.replace("\n", ",").split(","):
        name = chunk.strip()
        if name:
            names.append(name)

    seen = set()
    deduped: List[str] = []
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped
