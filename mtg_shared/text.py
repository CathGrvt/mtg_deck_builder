from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Sequence

DEFAULT_KEYWORD_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "your",
        "their",
        "about",
        "what",
        "which",
        "when",
        "where",
        "how",
        "why",
        "mtg",
        "magic",
        "gathering",
        "deck",
        "cards",
    }
)


def keyword_tokens(
    text: str,
    *,
    min_length: int = 4,
    stopwords: Iterable[str] | None = None,
) -> List[str]:
    selected_stopwords = set(DEFAULT_KEYWORD_STOPWORDS)
    if stopwords is not None:
        selected_stopwords = {str(item).strip().lower() for item in stopwords if str(item).strip()}

    return [
        token
        for token in re.findall(r"[A-Za-z0-9']+", str(text or "").lower())
        if len(token) >= max(1, int(min_length)) and token not in selected_stopwords
    ]


def dedupe_preserving_order(items: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
        return {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}

    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        return {}
    return {}
