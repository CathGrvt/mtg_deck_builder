from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional


# Start with practical high-signal aliases for the MTG sets users frequently
# mention by name in natural language.
SET_CODE_TO_NAME: Dict[str, str] = {
    "lrw": "Lorwyn",
    "mor": "Morningtide",
    "shm": "Shadowmoor",
    "eve": "Eventide",
    "stx": "Strixhaven: School of Mages",
}


SET_NAME_TO_CODE: Dict[str, str] = {
    "lorwyn": "lrw",
    "morningtide": "mor",
    "shadowmoor": "shm",
    "eventide": "eve",
    "strixhaven": "stx",
    "strixhaven school of mages": "stx",
    "school of mages": "stx",
}


def normalize_set_code(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    if text in SET_CODE_TO_NAME:
        return text
    return SET_NAME_TO_CODE.get(text, "")


def set_name_for_code(code: str) -> str:
    raw = str(code or "").strip().lower()
    if not raw:
        return ""
    canonical = SET_NAME_TO_CODE.get(raw, raw)
    if canonical in SET_CODE_TO_NAME:
        return SET_CODE_TO_NAME[canonical]
    if re.fullmatch(r"[a-z0-9]{2,5}", canonical):
        return canonical.upper()
    return ""


def extract_set_codes_from_text(
    text: str,
    valid_codes: Optional[Iterable[str]] = None,
) -> List[str]:
    lowered = str(text or "").lower()
    if not lowered:
        return []

    allowed_codes = {str(item).strip().lower() for item in (valid_codes or []) if str(item).strip()}
    if not allowed_codes:
        allowed_codes = set(SET_CODE_TO_NAME.keys())

    found: List[str] = []
    seen = set()

    # Match aliases by explicit name first.
    for alias, code in sorted(SET_NAME_TO_CODE.items(), key=lambda item: -len(item[0])):
        if alias in lowered and code in allowed_codes and code not in seen:
            seen.add(code)
            found.append(code)

    # Also honor explicit set-code mentions in user text.
    for token in re.findall(r"\b[a-z0-9]{2,5}\b", lowered):
        if token in allowed_codes:
            code = token
        else:
            # Fallback for codes like m20/mh3 that include digits.
            code = token if any(ch.isdigit() for ch in token) and token in allowed_codes else ""
        if code and code not in seen:
            seen.add(code)
            found.append(code)

    return found
