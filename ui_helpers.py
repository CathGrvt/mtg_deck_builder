from typing import List


def parse_card_list(raw: str) -> List[str]:
    """
    Parse comma/newline separated card names into a clean list.
    """
    if not raw.strip():
        return []

    names: List[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        name = chunk.strip()
        if name:
            names.append(name)

    seen = set()
    deduped: List[str] = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(name)
    return deduped
