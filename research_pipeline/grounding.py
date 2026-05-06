from __future__ import annotations

import re
from typing import Iterable, Set


def tokenize(text: str) -> Set[str]:
    tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9']+", text.lower())
        if len(token) >= 3
    }
    return tokens


def lexical_overlap_score(text_a: str, text_b: str) -> float:
    a = tokenize(text_a)
    b = tokenize(text_b)
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return float(intersection / union)


def max_overlap_against_claim(claim: str, evidence_texts: Iterable[str]) -> float:
    best = 0.0
    for text in evidence_texts:
        score = lexical_overlap_score(claim, text)
        if score > best:
            best = score
    return best
