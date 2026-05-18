from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Mapping, Set, Tuple


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def tokenize(
    text: str,
    min_token_length: int = 3,
    stopwords: Iterable[str] | None = None,
) -> Set[str]:
    selected_stopwords = {str(item).strip().lower() for item in stopwords} if stopwords else set()
    tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9']+", text.lower())
        if len(token) >= max(1, int(min_token_length)) and token not in selected_stopwords
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


def claim_coverage_score(claim: str, evidence: str) -> float:
    claim_tokens = tokenize(claim)
    evidence_tokens = tokenize(evidence)
    if not claim_tokens or not evidence_tokens:
        return 0.0
    intersection = len(claim_tokens & evidence_tokens)
    return float(intersection / len(claim_tokens))


def build_token_idf(
    texts: Iterable[str],
    min_token_length: int = 4,
) -> Dict[str, float]:
    docs: List[Set[str]] = []
    for text in texts:
        tokens = tokenize(str(text or ""), min_token_length=min_token_length)
        if tokens:
            docs.append(tokens)

    doc_count = len(docs)
    if doc_count == 0:
        return {}

    doc_frequency: Dict[str, int] = {}
    for doc_tokens in docs:
        for token in doc_tokens:
            doc_frequency[token] = doc_frequency.get(token, 0) + 1

    return {
        token: float(math.log((1.0 + doc_count) / (1.0 + freq)) + 1.0)
        for token, freq in doc_frequency.items()
    }


def _weighted_token_sum(tokens: Set[str], token_idf: Mapping[str, float] | None) -> float:
    if not tokens:
        return 0.0
    if not token_idf:
        return float(len(tokens))
    return float(sum(float(token_idf.get(token, 1.0)) for token in tokens))


def topic_alignment(
    claim: str,
    topic: str,
    token_idf: Mapping[str, float] | None = None,
) -> Tuple[float, List[str]]:
    topic_keywords = tokenize(topic, min_token_length=4)
    claim_keywords = tokenize(claim, min_token_length=4)
    if not topic_keywords or not claim_keywords:
        return 0.0, []

    overlap = topic_keywords & claim_keywords
    if not overlap:
        return 0.0, []

    overlap_weight = _weighted_token_sum(overlap, token_idf)
    claim_weight = _weighted_token_sum(claim_keywords, token_idf)
    topic_weight = _weighted_token_sum(topic_keywords, token_idf)
    if claim_weight <= 0.0 or topic_weight <= 0.0:
        return 0.0, sorted(overlap)

    claim_precision = overlap_weight / claim_weight
    topic_recall = overlap_weight / topic_weight
    score = (0.7 * claim_precision) + (0.3 * topic_recall)
    return float(score), sorted(overlap)


def combined_overlap_score(
    claim: str,
    evidence: str,
    jaccard_weight: float = 0.4,
    coverage_weight: float = 0.6,
) -> float:
    jaccard = lexical_overlap_score(claim, evidence)
    coverage = claim_coverage_score(claim, evidence)
    return float((jaccard_weight * jaccard) + (coverage_weight * coverage))


def split_sentences(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(raw) if segment and segment.strip()]
    if parts:
        return parts
    return [raw]


def evidence_spans(
    text: str,
    max_span_sentences: int = 2,
    max_span_chars: int = 420,
) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    spans: List[str] = []
    cap = max(1, int(max_span_sentences))
    for span_size in range(1, cap + 1):
        if len(sentences) < span_size:
            continue
        for idx in range(0, len(sentences) - span_size + 1):
            span = " ".join(sentences[idx : idx + span_size]).strip()
            if not span:
                continue
            if len(span) > max_span_chars:
                span = span[:max_span_chars].rstrip()
            if span:
                spans.append(span)

    if not spans:
        fallback = " ".join(sentences).strip()
        if fallback:
            return [fallback[:max_span_chars].rstrip()]
    return spans


def best_overlap_against_claim(
    claim: str,
    evidence_texts: Iterable[str],
    max_span_sentences: int = 2,
) -> Tuple[float, str]:
    best = 0.0
    best_span = ""

    for text in evidence_texts:
        spans = evidence_spans(text, max_span_sentences=max_span_sentences)
        if not spans and str(text or "").strip():
            spans = [str(text).strip()]
        for span in spans:
            score = combined_overlap_score(claim, span)
            if score > best:
                best = score
                best_span = span

    return best, best_span


def max_overlap_against_claim(claim: str, evidence_texts: Iterable[str]) -> float:
    score, _ = best_overlap_against_claim(claim, evidence_texts)
    return score
