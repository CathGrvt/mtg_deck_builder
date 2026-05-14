from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from gcp_agent_runtime.contracts import DeckRecommendationRequest, RetrievalPlan


def _keyword_tokens(text: str) -> List[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "what",
        "which",
        "where",
        "when",
        "deck",
        "cards",
        "commander",
        "mtg",
    }
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9']+", str(text).lower())
        if len(token) >= 4 and token not in stopwords
    ]


@dataclass
class QueryRewriteSettings:
    max_rewrites: int = 5
    top_k_per_query: int = 8
    max_chunks: int = 40


class QueryRewriteAgent:
    def __init__(self, settings: QueryRewriteSettings | None = None):
        self.settings = settings or QueryRewriteSettings()

    def build_retrieval_plan(
        self,
        request: DeckRecommendationRequest,
        additional_queries: Sequence[str] | None = None,
    ) -> RetrievalPlan:
        queries: List[str] = []
        base = request.user_query.strip()
        if base:
            queries.append(base)

        if request.colors:
            joined = "".join(request.colors)
            queries.append(f"{base} color identity {joined}".strip())

        if request.archetype_hint:
            queries.append(f"{base} archetype {request.archetype_hint}".strip())

        if request.must_include:
            include_str = ", ".join(request.must_include[:6])
            queries.append(f"{base} include cards {include_str}".strip())

        keywords = _keyword_tokens(base)
        if keywords:
            queries.append(" ".join(keywords[:6]))

        if additional_queries:
            for query in additional_queries:
                query = str(query).strip()
                if query:
                    queries.append(query)

        deduped: List[str] = []
        seen = set()
        for query in queries:
            key = query.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(query)

        if not deduped:
            deduped.append("mtg commander deck recommendation")

        return RetrievalPlan(
            rewritten_queries=deduped[: max(1, self.settings.max_rewrites)],
            corpus_targets=["decklist", "card_db", "meta_json"],
            metadata_filters={
                "format": request.format.lower(),
                "colors": list(request.colors),
            },
            top_k_per_query=max(1, int(self.settings.top_k_per_query)),
            max_chunks=max(1, int(self.settings.max_chunks)),
        )
