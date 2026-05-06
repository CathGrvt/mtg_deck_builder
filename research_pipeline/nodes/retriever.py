from __future__ import annotations

from typing import Any, Dict, List

from research_pipeline.models import RetrievedChunk
from research_pipeline.retrieval.index import HybridRetrievalIndex
from research_pipeline.trace import TraceLogger


def run_retriever_node(
    state: Dict[str, Any],
    index: HybridRetrievalIndex,
    top_k_per_query: int,
    trace: TraceLogger | None = None,
) -> Dict[str, Any]:
    queries = list(state.get("active_queries") or state.get("subquestions") or [])
    if not queries:
        topic = str(state.get("topic", "")).strip()
        if topic:
            queries = [topic]

    started = trace.log_node_start(
        "retriever",
        {"query_count": len(queries), "top_k_per_query": top_k_per_query},
    ) if trace else None

    existing: Dict[str, RetrievedChunk] = {}
    for item in state.get("retrieved_chunks", []):
        if isinstance(item, dict):
            chunk = RetrievedChunk.from_dict(item)
        elif isinstance(item, RetrievedChunk):
            chunk = item
        else:
            continue
        existing[chunk.chunk_id] = chunk

    query_hits: Dict[str, List[str]] = {}
    for query in queries:
        hits = index.search(query=query, top_k=top_k_per_query)
        query_hits[query] = [hit.chunk_id for hit in hits]
        for hit in hits:
            previous = existing.get(hit.chunk_id)
            if previous is None or hit.score > previous.score:
                existing[hit.chunk_id] = hit

    merged = sorted(existing.values(), key=lambda item: item.score, reverse=True)
    max_total_chunks = max(10, top_k_per_query * max(1, len(queries)))
    merged = merged[:max_total_chunks]

    if trace and started is not None:
        trace.log_node_end(
            "retriever",
            started,
            {
                "retrieved_total": len(merged),
                "unique_docs": len({item.doc_id for item in merged}),
            },
        )

    return {
        "retrieved_chunks": [item.to_dict() for item in merged],
        "query_hits": query_hits,
    }
