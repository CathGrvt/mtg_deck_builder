from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from gcp_agent_runtime.contracts import RetrievedEvidence, RetrievalBundle, RetrievalPlan
from research_pipeline.retrieval.corpus import build_domain_corpus
from research_pipeline.retrieval.index import HybridRetrievalIndex


class RetrieverClient(Protocol):
    def retrieve(self, plan: RetrievalPlan) -> List[RetrievedEvidence]:
        raise NotImplementedError


@dataclass
class LocalRetrieverConfig:
    cards_csv: str = "data/commander_cards.csv"
    decks_dir: str = "current_commander_decks"
    meta_json_paths: Optional[List[str]] = None
    enable_semantic: bool = True
    lexical_weight: float = 0.6
    semantic_weight: float = 0.4

    @classmethod
    def from_env(cls) -> "LocalRetrieverConfig":
        meta_paths = [
            item.strip()
            for item in os.getenv("MTG_LOCAL_RETRIEVER_META_JSON_PATHS", "").split(",")
            if item.strip()
        ]
        return cls(
            cards_csv=os.getenv("MTG_LOCAL_RETRIEVER_CARDS_CSV", cls.cards_csv),
            decks_dir=os.getenv("MTG_LOCAL_RETRIEVER_DECKS_DIR", cls.decks_dir),
            meta_json_paths=meta_paths or None,
            enable_semantic=os.getenv("MTG_LOCAL_RETRIEVER_ENABLE_SEMANTIC", "true").strip().lower()
            in {"1", "true", "yes", "on"},
            lexical_weight=float(os.getenv("MTG_LOCAL_RETRIEVER_LEXICAL_WEIGHT", str(cls.lexical_weight))),
            semantic_weight=float(os.getenv("MTG_LOCAL_RETRIEVER_SEMANTIC_WEIGHT", str(cls.semantic_weight))),
        )


class LocalHybridRetrieverClient:
    def __init__(self, config: Optional[LocalRetrieverConfig] = None):
        self.config = config or LocalRetrieverConfig()
        self._index: Optional[HybridRetrievalIndex] = None

    def _ensure_index(self) -> HybridRetrievalIndex:
        if self._index is not None:
            return self._index

        chunks = build_domain_corpus(
            cards_csv=self.config.cards_csv,
            decks_dir=self.config.decks_dir,
            meta_json_paths=self.config.meta_json_paths,
        )
        if not chunks:
            raise ValueError("Unable to build retrieval corpus from local data paths.")

        self._index = HybridRetrievalIndex(
            chunks=chunks,
            lexical_weight=self.config.lexical_weight,
            semantic_weight=self.config.semantic_weight,
            enable_semantic=self.config.enable_semantic,
        )
        return self._index

    @staticmethod
    def _passes_source_filter(source: str, plan: RetrievalPlan) -> bool:
        if not plan.corpus_targets:
            return True
        return source in set(plan.corpus_targets)

    def retrieve(self, plan: RetrievalPlan) -> List[RetrievedEvidence]:
        index = self._ensure_index()
        merged: Dict[str, RetrievedEvidence] = {}

        for query in plan.rewritten_queries:
            hits = index.search(query=query, top_k=plan.top_k_per_query)
            for hit in hits:
                if not self._passes_source_filter(hit.source, plan):
                    continue
                evidence = RetrievedEvidence(
                    doc_id=hit.doc_id,
                    chunk_id=hit.chunk_id,
                    source=hit.source,
                    title=hit.title,
                    text=hit.text,
                    score=float(hit.score),
                    metadata=dict(hit.metadata),
                )
                previous = merged.get(evidence.chunk_id)
                if previous is None or evidence.score > previous.score:
                    merged[evidence.chunk_id] = evidence

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ranked[: max(1, int(plan.max_chunks))]


@dataclass
class VertexRagRetrieverConfig:
    project_id: str
    location: str = "us-central1"
    rag_corpora: Optional[List[str]] = None
    top_k: int = 8


class VertexRagRetrieverClient:
    """
    Managed RAG Engine retrieval client.

    This class intentionally degrades with a clear error when Vertex SDK
    dependencies are unavailable in local environments. It can be used in
    production deployments where Vertex SDK is installed and configured.
    """

    def __init__(self, config: VertexRagRetrieverConfig):
        self.config = config

    def retrieve(self, plan: RetrievalPlan) -> List[RetrievedEvidence]:
        try:
            from vertexai.preview import rag  # type: ignore
        except Exception as exc:  # pragma: no cover - SDK may be unavailable locally
            raise RuntimeError(
                "Vertex RAG retrieval requested but Vertex SDK is unavailable. "
                "Install google-cloud-aiplatform with rag support in deployment."
            ) from exc

        rag_resources = []
        for corpus in self.config.rag_corpora or []:
            rag_resources.append(rag.RagResource(rag_corpus=corpus))
        if not rag_resources:
            raise ValueError("No rag_corpora configured for VertexRagRetrieverClient.")

        merged: Dict[str, RetrievedEvidence] = {}
        for query in plan.rewritten_queries:
            response = rag.retrieval_query(
                rag_resources=rag_resources,
                text=query,
                rag_retrieval_config=rag.RagRetrievalConfig(
                    top_k=max(1, int(plan.top_k_per_query)),
                ),
            )
            contexts = getattr(response, "contexts", None)
            if contexts is None:
                continue
            context_items = getattr(contexts, "contexts", [])
            for idx, item in enumerate(context_items):
                source_uri = getattr(item, "source_uri", "vertex_rag")
                text = getattr(item, "text", "")
                chunk_id = f"{source_uri}::chunk-{idx:03d}"
                evidence = RetrievedEvidence(
                    doc_id=source_uri,
                    chunk_id=chunk_id,
                    source="vertex_rag",
                    title=source_uri,
                    text=text,
                    score=float(max(0.0, 1.0 - (idx * 0.01))),
                    metadata={"source_uri": source_uri, "query": query},
                )
                previous = merged.get(chunk_id)
                if previous is None or evidence.score > previous.score:
                    merged[chunk_id] = evidence

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ranked[: max(1, int(plan.max_chunks))]


class RetrieverAgent:
    def __init__(self, client: RetrieverClient):
        self.client = client

    def retrieve(self, plan: RetrievalPlan) -> RetrievalBundle:
        chunks = self.client.retrieve(plan)
        query_hits: Dict[str, List[str]] = {}

        for query in plan.rewritten_queries:
            lowered = query.lower()
            query_hits[query] = [
                item.chunk_id
                for item in chunks
                if lowered in item.text.lower() or lowered in item.title.lower()
            ][: plan.top_k_per_query]

        provenance: Dict[str, Any] = {
            "query_hits": query_hits,
            "unique_docs": len({item.doc_id for item in chunks}),
            "chunk_count": len(chunks),
        }
        return RetrievalBundle(plan=plan, chunks=chunks, provenance=provenance)
