"""Hybrid retrieval: dense vector search + BM25 lexical search, fused with
Reciprocal Rank Fusion, then optionally reranked by an LLM. Each stage can
be toggled off via config, degrading gracefully back to plain vector search.
"""

import logging

from app.config import Settings
from app.rag.bm25_index import BM25Index
from app.rag.embeddings import Embedder
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.reranker import Reranker
from app.rag.vectorstore import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        bm25_index: BM25Index,
        reranker: Reranker,
    ):
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._bm25_index = bm25_index
        self._reranker = reranker

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        s = self._settings

        query_vector = self._embedder.embed_query(query)
        vector_hits = self._store.query(query_vector, top_k=s.vector_candidates)
        vector_by_id = {c.id: c for c in vector_hits}

        if s.enable_hybrid_search:
            bm25_ids = self._bm25_index.query(query, top_k=s.bm25_candidates)
            fused_ids = reciprocal_rank_fusion(
                [[c.id for c in vector_hits], bm25_ids]
            )
        else:
            fused_ids = [c.id for c in vector_hits]

        if not fused_ids:
            return []

        # Resolve every fused id to full chunk content. Ids already present
        # from the vector query are reused (they carry a similarity score);
        # BM25-only ids are looked up from the store.
        missing_ids = [cid for cid in fused_ids if cid not in vector_by_id]
        resolved = {**vector_by_id, **self._store.get_by_ids(missing_ids)}
        candidate_pool = [resolved[cid] for cid in fused_ids if cid in resolved]

        if not s.enable_reranking:
            filtered = [
                c
                for c in candidate_pool
                if c.similarity is None or c.similarity >= s.similarity_threshold
            ]
            return filtered[: s.top_k]

        pool = candidate_pool[: s.rerank_candidates]
        ranked_ids = self._reranker.rerank(
            query, {c.id: c.text for c in pool}, top_k=s.top_k
        )
        by_id = {c.id: c for c in pool}
        return [by_id[cid] for cid in ranked_ids if cid in by_id]
