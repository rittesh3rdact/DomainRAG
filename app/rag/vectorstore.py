"""Thin wrapper around a persistent Chroma collection.

Embeddings are computed ourselves (via Embedder) and passed in directly, so
Chroma is used purely as a vector index + metadata/document store, not as an
embedding provider. It's also the canonical lookup for chunk text/metadata
by id, used by the BM25 side of hybrid retrieval to resolve its keyword-only
hits back into full chunks, and by ingestion to detect whether a
previously-indexed chunk's source content has since changed.
"""

from dataclasses import dataclass

import chromadb


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    chunk_index: int
    similarity: float | None  # cosine similarity, higher is better; None if not vector-scored
    content_hash: str | None = None  # Chunk.content_hash at ingestion time


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    @staticmethod
    def _to_chunk(chunk_id: str, text: str, metadata: dict, similarity: float | None) -> RetrievedChunk:
        return RetrievedChunk(
            id=chunk_id,
            text=text,
            source=metadata["source"],
            chunk_index=metadata["chunk_index"],
            similarity=similarity,
            content_hash=metadata.get("content_hash"),
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
        )

        return [
            self._to_chunk(chunk_id, text, metadata, 1 - distance)  # cosine distance -> similarity
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    def get_all(self) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []

        result = self._collection.get(include=["documents", "metadatas"])
        return [
            self._to_chunk(chunk_id, text, metadata, None)
            for chunk_id, text, metadata in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def get_by_ids(self, ids: list[str]) -> dict[str, RetrievedChunk]:
        if not ids:
            return {}

        result = self._collection.get(ids=ids, include=["documents", "metadatas"])
        return {
            chunk_id: self._to_chunk(chunk_id, text, metadata, None)
            for chunk_id, text, metadata in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        }
