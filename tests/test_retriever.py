from unittest.mock import MagicMock

from app.config import Settings
from app.rag.retriever import HybridRetriever
from app.rag.vectorstore import RetrievedChunk


def _settings(**overrides) -> Settings:
    return Settings(google_api_key="test-key", **overrides)


def _chunk(id_, text, source="doc.txt", similarity=0.5):
    return RetrievedChunk(id=id_, text=text, source=source, chunk_index=0, similarity=similarity)


def test_retrieve_without_hybrid_or_rerank_filters_by_threshold():
    settings = _settings(enable_hybrid_search=False, enable_reranking=False, similarity_threshold=0.4)
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1]
    store = MagicMock()
    store.query.return_value = [
        _chunk("1", "relevant", similarity=0.8),
        _chunk("2", "irrelevant", similarity=0.1),
    ]
    bm25 = MagicMock()
    reranker = MagicMock()

    retriever = HybridRetriever(settings, embedder, store, bm25, reranker)
    results = retriever.retrieve("query")

    assert [c.id for c in results] == ["1"]
    bm25.query.assert_not_called()
    reranker.rerank.assert_not_called()


def test_retrieve_fuses_vector_and_bm25_then_reranks():
    settings = _settings(enable_hybrid_search=True, enable_reranking=True)
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1]
    store = MagicMock()
    store.query.return_value = [_chunk("1", "vector hit")]
    store.get_by_ids.return_value = {"2": _chunk("2", "bm25 only hit")}
    bm25 = MagicMock()
    bm25.query.return_value = ["2", "1"]
    reranker = MagicMock()
    reranker.rerank.return_value = ["2"]

    retriever = HybridRetriever(settings, embedder, store, bm25, reranker)
    results = retriever.retrieve("query")

    assert [c.id for c in results] == ["2"]
    store.get_by_ids.assert_called_once_with(["2"])
    reranker.rerank.assert_called_once()


def test_retrieve_returns_empty_when_nothing_found():
    settings = _settings()
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1]
    store = MagicMock()
    store.query.return_value = []
    bm25 = MagicMock()
    bm25.query.return_value = []
    reranker = MagicMock()

    retriever = HybridRetriever(settings, embedder, store, bm25, reranker)
    assert retriever.retrieve("query") == []
    reranker.rerank.assert_not_called()
