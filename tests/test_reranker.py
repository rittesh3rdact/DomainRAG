from types import SimpleNamespace
from unittest.mock import MagicMock

from app.rag.rate_limiter import RateLimiter
from app.rag.reranker import Reranker


def test_rerank_returns_model_order():
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(
        text='{"ranked_ids": ["b", "a"]}'
    )
    reranker = Reranker(client, ["test-model"], RateLimiter(0))

    result = reranker.rerank("query", {"a": "text a", "b": "text b", "c": "text c"}, top_k=2)

    assert result == ["b", "a"]


def test_rerank_filters_out_ids_not_in_candidates():
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(
        text='{"ranked_ids": ["ghost", "a"]}'
    )
    reranker = Reranker(client, ["test-model"], RateLimiter(0))

    result = reranker.rerank("query", {"a": "text a"}, top_k=5)

    assert result == ["a"]


def test_rerank_falls_back_to_original_order_on_failure():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("boom")
    reranker = Reranker(client, ["test-model"], RateLimiter(0))

    result = reranker.rerank("query", {"a": "text a", "b": "text b"}, top_k=1)

    assert result == ["a"]


def test_rerank_empty_candidates_returns_empty():
    reranker = Reranker(MagicMock(), ["test-model"], RateLimiter(0))
    assert reranker.rerank("query", {}, top_k=5) == []
