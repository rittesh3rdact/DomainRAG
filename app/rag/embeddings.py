"""Gemini embedding wrapper (Google AI Studio, free tier)."""

from google import genai
from google.genai import types

from app.rag.rate_limiter import RateLimiter
from app.rag.retry import with_retry

_BATCH_SIZE = 90  # stay under the API's per-request item cap


class Embedder:
    def __init__(self, client: genai.Client, model: str, rate_limiter: RateLimiter):
        self._client = client
        self._model = model
        self._rate_limiter = rate_limiter

    @with_retry()
    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        self._rate_limiter.wait()
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            vectors.extend(self._embed_batch(batch, "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], "RETRIEVAL_QUERY")[0]
