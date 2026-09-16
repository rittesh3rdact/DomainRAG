"""LLM-based listwise reranking (RankGPT-style: Sun et al., 2023,
"Is ChatGPT Good at Search? Investigating Large Language Models as
Re-Ranking Agents"). A fast first-stage retriever (vector + BM25) optimizes
for recall over a larger candidate pool; this reorders that pool by asking
the model to judge relevance directly against the query, which consistently
beats raw embedding similarity for final ranking quality.
"""

import json
import logging

from google import genai
from google.genai import types

from app.rag.model_fallback import ModelFallback
from app.rag.rate_limiter import RateLimiter
from app.rag.retry import with_retry

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """Query: {query}

Below are candidate passages, each with an id. Rank them by how relevant \
they are to answering the query, most relevant first. Exclude passages that \
are not relevant at all. Return only passage ids.

{candidates}"""

_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "ranked_ids": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
        )
    },
    required=["ranked_ids"],
)


class Reranker:
    def __init__(self, client: genai.Client, models: list[str], rate_limiter: RateLimiter):
        self._client = client
        self._fallback = ModelFallback(models)
        self._rate_limiter = rate_limiter

    @with_retry()
    def _rank_with_model(self, model: str, prompt: str) -> list[str]:
        self._rate_limiter.wait()
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
            ),
        )
        return json.loads(response.text).get("ranked_ids", [])

    def rerank(
        self, query: str, candidates: dict[str, str], top_k: int
    ) -> list[str]:
        """candidates: id -> text. Returns ids ordered by relevance, best first."""
        if not candidates:
            return []

        listing = "\n\n".join(f"[{cid}]\n{text}" for cid, text in candidates.items())
        prompt = _PROMPT_TEMPLATE.format(query=query, candidates=listing)

        try:
            ranked_ids = self._fallback.call(lambda model: self._rank_with_model(model, prompt))
        except Exception:
            logger.exception("Reranking failed, falling back to original candidate order")
            return list(candidates.keys())[:top_k]

        # Keep only ids the model actually returned that we gave it, in its
        # order; if it returned nothing usable, fall back to original order.
        valid_ranked = [cid for cid in ranked_ids if cid in candidates]
        if not valid_ranked:
            return list(candidates.keys())[:top_k]
        return valid_ranked[:top_k]
