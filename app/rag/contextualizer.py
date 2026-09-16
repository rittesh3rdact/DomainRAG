"""Contextual Retrieval (Anthropic, 2024): https://www.anthropic.com/news/contextual-retrieval

Embedding a chunk in isolation loses whatever made it unambiguous in the
original document (what product a pronoun refers to, which section a table
belongs to, etc). Before embedding/indexing, we ask an LLM to generate a
short sentence situating the chunk within its document, and prepend that to
the chunk text.

Anthropic's own writeup assumes this runs cheaply because Claude's prompt
caching lets the full document be reused across many chunk-level calls for
free. Without an equivalent cache in play here, resending an entire
document per chunk gets expensive fast -- a 60KB document with 60 chunks
would mean ~3.6MB of repeated input text. Instead, each call gets a bounded
window of text surrounding the chunk (plus a short leading excerpt for
document-level orientation), which captures nearly all the situating value
at a small, constant token cost regardless of document size.
"""

import logging

from google import genai
from google.genai import types

from app.rag.chunker import Chunk
from app.rag.model_fallback import ModelFallback
from app.rag.rate_limiter import RateLimiter
from app.rag.retry import with_retry

logger = logging.getLogger(__name__)

_WINDOW_CHARS = 1500  # context on each side of the chunk
_INTRO_CHARS = 300  # leading excerpt for document-level orientation

_PROMPT_TEMPLATE = """<document_excerpt>
{document}
</document_excerpt>

Here is a chunk from within (or near) that excerpt that we want to situate:
<chunk>
{chunk}
</chunk>

Give a short, succinct context (1-2 sentences) to situate this chunk within \
the document, for the purpose of improving search retrieval of the chunk. \
Answer only with the succinct context, nothing else."""


def _build_excerpt(document_text: str, start: int, length: int) -> str:
    window_start = max(0, start - _WINDOW_CHARS)
    window_end = min(len(document_text), start + length + _WINDOW_CHARS)
    excerpt = document_text[window_start:window_end]
    if window_start == 0:
        return excerpt
    return f"{document_text[:_INTRO_CHARS]}\n...\n{excerpt}"


class Contextualizer:
    def __init__(self, client: genai.Client, models: list[str], rate_limiter: RateLimiter):
        self._client = client
        self._fallback = ModelFallback(models)
        self._rate_limiter = rate_limiter

    @with_retry()
    def _generate_context_with_model(self, model: str, excerpt: str, chunk_text: str) -> str:
        self._rate_limiter.wait()
        response = self._client.models.generate_content(
            model=model,
            contents=_PROMPT_TEMPLATE.format(document=excerpt, chunk=chunk_text),
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return (response.text or "").strip()

    def contextualize(self, document_text: str, chunk: Chunk) -> str:
        """Return chunk.text prefixed with an LLM-generated situating sentence.

        Falls back to the bare chunk on any generation failure (across all
        fallback models), so a single flaky call never blocks the whole
        ingestion run.
        """
        excerpt = _build_excerpt(document_text, chunk.start, len(chunk.text))
        try:
            context = self._fallback.call(
                lambda model: self._generate_context_with_model(model, excerpt, chunk.text)
            )
        except Exception:
            logger.exception("Contextualization failed for a chunk, using bare chunk text")
            return chunk.text

        if not context:
            return chunk.text
        return f"{context}\n\n{chunk.text}"
