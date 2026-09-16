"""Agentic RAG loop: the model is given a search tool and decides itself
whether, and how many times, to call it before answering -- rather than a
single fixed retrieval per turn. This handles multi-part questions (the
model can issue separate searches per sub-question) and weak initial
results (it can reformulate and search again) far better than naive RAG.

Tool-calling turns run non-streaming (a function call must arrive as a
complete structured object); once the model is done searching, the final
answer is streamed to the client.
"""

import logging
from collections.abc import Iterator

from google import genai
from google.genai import types

from app.config import Settings
from app.rag.model_fallback import ModelFallback
from app.rag.rate_limiter import RateLimiter
from app.rag.retry import with_retry
from app.rag.tools import SEARCH_TOOL, SearchTool
from app.rag.vectorstore import RetrievedChunk

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = "I don't have enough information in the provided documents to answer that."


def build_system_instruction(role_description: str) -> str:
    """role_description comes from Settings.assistant_role_description, so
    the prompt reflects whatever's actually in data/ without editing code."""
    return f"""You are {role_description}

You answer using only what you find via the search_documents tool -- never your own \
outside knowledge, even if you're confident about it. If a question falls outside your \
domains entirely, say so rather than guessing or answering from general knowledge.

How you work:
- Search before answering anything factual. Search again, with a different or more \
specific query, if the first results don't cover the question -- for example when a \
question has multiple parts, or the first search comes back thin.
- Ground every claim in what you actually found. If the search results don't cover the \
question, say so plainly: "{NO_CONTEXT_MESSAGE}" Don't soften it, hedge around it, or \
guess anyway.
- If sources disagree or seem inconsistent with each other, say so instead of silently \
picking one.
- When something comes from a document, mention where it came from the way a person \
would in conversation -- e.g. "the finance guide covers this" or "according to the \
pickleball rules" -- not a bracketed citation tag.
- Write like a person talking, not a document: plain sentences, no markdown at all -- no \
asterisks, no bullet lists, no headers. If you're naming a few things, weave them into a \
sentence instead of a list.
- Be direct and concise. Don't repeat the question back, don't narrate that you're \
searching, and don't add hedging filler ("it seems", "it appears") when the source is clear.
"""

class RAGAgent:
    def __init__(
        self,
        client: genai.Client,
        models: list[str],
        search_tool: SearchTool,
        max_tool_calls: int,
        rate_limiter: RateLimiter,
        role_description: str = "a knowledgeable domain expert assistant",
    ):
        self._client = client
        self._fallback = ModelFallback(models)
        self._search_tool = search_tool
        self._max_tool_calls = max_tool_calls
        self._rate_limiter = rate_limiter
        self._system_instruction = build_system_instruction(role_description)

    @with_retry()
    def _generate_with_model(self, model: str, contents: list[types.Content], allow_tools: bool):
        self._rate_limiter.wait()
        return self._client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self._system_instruction,
                temperature=0.2,
                tools=[SEARCH_TOOL] if allow_tools else None,
            ),
        )

    def _generate(self, contents: list[types.Content], allow_tools: bool):
        return self._fallback.call(
            lambda model: self._generate_with_model(model, contents, allow_tools)
        )

    @with_retry()
    def _stream_with_model(self, model: str, contents: list[types.Content]):
        self._rate_limiter.wait()
        return self._client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self._system_instruction,
                temperature=0.2,
            ),
        )

    def _stream(self, contents: list[types.Content]):
        return self._fallback.call(lambda model: self._stream_with_model(model, contents))

    @staticmethod
    def _function_calls(response) -> list:
        parts = response.candidates[0].content.parts or []
        return [p.function_call for p in parts if p.function_call is not None]

    def answer(
        self, question: str, history: list[types.Content]
    ) -> tuple[Iterator[str], list[RetrievedChunk]]:
        contents = [*history, types.Content(role="user", parts=[types.Part(text=question)])]
        all_chunks: dict[str, RetrievedChunk] = {}

        for _ in range(self._max_tool_calls):
            response = self._generate(contents, allow_tools=True)
            calls = self._function_calls(response)
            if not calls:
                text = response.text or ""
                return iter([text]), list(all_chunks.values())

            contents.append(response.candidates[0].content)
            for call in calls:
                result, chunks = self._search_tool.run(call.args.get("query", ""))
                for chunk in chunks:
                    all_chunks[chunk.id] = chunk
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=call.name, response=result
                            )
                        ],
                    )
                )

        # Tool-call budget exhausted: force a final, tool-free answer.
        logger.warning("Max tool calls (%d) reached, forcing final answer", self._max_tool_calls)

        def final_stream() -> Iterator[str]:
            for chunk in self._stream(contents):
                if chunk.text:
                    yield chunk.text

        return final_stream(), list(all_chunks.values())
