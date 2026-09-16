"""FastAPI app exposing a streaming chat endpoint over the agentic RAG pipeline."""

import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel

from app.config import settings
from app.rag.agent import NO_CONTEXT_MESSAGE, RAGAgent
from app.rag.bm25_index import BM25Index
from app.rag.embeddings import Embedder
from app.rag.rate_limiter import RateLimiter
from app.rag.reranker import Reranker
from app.rag.retriever import HybridRetriever
from app.rag.tools import SearchTool
from app.rag.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Domain RAG Chatbot")

_client = genai.Client(api_key=settings.google_api_key)
_embedder = Embedder(
    _client, settings.embedding_model, RateLimiter(settings.embedding_model_rpm)
)
_store = VectorStore(settings.chroma_dir, settings.collection_name)
_bm25_index = BM25Index.load(settings.bm25_index_path)
_reranker = Reranker(_client, settings.utility_models, RateLimiter(settings.utility_model_rpm))
_retriever = HybridRetriever(settings, _embedder, _store, _bm25_index, _reranker)
_search_tool = SearchTool(_retriever)
_agent = RAGAgent(
    _client,
    settings.generation_models,
    _search_tool,
    settings.max_tool_calls,
    RateLimiter(settings.generation_model_rpm),
    settings.assistant_role_description,
)

# In-memory per-session conversation history. Fine for a single-process demo;
# swap for Redis/a DB before running multiple workers or wanting persistence.
_sessions: dict[str, list[types.Content]] = {}

_chat_throttle = RateLimiter(settings.chat_requests_per_minute)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


@app.get("/api/status")
def status() -> dict:
    return {"indexed_chunks": _store.count()}


@app.post("/api/session")
def create_session() -> dict:
    session_id = str(uuid.uuid4())
    _sessions[session_id] = []
    return {"session_id": session_id}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    if not _chat_throttle.try_acquire():
        raise HTTPException(
            status_code=429,
            detail="Too many requests right now -- please wait a moment and try again.",
        )

    if _store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed yet. Run `python -m scripts.ingest_cli` first.",
        )

    session_id = request.session_id or str(uuid.uuid4())
    history = _sessions.setdefault(session_id, [])

    def event_stream():
        answer_parts: list[str] = []
        try:
            stream, chunks = _agent.answer(request.message, history)
            for token in stream:
                answer_parts.append(token)
                yield _sse({"type": "token", "text": token})

            full_answer = "".join(answer_parts)
            sources = [] if NO_CONTEXT_MESSAGE in full_answer else sorted({c.source for c in chunks})
            yield _sse({"type": "sources", "sources": sources})
        except Exception:
            logger.exception("Error while answering")
            yield _sse({"type": "error", "message": "Something went wrong generating a response."})
            return
        finally:
            yield _sse({"type": "done", "session_id": session_id})

        history.append(types.Content(role="user", parts=[types.Part(text=request.message)]))
        history.append(types.Content(role="model", parts=[types.Part(text=full_answer)]))
        del history[: max(0, len(history) - settings.max_history_turns * 2)]

    return StreamingResponse(event_stream(), media_type="text/event-stream")


_static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
