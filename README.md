# Domain RAG Chatbot

A retrieval-augmented chatbot that answers questions strictly from your own
documents. Uses free-tier Gemini models (Google AI Studio) for embeddings
and generation, Chroma + BM25 for hybrid retrieval, and FastAPI + a small
streaming web UI for the frontend.

## Architecture

```
data/*.pdf|.txt|.md
      │  load + chunk (markdown header-aware, then recursive splitter)
      ▼
Contextualizer (LLM prepends a situating sentence per chunk,   ── Anthropic
      │         using a bounded surrounding-text window,          Contextual
      │         not the whole document)                           Retrieval
      ▼
Embedder (Gemini embedding-001)              BM25Index (lexical, keyword)
      ▼                                              ▼
Chroma (persistent, cosine similarity)      rank_bm25 (persisted pickle)
      │                                              │
      └──────────────┬───────────────────────────────┘
                      ▼
         Reciprocal Rank Fusion (Cormack et al., 2009)
                      ▼
          LLM Reranker (RankGPT-style listwise reranking)
                      ▼
      RAGAgent: model gets a search_documents TOOL and decides
      itself when/how many times to call it (agentic RAG),
      then streams a final, source-cited answer
                      ▼
      FastAPI /api/chat (SSE streaming, per-session history)
                      ▼
      Static chat UI (source citations shown per answer)
```

This was deliberately upgraded from a naive "embed → top-k → answer" pipeline
to reflect current production RAG practice. Each stage maps to a specific
technique:

- **Contextual Retrieval** ([Anthropic, 2024](https://www.anthropic.com/news/contextual-retrieval))
  — a chunk embedded in isolation loses whatever made it unambiguous in the
  source document. Before embedding/indexing, an LLM call prepends a short
  sentence situating the chunk. One extra call per chunk, at ingestion time
  only. Toggle: `ENABLE_CONTEXTUAL_RETRIEVAL`. Anthropic's writeup assumes
  this is cheap because of Claude's prompt caching; without an equivalent
  cache here, each call gets a *bounded window* of text around the chunk
  (not the whole document) so token cost stays roughly constant regardless
  of document size, instead of scaling linearly with (document size ×
  chunk count).
- **Markdown-aware chunking** — `.md` documents are split on headers first
  (`#`/`##`/`###`), then any oversized section is further split by size.
  This keeps a chunk from straddling two unrelated sections, and since the
  header line stays in the chunk body, each chunk carries a bit of free
  structural context even before contextualization runs.
- **Content-hash-aware resumable ingestion** — each chunk's persisted
  metadata includes a hash of its source text. A rerun treats a chunk as
  "already indexed" only if both its id *and* content hash match, so
  editing a document and re-running ingestion (without `--reset`) correctly
  re-processes the changed parts instead of silently leaving stale content
  in the index.
- **Hybrid search** (dense vector + BM25 lexical, fused via **Reciprocal
  Rank Fusion**, Cormack, Clarke & Buettcher 2009) — dense embeddings miss
  exact keyword/acronym/product-code matches that lexical search catches,
  and vice versa; RRF combines both rankings without needing comparable
  score scales. Toggle: `ENABLE_HYBRID_SEARCH`.
- **LLM reranking** (RankGPT-style listwise reranking, Sun et al. 2023) — a
  larger first-stage candidate pool (optimized for recall) gets reordered by
  asking the model to judge relevance directly, which outperforms raw
  embedding-similarity ranking for final answer quality. Toggle:
  `ENABLE_RERANKING`.
- **Agentic RAG** — instead of one fixed retrieval per turn, the model is
  given a `search_documents` tool and decides itself whether and how many
  times to call it (capped at `MAX_TOOL_CALLS`). This correctly handles
  multi-part questions (separate searches per sub-question) and weak first
  results (it can reformulate and search again), instead of being locked
  into whatever a single retrieval call happened to surface.
- **Grounding & anti-hallucination**: the system prompt forces the model to
  answer only from tool results, cite `[source: file]` inline, and return a
  fixed "I don't have enough information" line when nothing relevant comes
  back — never a guess dressed up as an answer.
- **Streaming**: the final answer streams token-by-token over Server-Sent
  Events; only the tool-calling turns (which need complete structured
  function-call objects) are non-streaming.
- **Rate limiting + retry/backoff**: every embedding, contextualization,
  reranking and generation call is proactively paced to stay under a
  per-minute budget (`EMBEDDING_MODEL_RPM` / `UTILITY_MODEL_RPM` /
  `GENERATION_MODEL_RPM`), so ingestion and chat don't burst past free-tier
  limits and then have to recover. Calls still retry on 429/5xx with
  exponential backoff as a second line of defense for whatever the pacing
  doesn't catch (e.g. daily quota exhaustion, transient 503s).
- **Conversation memory**: kept in-memory per session, capped to the last
  `MAX_HISTORY_TURNS` turns. Single-process demo store — swap for Redis/a DB
  before running multiple workers.
- **Model fallback**: generation and utility calls try a primary model
  first, then fall through an ordered list of alternates (`GENERATION_MODEL_FALLBACKS`,
  `UTILITY_MODEL_FALLBACKS`) if it fails — protects against one model's
  quota exhaustion, overload, or retirement taking the whole thing down.
- **API-level throttling**: `/api/chat` rejects (`429`) requests beyond
  `CHAT_REQUESTS_PER_MINUTE` immediately, before they ever reach the model
  — protects your daily quota from a burst of requests rather than queueing
  them and hitting quota exhaustion anyway, just later.

All of the above can be dialed back via `.env` flags if you want the
simpler/cheaper single-retrieval pipeline (set `ENABLE_CONTEXTUAL_RETRIEVAL`,
`ENABLE_HYBRID_SEARCH` and `ENABLE_RERANKING` to `false`) — useful on a
tight free-tier quota or a very small document set where the added
sophistication doesn't pay for itself.

## Setup

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Create your env file and add the key:
   ```bash
   cp .env.example .env
   # edit .env and set GOOGLE_API_KEY
   ```
3. Install dependencies (Python 3.11+):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```
4. Put your domain documents in `data/` (a sample file is there to start).
5. Build the vector + BM25 indexes:
   ```bash
   python -m scripts.ingest_cli --reset
   ```
   Use `--reset` any time chunking behavior itself changes (e.g. after
   pulling an update to `chunk_documents`) or you've replaced the documents
   in `data/` — otherwise chunks from the old boundaries can linger
   alongside new ones. Day-to-day, once your index is built, ingestion is
   **resumable**: chunks are processed and persisted in small batches, and a
   rerun skips whatever's already indexed — so if it's interrupted (e.g. by
   a daily quota limit), you don't lose progress or re-spend quota
   re-processing chunks that already succeeded. Just rerun the same command
   again (after your quota resets, if that's what stopped it).
6. Start the server:
   ```bash
   uvicorn app.main:app --reload
   ```
7. Open http://localhost:8000

## Tests

```bash
pytest
```

Covers: chunking, Reciprocal Rank Fusion, BM25 indexing, the reranker, the
hybrid retriever, and the agentic tool-calling loop (all with the Gemini
client mocked — no live API calls in tests).

## Configuration

All settings are environment variables (see `.env.example`): chunk size/
overlap, retrieval pool sizes, top-k, similarity threshold, model names,
feature toggles, and storage paths.

## Notes on free-tier limits

Google AI Studio's free tier enforces per-minute and per-day request caps
that vary by model. As of testing this build, `gemini-flash-latest`'s free
tier allowed only **20 requests/day** on a fresh project — each chat turn
can use 2+ of those (one per tool call, plus contextualization calls at
ingestion time), so you can exhaust a day's quota in a handful of
conversations. Check your actual limits at
https://ai.google.dev/gemini-api/docs/rate-limits, and if you hit them
often, consider: setting `GENERATION_MODEL`/`UTILITY_MODEL` to a specific
lighter model instead of a `-latest` alias, disabling contextual retrieval
for ingestion, lowering `MAX_TOOL_CALLS`, or upgrading to a paid tier. The
retry logic backs off automatically on `429`/`503`, but it can't manufacture
quota that isn't there.
