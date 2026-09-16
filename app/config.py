from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str

    # Injected into the system prompt to give the assistant a role tailored
    # to what's actually in data/, rather than a generic "domain expert".
    # Update this if you swap in a different document set.
    assistant_role_description: str = (
        "a knowledgeable assistant across three specific domains: personal finance, "
        "Indian law, and pickleball rules and strategy. You don't cover other topics."
    )

    embedding_model: str = "models/gemini-embedding-001"

    # "-latest" aliases track Google's current stable flash model, but that
    # also means the underlying model changes over time and can hit issues
    # (quota exhaustion, overload) independent of your own usage. If the
    # primary model fails, these fallbacks are tried in order -- comma
    # separated, no spaces needed. Pick models confirmed available on your
    # key via `client.models.list()` if you customize this.
    generation_model: str = "gemini-flash-latest"
    generation_model_fallbacks: str = "gemini-2.5-flash,gemini-2.5-pro"

    # Cheaper/faster model used for per-chunk contextualization and
    # reranking, where a lighter model is a good cost/quality trade-off.
    utility_model: str = "gemini-flash-lite-latest"
    utility_model_fallbacks: str = "gemini-2.5-flash-lite,gemini-2.5-flash"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval pool sizes: candidates are pulled generously from each
    # retriever, fused, then reranked down to top_k for the final prompt.
    vector_candidates: int = 15
    bm25_candidates: int = 15
    rerank_candidates: int = 10
    top_k: int = 5
    similarity_threshold: float = 0.35  # min cosine similarity to keep a chunk

    enable_contextual_retrieval: bool = True
    enable_hybrid_search: bool = True
    enable_reranking: bool = True
    max_tool_calls: int = 3  # cap on agentic retrieval iterations per turn

    # Proactive requests-per-minute pacing per model, to stay under free-tier
    # quotas instead of bursting and reactively backing off after 429s.
    # Conservative defaults -- raise them if your quota tier allows more;
    # check actual limits at https://ai.google.dev/gemini-api/docs/rate-limits.
    embedding_model_rpm: int = 10
    utility_model_rpm: int = 10
    generation_model_rpm: int = 10

    # Requests the /api/chat endpoint itself will accept per minute, across
    # all clients -- rejected with 429 immediately rather than queued, so a
    # burst of incoming requests can't silently drain the daily quota.
    chat_requests_per_minute: int = 10

    chroma_dir: str = "./chroma_db"
    collection_name: str = "domain_docs"
    bm25_index_path: str = "./chroma_db/bm25_index.pkl"
    data_dir: str = "./data"

    max_history_turns: int = 6

    @property
    def generation_models(self) -> list[str]:
        return _model_chain(self.generation_model, self.generation_model_fallbacks)

    @property
    def utility_models(self) -> list[str]:
        return _model_chain(self.utility_model, self.utility_model_fallbacks)


def _model_chain(primary: str, fallbacks: str) -> list[str]:
    extras = [m.strip() for m in fallbacks.split(",") if m.strip()]
    return [primary, *[m for m in extras if m != primary]]


settings = Settings()
