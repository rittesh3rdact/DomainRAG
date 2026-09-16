"""Build (or resume/extend) the vector + BM25 indexes from everything under
DATA_DIR.

Processes and persists chunks in small batches so a mid-run failure (e.g. a
quota limit) doesn't lose already-completed work: each batch's
contextualization + embedding + storage completes before moving to the
next. Rerunning without --reset skips chunks already indexed, so
interrupted runs resume instead of re-spending API quota from scratch.

Usage:
    python -m scripts.ingest_cli [--reset]
"""

import argparse
import logging
import sys

from google import genai

from app.config import settings
from app.rag.bm25_index import BM25Document, BM25Index
from app.rag.chunker import chunk_documents, Chunk
from app.rag.contextualizer import Contextualizer
from app.rag.embeddings import Embedder
from app.rag.loader import load_documents
from app.rag.rate_limiter import RateLimiter
from app.rag.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_BATCH_SIZE = 20


def _process_batch(
    batch: list[Chunk],
    document_text_by_source: dict[str, str],
    contextualizer: Contextualizer | None,
    embedder: Embedder,
    store: VectorStore,
) -> None:
    if contextualizer is not None:
        texts = [
            contextualizer.contextualize(document_text_by_source[c.source], c)
            for c in batch
        ]
    else:
        texts = [c.text for c in batch]

    embeddings = embedder.embed_documents(texts)
    store.add(
        ids=[c.id for c in batch],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"source": c.source, "chunk_index": c.chunk_index, "content_hash": c.content_hash}
            for c in batch
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true", help="Drop the existing collection before ingesting"
    )
    args = parser.parse_args()

    documents = load_documents(settings.data_dir)
    if not documents:
        logger.error("No supported documents (.pdf, .txt, .md) found in %s", settings.data_dir)
        sys.exit(1)
    logger.info("Loaded %d document(s)", len(documents))

    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    logger.info("Split into %d chunk(s)", len(chunks))

    client = genai.Client(api_key=settings.google_api_key)
    embedder = Embedder(
        client, settings.embedding_model, RateLimiter(settings.embedding_model_rpm)
    )
    store = VectorStore(settings.chroma_dir, settings.collection_name)

    if args.reset:
        logger.info("Resetting existing collection")
        store.reset()

    already_indexed = store.get_by_ids([c.id for c in chunks])
    pending = [
        c
        for c in chunks
        if c.id not in already_indexed
        or already_indexed[c.id].content_hash != c.content_hash
    ]
    unchanged = len(chunks) - len(pending)
    logger.info("%d chunk(s) unchanged, %d pending (new or edited)", unchanged, len(pending))

    contextualizer = None
    if settings.enable_contextual_retrieval and pending:
        logger.info(
            "Generating contextual prefixes (Anthropic Contextual Retrieval technique) "
            "-- one LLM call per chunk, paced at %d/min, processed in batches of %d so "
            "progress is saved incrementally",
            settings.utility_model_rpm,
            _BATCH_SIZE,
        )
        contextualizer = Contextualizer(
            client, settings.utility_models, RateLimiter(settings.utility_model_rpm)
        )

    document_text_by_source = {d.source: d.text for d in documents}

    for batch_start in range(0, len(pending), _BATCH_SIZE):
        batch = pending[batch_start : batch_start + _BATCH_SIZE]
        _process_batch(batch, document_text_by_source, contextualizer, embedder, store)
        logger.info(
            "Persisted %d/%d pending chunk(s)", batch_start + len(batch), len(pending)
        )

    logger.info("Vector store total: %d", store.count())

    if settings.enable_hybrid_search:
        all_chunks = store.get_all()
        bm25_index = BM25Index()
        bm25_index.build([BM25Document(id=c.id, text=c.text) for c in all_chunks])
        bm25_index.save(settings.bm25_index_path)
        logger.info("Built BM25 lexical index (%d chunks) at %s", len(all_chunks), settings.bm25_index_path)


if __name__ == "__main__":
    main()
