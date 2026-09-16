from app.rag.chunker import Chunk
from app.rag.vectorstore import VectorStore


def _store(tmp_path) -> VectorStore:
    return VectorStore(str(tmp_path / "chroma"), "test_collection")


def test_get_by_ids_returns_only_existing(tmp_path):
    store = _store(tmp_path)
    store.add(
        ids=["a::0"],
        embeddings=[[0.1, 0.2]],
        documents=["hello"],
        metadatas=[{"source": "a.txt", "chunk_index": 0}],
    )

    result = store.get_by_ids(["a::0", "missing::0"])

    assert set(result.keys()) == {"a::0"}
    assert result["a::0"].text == "hello"


def test_get_all_returns_everything_persisted(tmp_path):
    store = _store(tmp_path)
    store.add(
        ids=["a::0", "b::0"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        documents=["hello", "world"],
        metadatas=[
            {"source": "a.txt", "chunk_index": 0},
            {"source": "b.txt", "chunk_index": 0},
        ],
    )

    all_chunks = store.get_all()

    assert {c.id for c in all_chunks} == {"a::0", "b::0"}


def test_get_all_on_empty_store_returns_empty(tmp_path):
    store = _store(tmp_path)
    assert store.get_all() == []


def test_resumable_ingest_skips_already_indexed_chunks(tmp_path):
    """Simulates the ingest_cli resumability check: chunks already persisted
    should be excluded from the 'pending' set on a rerun."""
    store = _store(tmp_path)
    store.add(
        ids=["a::0"],
        embeddings=[[0.1, 0.2]],
        documents=["already done"],
        metadatas=[{"source": "a.txt", "chunk_index": 0}],
    )

    all_ids = ["a::0", "a::1", "b::0"]
    already_indexed = store.get_by_ids(all_ids)
    pending = [cid for cid in all_ids if cid not in already_indexed]

    assert pending == ["a::1", "b::0"]


def _pending(chunks: list[Chunk], already_indexed: dict) -> list[Chunk]:
    """Mirrors ingest_cli.py's staleness-aware pending-chunk detection."""
    return [
        c
        for c in chunks
        if c.id not in already_indexed
        or already_indexed[c.id].content_hash != c.content_hash
    ]


def test_content_hash_mismatch_marks_chunk_as_pending_again(tmp_path):
    store = _store(tmp_path)
    old_chunk = Chunk(text="original text", source="a.md", chunk_index=0)
    store.add(
        ids=[old_chunk.id],
        embeddings=[[0.1, 0.2]],
        documents=[old_chunk.text],
        metadatas=[
            {"source": "a.md", "chunk_index": 0, "content_hash": old_chunk.content_hash}
        ],
    )

    edited_chunk = Chunk(text="edited text -- source file changed", source="a.md", chunk_index=0)
    already_indexed = store.get_by_ids([edited_chunk.id])

    assert _pending([edited_chunk], already_indexed) == [edited_chunk]


def test_unchanged_chunk_is_not_pending(tmp_path):
    store = _store(tmp_path)
    chunk = Chunk(text="unchanged text", source="a.md", chunk_index=0)
    store.add(
        ids=[chunk.id],
        embeddings=[[0.1, 0.2]],
        documents=[chunk.text],
        metadatas=[{"source": "a.md", "chunk_index": 0, "content_hash": chunk.content_hash}],
    )

    same_chunk = Chunk(text="unchanged text", source="a.md", chunk_index=0)
    already_indexed = store.get_by_ids([same_chunk.id])

    assert _pending([same_chunk], already_indexed) == []


def test_legacy_chunk_without_stored_hash_is_treated_as_pending(tmp_path):
    """Chunks indexed before content_hash existed have no hash in metadata
    and must be safely re-verified rather than silently trusted as fresh."""
    store = _store(tmp_path)
    store.add(
        ids=["a::0"],
        embeddings=[[0.1, 0.2]],
        documents=["some text"],
        metadatas=[{"source": "a.md", "chunk_index": 0}],  # no content_hash
    )

    chunk = Chunk(text="some text", source="a.md", chunk_index=0)
    already_indexed = store.get_by_ids([chunk.id])

    assert _pending([chunk], already_indexed) == [chunk]
