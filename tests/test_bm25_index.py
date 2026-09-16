from app.rag.bm25_index import BM25Document, BM25Index


def test_query_ranks_keyword_matches_first():
    index = BM25Index()
    index.build(
        [
            BM25Document(id="1", text="the quick brown fox jumps over the lazy dog"),
            BM25Document(id="2", text="completely unrelated passage about cooking pasta"),
            BM25Document(id="3", text="another fox related document about foxes in the wild"),
        ]
    )

    results = index.query("fox", top_k=2)

    assert set(results) <= {"1", "3"}
    assert "2" not in results


def test_query_on_empty_index_returns_empty():
    index = BM25Index()
    index.build([])
    assert index.query("anything", top_k=5) == []


def test_save_and_load_roundtrip(tmp_path):
    index = BM25Index()
    index.build(
        [
            BM25Document(id="1", text="hello world"),
            BM25Document(id="2", text="goodbye moon"),
        ]
    )
    path = tmp_path / "bm25.pkl"
    index.save(str(path))

    loaded = BM25Index.load(str(path))
    assert loaded.query("hello", top_k=1) == ["1"]


def test_load_missing_file_returns_empty_index(tmp_path):
    loaded = BM25Index.load(str(tmp_path / "missing.pkl"))
    assert loaded.query("anything", top_k=5) == []
