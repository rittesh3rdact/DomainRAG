from app.rag.contextualizer import _INTRO_CHARS, _WINDOW_CHARS, _build_excerpt


def test_excerpt_bounded_regardless_of_document_size():
    document = "x" * 100_000
    excerpt = _build_excerpt(document, start=50_000, length=100)

    # roughly: intro + separator + window before/after the chunk
    assert len(excerpt) < _INTRO_CHARS + _WINDOW_CHARS * 2 + 200


def test_excerpt_includes_chunk_region():
    document = "before " * 3000 + "TARGET_CHUNK_TEXT" + " after" * 3000
    start = document.index("TARGET_CHUNK_TEXT")
    excerpt = _build_excerpt(document, start=start, length=len("TARGET_CHUNK_TEXT"))

    assert "TARGET_CHUNK_TEXT" in excerpt


def test_short_document_returns_whole_thing_without_intro_prefix():
    document = "a short document, well under the window size"
    excerpt = _build_excerpt(document, start=0, length=len(document))
    assert excerpt == document


def test_excerpt_near_document_start_has_no_redundant_intro():
    document = "y" * 50_000
    excerpt = _build_excerpt(document, start=0, length=100)
    # window_start == 0 here, so no separate intro block should be prepended
    assert "..." not in excerpt
