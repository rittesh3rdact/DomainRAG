from app.rag.chunker import chunk_documents
from app.rag.loader import RawDocument


def test_chunk_documents_preserves_source_and_order():
    docs = [RawDocument(text="A. " * 500, source="doc1.txt")]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(c.source == "doc1.txt" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_documents_respects_chunk_size():
    docs = [RawDocument(text="word " * 1000, source="doc.txt")]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=0)

    assert all(len(c.text) <= 220 for c in chunks)  # small slack for separator matching


def test_chunk_ids_are_unique():
    docs = [
        RawDocument(text="content one " * 50, source="a.txt"),
        RawDocument(text="content two " * 50, source="b.txt"),
    ]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_markdown_documents_split_on_headers():
    text = (
        "# Intro\n\nintro text here.\n\n"
        "## Section A\n\n" + ("section a content. " * 40) + "\n\n"
        "## Section B\n\n" + ("section b content. " * 40)
    )
    docs = [RawDocument(text=text, source="doc.md")]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=0)

    # each chunk should stay within one section, not straddle both headers
    assert any("Section A" in c.text for c in chunks)
    assert any("Section B" in c.text for c in chunks)
    assert not any("Section A" in c.text and "section b content" in c.text for c in chunks)


def test_non_markdown_documents_not_header_split():
    text = "# not a real header, just text\n\n" + ("plain text. " * 100)
    docs = [RawDocument(text=text, source="doc.txt")]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=0)
    assert len(chunks) > 1  # still chunks by size, just not header-aware


def test_content_hash_changes_when_text_changes():
    docs_v1 = [RawDocument(text="original content " * 20, source="a.txt")]
    docs_v2 = [RawDocument(text="edited content " * 20, source="a.txt")]

    chunk_v1 = chunk_documents(docs_v1, chunk_size=1000, chunk_overlap=0)[0]
    chunk_v2 = chunk_documents(docs_v2, chunk_size=1000, chunk_overlap=0)[0]

    assert chunk_v1.content_hash != chunk_v2.content_hash


def test_content_hash_stable_for_same_text():
    docs = [RawDocument(text="stable content " * 20, source="a.txt")]
    chunk_a = chunk_documents(docs, chunk_size=1000, chunk_overlap=0)[0]
    chunk_b = chunk_documents(docs, chunk_size=1000, chunk_overlap=0)[0]
    assert chunk_a.content_hash == chunk_b.content_hash


def test_chunk_start_offset_is_located_within_document():
    text = "prefix filler text. " * 30 + "UNIQUE_MARKER_SECTION " + "suffix filler text. " * 30
    docs = [RawDocument(text=text, source="a.txt")]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=0)

    marker_chunk = next(c for c in chunks if "UNIQUE_MARKER_SECTION" in c.text)
    assert text[marker_chunk.start : marker_chunk.start + 20] == marker_chunk.text[:20]
