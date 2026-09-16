"""Split loaded documents into overlapping chunks suitable for embedding.

Markdown documents are split on headers first, then any oversized section is
further split by the character splitter -- this keeps a chunk from crossing
an unrelated section boundary, and (as a side effect of keeping the header
line in the chunk body) gives each chunk a bit of free structural context.
"""

import hashlib
from dataclasses import dataclass

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.rag.loader import RawDocument

_MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    start: int = 0  # approximate character offset within the source document

    @property
    def id(self) -> str:
        return f"{self.source}::{self.chunk_index}"

    @property
    def content_hash(self) -> str:
        """Short hash of the chunk's raw text, used to detect when a source
        document has changed so a resumed ingest re-processes it instead of
        skipping it as 'already indexed' with stale content."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


def _locate(document_text: str, piece: str) -> int:
    """Best-effort offset of piece within document_text, via a fingerprint
    of its first ~80 chars. Only used to build a surrounding-context window
    for contextualization, so approximate is fine -- worst case (not found)
    just falls back to the start of the document."""
    pos = document_text.find(piece[:80])
    return pos if pos != -1 else 0


def _split_into_sections(document: RawDocument) -> list[str]:
    if not document.source.lower().endswith(".md"):
        return [document.text]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MARKDOWN_HEADERS, strip_headers=False
    )
    return [d.page_content for d in header_splitter.split_text(document.text)]


def chunk_documents(
    documents: list[RawDocument], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for document in documents:
        sections = _split_into_sections(document)
        pieces = [p for section in sections for p in char_splitter.split_text(section)]
        for i, piece in enumerate(pieces):
            start = _locate(document.text, piece)
            chunks.append(Chunk(text=piece, source=document.source, chunk_index=i, start=start))

    return chunks
