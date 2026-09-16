"""Load raw text out of a folder of domain documents (.pdf, .txt, .md)."""

import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class RawDocument:
    text: str
    source: str  # relative file path, used for citations


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(data_dir: str) -> list[RawDocument]:
    """Recursively load every supported file under data_dir."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: list[RawDocument] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue

        try:
            text = _load_pdf(path) if path.suffix.lower() == ".pdf" else _load_text(path)
        except Exception:
            logger.exception("Failed to load %s, skipping", path)
            continue

        if not text.strip():
            logger.warning("No extractable text in %s, skipping", path)
            continue

        documents.append(RawDocument(text=text, source=str(path.relative_to(root))))

    return documents
