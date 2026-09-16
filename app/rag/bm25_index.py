"""Lexical (keyword) retrieval side of hybrid search, via BM25 (Robertson &
Zaragoza, 2009 -- the standard probabilistic ranking function; still the
strongest simple baseline for exact keyword/term matches that dense
embeddings can miss, e.g. product codes, acronyms, proper nouns)."""

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Document:
    id: str
    text: str


class BM25Index:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []

    def build(self, documents: list[BM25Document]) -> None:
        self._ids = [d.id for d in documents]
        tokenized_corpus = [_tokenize(d.text) for d in documents]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def query(self, text: str, top_k: int) -> list[str]:
        if self._bm25 is None or not self._ids:
            return []

        query_terms = set(_tokenize(text))
        if not query_terms:
            return []

        scores = self._bm25.get_scores(list(query_terms))
        # BM25 IDF can be zero/negative for terms common across a small
        # corpus, so we don't filter on score sign -- only on whether the
        # document shares any query term at all.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._ids[i] for i in ranked[:top_k]]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "ids": self._ids}, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        index = cls()
        if not Path(path).exists():
            return index
        with open(path, "rb") as f:
            state = pickle.load(f)
        index._bm25 = state["bm25"]
        index._ids = state["ids"]
        return index
