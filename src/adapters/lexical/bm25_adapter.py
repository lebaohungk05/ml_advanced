"""BM25 lexical baseline (Retriever port, NOT an Embedder).

TEXT ONLY. BM25 scores a query against product text; there is no vector space
and no image path, so it implements ``Retriever`` directly instead of plugging
into ``DenseRetriever``. Image queries must raise. It is baseline #1 in DeCuong
Mục 5.2 and matters more than it looks: Vietnamese shoppers type exact product
vocabulary ("áo bomber", "quần baggy"), which lexical matching nails and dense
retrieval sometimes misses.

Design notes:
* ``rank_bm25`` is imported inside ``__init__`` so the module stays importable
  without the ML extras installed.
* Documents are ``Product.to_text()`` so this and the dense systems index
  exactly the same surface form.
* Tokenization decides everything here: NFC-normalize, lowercase, keep
  diacritics (DeCuong Mục 4.3), split on whitespace. Vietnamese is
  space-separated by syllable, so a word-segmenter or bigrams is the next thing
  to try before concluding BM25 is weak.
* ``query.filters`` is applied here — there is no store to delegate to.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from src.core.models import Product, Query, SearchResult


class BM25Retriever:
    """Okapi BM25 over product text. Text queries only."""

    def __init__(
        self,
        name: str = "bm25",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        from rank_bm25 import BM25Okapi

        self._name = name
        self.k1 = k1
        self.b = b
        self._products: list[Product] = []
        self._bm25_cls = BM25Okapi
        self._bm25: Any | None = None

    @property
    def name(self) -> str:
        return self._name

    def index(self, products: Sequence[Product]) -> None:
        self._products = list(products)
        corpus = [_tokenize(product.to_text()) for product in self._products]
        # BM25Okapi divides by the corpus size, so an empty corpus cannot be built.
        self._bm25 = self._bm25_cls(corpus, k1=self.k1, b=self.b) if corpus else None

    def search(self, query: Query) -> list[SearchResult]:
        if query.has_image and not query.has_text:
            raise NotImplementedError(
                f"{self._name} is a lexical text index: it cannot score image-only queries"
            )
        if self._bm25 is None:
            return []

        tokens = _tokenize(query.normalized_text())
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        candidates = [
            (float(scores[i]), i)
            for i, product in enumerate(self._products)
            if _matches(product, query.filters)
        ]
        # Negated score keeps the sort stable on ties: original index order wins.
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            SearchResult(product=self._products[i], score=score, rank=rank)
            for rank, (score, i) in enumerate(candidates[: query.top_k], start=1)
        ]


def _tokenize(text: str) -> list[str]:
    """NFC-normalize, lowercase and split on whitespace, keeping diacritics."""
    return unicodedata.normalize("NFC", text).lower().split()


def _matches(product: Product, filters: Mapping[str, str]) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = getattr(product, key, None)
        if actual is None:
            actual = product.attributes.get(key)
        if actual is None or str(actual) != str(expected):
            return False
    return True
