"""STUB — BM25 lexical baseline (Retriever port, NOT an Embedder).

TEXT ONLY. BM25 scores a query against product text; there is no vector space
and no image path, so it implements ``Retriever`` directly instead of plugging
into ``DenseRetriever``. Image queries must raise. It is baseline #1 in DeCuong
Mục 5.2 and matters more than it looks: Vietnamese shoppers type exact product
vocabulary ("áo bomber", "quần baggy"), which lexical matching nails and dense
retrieval sometimes misses.

Pattern to port: no reference repo ships BM25, so use the ``rank_bm25`` package
(``BM25Okapi``) and follow the query-loop shape in
``reference_repos/marqo-FashionCLIP/utils/retrieval.py`` (``run_retrieval``:
score all docs per query, take top-k, keep doc ids aligned).

Implementation notes for whoever picks this up:
* Import ``rank_bm25`` INSIDE ``__init__`` (never at module top).
* Index ``Product.to_text()`` so this and the dense systems see identical text.
* Tokenization decides everything here: NFC-normalize, lowercase, keep
  diacritics (DeCuong Mục 4.3). Vietnamese is space-separated by syllable, so
  consider a word-segmenter or bigrams before concluding BM25 is weak.
* Apply ``query.filters`` yourself — there is no store to delegate to.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.models import Product, Query, SearchResult

_TODO = (
    "TODO(Sprint 2, Hiếu): implement with rank_bm25.BM25Okapi, query loop shaped "
    "like reference_repos/marqo-FashionCLIP/utils/retrieval.py (run_retrieval)"
)


class BM25Retriever:
    """Okapi BM25 over product text. Text queries only."""

    def __init__(
        self,
        name: str = "bm25",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._name = name
        self.k1 = k1
        self.b = b
        self._products: list[Product] = []

    @property
    def name(self) -> str:
        return self._name

    def index(self, products: Sequence[Product]) -> None:
        raise NotImplementedError(_TODO)

    def search(self, query: Query) -> list[SearchResult]:
        raise NotImplementedError(_TODO)
