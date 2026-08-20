"""FROZEN contracts between the core and every adapter.

Adding, removing or changing the signature of a method here invalidates every
teammate's adapter at once. Coordinate with the whole team before editing this
file (see README, section "Frozen ports").

Standard library only: ports must be importable with no ML extras installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from src.core.models import Matrix, Product, Query, RelevanceLabel, SearchResult, Vector

RunResults = Mapping[str, Sequence[SearchResult]]
"""Results of one system over the whole query set, keyed by ``query_id``."""


@runtime_checkable
class Embedder(Protocol):
    """Maps text and/or images into one shared vector space.

    Implementations must return L2-normalized vectors so that a dot product is a
    cosine similarity. Image-only backbones (e.g. ResNet) may raise
    ``NotImplementedError`` from :meth:`encode_text` — they must say so in their
    class docstring.
    """

    @property
    def name(self) -> str:
        """Short identifier used in configs, result tables and ``runs.csv``."""

    @property
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""

    def encode_text(self, texts: Sequence[str]) -> Matrix:
        """Embed a batch of texts, one row per input, in input order."""

    def encode_image(self, images: Sequence[bytes]) -> Matrix:
        """Embed a batch of raw image bytes, one row per input, in input order."""


@runtime_checkable
class VectorStore(Protocol):
    """Stores product vectors and answers nearest-neighbour queries."""

    @property
    def dim(self) -> int:
        """Dimensionality accepted by this store."""

    def upsert(self, products: Sequence[Product], vectors: Matrix) -> None:
        """Insert or replace ``products``; ``vectors[i]`` belongs to ``products[i]``."""

    def search(
        self,
        vector: Vector,
        top_k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        """Return the ``top_k`` closest hits, rank 1 first.

        ``filters`` is an exact-match payload filter (e.g. ``{"category": "áo"}``).
        """

    def count(self) -> int:
        """Number of indexed products."""


@runtime_checkable
class Retriever(Protocol):
    """A complete searchable system — this is what gets benchmarked.

    Dense (embedder + vector store), lexical (BM25) and hybrid systems all sit
    behind this one port so that ``evaluate.py`` can compare them blindly.
    """

    @property
    def name(self) -> str:
        """Short identifier used in result tables and ``runs.csv``."""

    def index(self, products: Sequence[Product]) -> None:
        """Build (or rebuild) the index over ``products``."""

    def search(self, query: Query) -> list[SearchResult]:
        """Answer one query, best hit first."""


@runtime_checkable
class MetricsPort(Protocol):
    """Ranking metrics over a run and the pooled 0/1/2 relevance labels."""

    def recall_at_k(self, run: RunResults, labels: Sequence[RelevanceLabel], k: int) -> float:
        """Share of queries with at least one relevant hit in the top ``k``."""

    def mrr(self, run: RunResults, labels: Sequence[RelevanceLabel]) -> float:
        """Mean reciprocal rank of the first relevant hit."""

    def ndcg_at_k(self, run: RunResults, labels: Sequence[RelevanceLabel], k: int) -> float:
        """Graded nDCG at ``k``, using the 0/1/2 grades as gains."""
