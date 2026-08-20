"""Brute-force numpy vector store — the zero-infrastructure default.

Exact cosine similarity over the whole catalog. Fine for the ~20-product demo
and for unit tests; Qdrant + HNSW takes over at dataset scale (see
``qdrant_adapter.py``). Because it is exact, it also doubles as the ground truth
when checking how much recall the approximate index gives up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.core.models import Matrix, Product, SearchResult, Vector


class InMemoryVectorStore:
    """In-process VectorStore with exact cosine search and payload filters."""

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self._dim = dim
        self._products: list[Product] = []
        self._index_by_id: dict[str, int] = {}
        self._vectors: NDArray[np.float32] = np.zeros((0, dim), dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    def count(self) -> int:
        return len(self._products)

    def upsert(self, products: Sequence[Product], vectors: Matrix) -> None:
        if len(products) != len(vectors):
            raise ValueError(f"got {len(products)} products but {len(vectors)} vectors")
        if not products:
            return
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._dim:
            raise ValueError(f"vectors must have shape (n, {self._dim}), got {matrix.shape}")

        rows: list[NDArray[np.float32]] = [self._vectors]
        appended: list[NDArray[np.float32]] = []
        for product, row in zip(products, matrix, strict=True):
            existing = self._index_by_id.get(product.product_id)
            if existing is None:
                self._index_by_id[product.product_id] = len(self._products)
                self._products.append(product)
                appended.append(row)
            else:
                self._products[existing] = product
                self._vectors[existing] = row
        if appended:
            rows.append(np.asarray(appended, dtype=np.float32))
            self._vectors = np.vstack(rows)

    def search(
        self,
        vector: Vector,
        top_k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query = np.asarray(vector, dtype=np.float32)
        if query.shape != (self._dim,):
            raise ValueError(f"query vector must have shape ({self._dim},), got {query.shape}")
        if not self._products:
            return []

        candidates = [i for i, p in enumerate(self._products) if _matches(p, filters)]
        if not candidates:
            return []

        scores = _cosine_similarity(self._vectors[candidates], query)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            SearchResult(
                product=self._products[candidates[int(pos)]],
                score=float(scores[int(pos)]),
                rank=rank,
            )
            for rank, pos in enumerate(order, start=1)
        ]


def _cosine_similarity(
    matrix: NDArray[np.float32], query: NDArray[np.float32]
) -> NDArray[np.float32]:
    # Embedders are contracted to return L2-normalized vectors, but renormalizing
    # here keeps the store honest if an adapter ever breaks that promise.
    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = float(np.linalg.norm(query))
    denominator = matrix_norms * query_norm
    dots = matrix @ query
    scores: NDArray[np.float32] = np.divide(
        dots, denominator, out=np.zeros_like(dots), where=denominator > 0
    ).astype(np.float32)
    return scores


def _matches(product: Product, filters: Mapping[str, str] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = getattr(product, key, None)
        if actual is None:
            actual = product.attributes.get(key)
        if actual is None or str(actual).casefold() != str(expected).casefold():
            return False
    return True
