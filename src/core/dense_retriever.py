"""Composes an :class:`Embedder` and a :class:`VectorStore` into a Retriever."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.core.models import Product, Query, SearchResult
from src.core.ports import Embedder, VectorStore


class DenseRetriever:
    """Embed-then-search retrieval, agnostic to which backbone or store is used.

    Any Embedder (fake, CLIP, SigLIP 2, ViSigLIP-OT) can be paired with any
    VectorStore (in-memory, Qdrant) without touching this class.
    """

    def __init__(self, embedder: Embedder, store: VectorStore, name: str) -> None:
        if embedder.dim != store.dim:
            raise ValueError(
                f"embedder dim {embedder.dim} does not match store dim {store.dim}"
            )
        self._embedder = embedder
        self._store = store
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def store(self) -> VectorStore:
        return self._store

    def index(self, products: Sequence[Product]) -> None:
        if not products:
            return
        vectors = self._embedder.encode_text([p.to_text() for p in products])
        self._store.upsert(products, vectors)

    def search(self, query: Query) -> list[SearchResult]:
        vector = self._encode_query(query)
        return self._store.search(vector, query.top_k, dict(query.filters) or None)

    def _encode_query(self, query: Query) -> list[float]:
        if query.modality == "multimodal":
            raise NotImplementedError(
                "TODO(Sprint 4, Hùng): fusing a text vector with an image vector is an "
                "open design choice (weighted sum vs. re-ranking); send text-only or "
                "image-only queries for now"
            )
        if query.has_image:
            return self._embedder.encode_image([self._image_bytes(query)])[0]
        return self._embedder.encode_text([query.normalized_text()])[0]

    @staticmethod
    def _image_bytes(query: Query) -> bytes:
        if query.image_bytes is not None:
            return query.image_bytes
        assert query.image_path is not None  # guaranteed by Query.has_image
        return Path(query.image_path).read_bytes()
