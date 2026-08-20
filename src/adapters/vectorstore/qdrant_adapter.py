"""STUB — Qdrant vector store (VectorStore port).

Pattern to port: ``reference_repos/Multimodal-Image-Search-Engine/encoding.ipynb``
— it has the whole flow in a handful of cells: ``create_collection`` with
``rest.VectorParams(size=..., distance=rest.Distance.COSINE)``, batched
``upsert`` of ``rest.PointStruct(id, vector, payload)``, then ``search`` with
``with_payload=True``. ``reference_repos/Multimodal-Image-Search-Engine/app.py``
shows the query side against a hosted cluster.

Implementation notes for whoever picks this up:
* Import ``qdrant_client`` INSIDE ``__init__`` (never at module top).
* Credentials come from ``QDRANT_URL`` / ``QDRANT_API_KEY`` env vars only —
  never hardcode a URL or key, and never commit one. ``docker-compose.yml``
  starts a local instance at http://localhost:6333 with no key.
* Qdrant point ids must be int or UUID, so keep ``product_id`` in the payload and
  map it (a stable hash or an explicit id table) — the port hands back Products.
* Payload filters must translate to ``rest.Filter`` / ``FieldCondition``; an
  exact-match dict is all the port promises.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from src.core.models import Matrix, Product, SearchResult, Vector

_TODO = (
    "TODO(Sprint 2, Hiệp): port from reference_repos/Multimodal-Image-Search-Engine/"
    "encoding.ipynb (create_collection / upsert PointStruct / search)"
)

QDRANT_URL_ENV = "QDRANT_URL"
QDRANT_API_KEY_ENV = "QDRANT_API_KEY"


class QdrantVectorStore:
    """Qdrant-backed VectorStore with HNSW approximate search."""

    def __init__(
        self,
        dim: int,
        collection: str = "fashion_products",
        distance: str = "cosine",
        url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 128,
        timeout: float = 30.0,
    ) -> None:
        self._dim = dim
        self.collection = collection
        self.distance = distance
        self.url = url or os.environ.get(QDRANT_URL_ENV, "http://localhost:6333")
        self.batch_size = batch_size
        self.timeout = timeout
        # Read from the env at construction time; do not log or echo this value.
        self._api_key = api_key or os.environ.get(QDRANT_API_KEY_ENV)
        # Client construction belongs here, behind a local import.

    @property
    def dim(self) -> int:
        return self._dim

    def ensure_collection(self) -> None:
        """Create the collection if missing (idempotent)."""
        raise NotImplementedError(_TODO)

    def upsert(self, products: Sequence[Product], vectors: Matrix) -> None:
        raise NotImplementedError(_TODO)

    def search(
        self,
        vector: Vector,
        top_k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        raise NotImplementedError(_TODO)

    def count(self) -> int:
        raise NotImplementedError(_TODO)
