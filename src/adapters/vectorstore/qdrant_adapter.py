"""Qdrant vector store (VectorStore port) — HNSW approximate search.

Ported from ``reference_repos/Multimodal-Image-Search-Engine/encoding.ipynb``:
``create_collection`` with ``VectorParams(size=..., distance=Distance.COSINE)``,
batched ``upsert`` of ``PointStruct(id, vector, payload)``, then a query with
``with_payload=True``.

Design notes:
* ``qdrant_client`` is imported inside ``__init__`` so that importing this module
  (and therefore ``src.registry``) works without the ML extras installed.
* Credentials come from ``QDRANT_URL`` / ``QDRANT_API_KEY`` env vars only —
  never hardcoded, never logged. ``docker-compose.yml`` starts a local instance
  at http://localhost:6333 with no key.
* Qdrant point ids must be int or UUID, so the id is a UUID5 of ``product_id``:
  stable across processes, which is what makes ``upsert`` a real upsert. The
  original ``product_id`` and every other ``Product`` field live in the payload.
* Filter keys that are not ``Product`` fields are looked up under
  ``attributes.<key>``, matching ``InMemoryVectorStore`` semantics.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import TYPE_CHECKING, Any

from src.core.models import Matrix, Product, SearchResult, Vector

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the runtime import local
    from qdrant_client import QdrantClient

QDRANT_URL_ENV = "QDRANT_URL"
QDRANT_API_KEY_ENV = "QDRANT_API_KEY"

_DISTANCES = {
    "cosine": "Cosine",
    "dot": "Dot",
    "euclid": "Euclid",
    "euclidean": "Euclid",
    "manhattan": "Manhattan",
}

_PRODUCT_FIELDS = tuple(f.name for f in fields(Product))
_POINT_ID_NAMESPACE = uuid.UUID("6f9f2c1e-0d7a-5c8b-9f4e-3a1b2c3d4e5f")


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
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if distance.lower() not in _DISTANCES:
            raise ValueError(f"unsupported distance {distance!r}, expected one of {_DISTANCES}")

        from qdrant_client import QdrantClient as _QdrantClient
        from qdrant_client.http import models as rest

        self._dim = dim
        self.collection = collection
        self.distance = distance
        self.url = url or os.environ.get(QDRANT_URL_ENV, "http://localhost:6333")
        self.batch_size = batch_size
        self.timeout = timeout
        # Read from the env at construction time; do not log or echo this value.
        self._api_key = api_key or os.environ.get(QDRANT_API_KEY_ENV)
        self._rest = rest
        # check_compatibility=False keeps construction offline: the port allows
        # building a store (registry, conformance tests) with no server running.
        self._client: QdrantClient = _QdrantClient(
            url=self.url,
            api_key=self._api_key,
            timeout=int(self.timeout),
            check_compatibility=False,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def ensure_collection(self) -> None:
        """Create the collection if missing (idempotent)."""
        if self._client.collection_exists(collection_name=self.collection):
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=self._rest.VectorParams(
                size=self._dim,
                distance=self._rest.Distance(_DISTANCES[self.distance.lower()]),
            ),
        )

    def recreate_collection(self) -> None:
        """DESTRUCTIVE: drop the collection with all its points, then create it empty.

        Only ever called from ``python -m src.index --recreate``; nothing on the
        serving path may call this. Use it when the vectors already stored are no
        longer comparable to what the current embedder produces (different model
        or a different fine-tuned checkpoint).
        """
        if self._client.collection_exists(collection_name=self.collection):
            self._client.delete_collection(collection_name=self.collection)
        self.ensure_collection()

    def upsert(self, products: Sequence[Product], vectors: Matrix) -> None:
        if len(products) != len(vectors):
            raise ValueError(f"got {len(products)} products but {len(vectors)} vectors")
        if not products:
            return

        points = []
        for product, vector in zip(products, vectors, strict=True):
            if len(vector) != self._dim:
                raise ValueError(
                    f"vectors must have shape (n, {self._dim}), got a row of {len(vector)}"
                )
            points.append(
                self._rest.PointStruct(
                    id=_point_id(product.product_id),
                    vector=[float(value) for value in vector],
                    payload=_to_payload(product),
                )
            )

        for start in range(0, len(points), self.batch_size):
            self._client.upsert(
                collection_name=self.collection,
                points=points[start : start + self.batch_size],
                wait=True,
            )

    def search(
        self,
        vector: Vector,
        top_k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if len(vector) != self._dim:
            raise ValueError(f"query vector must have shape ({self._dim},), got ({len(vector)},)")

        response = self._client.query_points(
            collection_name=self.collection,
            query=[float(value) for value in vector],
            limit=top_k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        return [
            SearchResult(product=_from_payload(point.payload), score=float(point.score), rank=rank)
            for rank, point in enumerate(response.points, start=1)
        ]

    def count(self) -> int:
        return int(self._client.count(collection_name=self.collection, exact=True).count)

    def _build_filter(self, filters: Mapping[str, str] | None) -> Any:
        if not filters:
            return None
        # Typed as Any: rest.Filter.must is an invariant list of a big condition union.
        conditions: list[Any] = [
            self._rest.FieldCondition(
                key=key if key in _PRODUCT_FIELDS else f"attributes.{key}",
                match=self._rest.MatchValue(value=value),
            )
            for key, value in filters.items()
        ]
        return self._rest.Filter(must=conditions)


def _point_id(product_id: str) -> str:
    """Stable UUID for a catalog id, so re-upserting a product replaces its point."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, product_id))


def _to_payload(product: Product) -> dict[str, Any]:
    payload: dict[str, Any] = {name: getattr(product, name) for name in _PRODUCT_FIELDS}
    payload["attributes"] = dict(product.attributes)
    return payload


def _from_payload(payload: Mapping[str, Any] | None) -> Product:
    if payload is None:
        raise ValueError("Qdrant hit has no payload; the collection was written without one")
    return Product(
        product_id=str(payload["product_id"]),
        title=str(payload["title"]),
        category=str(payload["category"]),
        color=payload.get("color"),
        material=payload.get("material"),
        price_vnd=payload.get("price_vnd"),
        image_path=payload.get("image_path"),
        description=payload.get("description"),
        attributes=dict(payload.get("attributes") or {}),
    )
