"""Live Qdrant integration: collection lifecycle, upsert, ranking, payload filters.

Needs a running Qdrant (``docker compose up -d qdrant``), hence the ``qdrant``
marker — the default ``addopts`` deselects it.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

from src.adapters.vectorstore.qdrant_adapter import QdrantVectorStore
from src.core.models import Product

pytestmark = pytest.mark.qdrant


def _product(product_id: str, category: str) -> Product:
    return Product(
        product_id=product_id,
        title=f"Sản phẩm {product_id}",
        category=category,
        color="đen",
        attributes={"style": "basic"},
    )


PRODUCTS = [
    _product("P1", "áo khoác"),
    _product("P2", "váy"),
    _product("P3", "váy"),
]
VECTORS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
]
QUERY = [0.9, 0.1, 0.0, 0.0]  # clearly closest to P1


@pytest.fixture
def store() -> Iterator[QdrantVectorStore]:
    subject = QdrantVectorStore(dim=4, collection=f"test_smoke_{uuid.uuid4().hex[:8]}")
    subject.ensure_collection()
    subject.ensure_collection()  # idempotent
    try:
        subject.upsert(PRODUCTS, VECTORS)
        yield subject
    finally:
        subject._client.delete_collection(collection_name=subject.collection)


def test_search_ranks_nearest_product_first(store: QdrantVectorStore) -> None:
    hits = store.search(QUERY, top_k=2)

    assert len(hits) == 2
    assert hits[0].rank == 1
    assert hits[0].product_id == "P1"
    assert hits[0].product.title == "Sản phẩm P1"
    assert hits[0].product.attributes == {"style": "basic"}
    assert hits[1].rank == 2
    assert hits[0].score > hits[1].score


def test_count_reports_every_upserted_product(store: QdrantVectorStore) -> None:
    assert store.count() == 3


def test_category_filter_excludes_non_matching_products(store: QdrantVectorStore) -> None:
    hits = store.search(QUERY, top_k=5, filters={"category": "váy"})

    assert {hit.product_id for hit in hits} == {"P2", "P3"}
    assert store.search(QUERY, top_k=5, filters={"category": "không có"}) == []


def test_upsert_is_idempotent_on_the_same_product_id(store: QdrantVectorStore) -> None:
    store.upsert([_product("P1", "áo khoác")], [VECTORS[0]])
    assert store.count() == 3


def test_recreate_collection_drops_every_existing_point(store: QdrantVectorStore) -> None:
    store.recreate_collection()

    assert store.count() == 0
    store.upsert(PRODUCTS, VECTORS)
    assert store.count() == 3


def test_recreate_collection_creates_a_missing_collection(store: QdrantVectorStore) -> None:
    store._client.delete_collection(collection_name=store.collection)

    store.recreate_collection()

    assert store.count() == 0
