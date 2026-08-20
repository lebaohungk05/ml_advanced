"""In-memory store: exact cosine ranking, filters, upsert semantics."""

from __future__ import annotations

import pytest

from src.adapters.vectorstore.inmemory_adapter import InMemoryVectorStore
from src.core.models import Product


def _product(product_id: str, category: str = "áo", color: str = "đen") -> Product:
    return Product(
        product_id=product_id,
        title=f"Sản phẩm {product_id}",
        category=category,
        color=color,
        attributes={"style": "basic"},
    )


def test_search_ranks_by_cosine_similarity() -> None:
    store = InMemoryVectorStore(dim=2)
    store.upsert([_product("P1"), _product("P2")], [[1.0, 0.0], [0.0, 1.0]])

    hits = store.search([1.0, 0.0], top_k=2)

    assert [hit.product_id for hit in hits] == ["P1", "P2"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.0)


def test_search_respects_top_k() -> None:
    store = InMemoryVectorStore(dim=2)
    store.upsert([_product("P1"), _product("P2")], [[1.0, 0.0], [0.9, 0.1]])
    assert len(store.search([1.0, 0.0], top_k=1)) == 1


def test_payload_filter_matches_fields_and_attributes() -> None:
    store = InMemoryVectorStore(dim=2)
    store.upsert(
        [_product("P1", category="áo khoác"), _product("P2", category="váy")],
        [[1.0, 0.0], [1.0, 0.0]],
    )

    hits = store.search([1.0, 0.0], top_k=5, filters={"category": "váy"})
    assert [hit.product_id for hit in hits] == ["P2"]

    assert store.search([1.0, 0.0], top_k=5, filters={"style": "basic"})
    assert store.search([1.0, 0.0], top_k=5, filters={"category": "không có"}) == []


def test_upsert_replaces_existing_product_without_growing() -> None:
    store = InMemoryVectorStore(dim=2)
    store.upsert([_product("P1")], [[1.0, 0.0]])
    store.upsert([_product("P1", color="đỏ")], [[0.0, 1.0]])

    assert store.count() == 1
    hits = store.search([0.0, 1.0], top_k=1)
    assert hits[0].product.color == "đỏ"
    assert hits[0].score == pytest.approx(1.0)


def test_search_on_empty_store_returns_nothing() -> None:
    assert InMemoryVectorStore(dim=2).search([1.0, 0.0], top_k=5) == []


def test_rejects_mismatched_inputs() -> None:
    store = InMemoryVectorStore(dim=2)
    with pytest.raises(ValueError, match="but 0 vectors"):
        store.upsert([_product("P1")], [])
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        store.upsert([_product("P1")], [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        store.search([1.0], top_k=1)
