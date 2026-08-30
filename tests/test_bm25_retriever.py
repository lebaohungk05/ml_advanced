"""BM25 lexical baseline: Vietnamese ranking, filters, edge cases."""

from __future__ import annotations

import pytest

from src.adapters.lexical.bm25_adapter import BM25Retriever
from src.core.models import Product, Query

_CATALOG = [
    Product(
        product_id="P1",
        title="Áo sơ mi trắng tay dài",
        category="áo sơ mi",
        color="trắng",
        material="cotton",
    ),
    Product(
        product_id="P2",
        title="Quần jean xanh ống rộng",
        category="quần",
        color="xanh",
        material="denim",
    ),
    Product(
        product_id="P3",
        title="Váy hoa nhí dáng dài",
        category="váy",
        color="hồng",
    ),
    Product(
        product_id="P4",
        title="Áo khoác bomber đen",
        category="áo khoác",
        color="đen",
    ),
    Product(
        product_id="P5",
        title="Áo thun trắng basic",
        category="áo thun",
        color="trắng",
    ),
]


def _retriever() -> BM25Retriever:
    retriever = BM25Retriever()
    retriever.index(_CATALOG)
    return retriever


def test_name_defaults_to_bm25() -> None:
    assert BM25Retriever().name == "bm25"


def test_best_lexical_match_ranks_first() -> None:
    hits = _retriever().search(Query(query_id="q1", text="áo sơ mi trắng", top_k=3))

    assert len(hits) == 3
    assert hits[0].product_id == "P1"
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert hits[0].score > hits[1].score


def test_diacritics_are_kept_so_quan_matches_the_jeans() -> None:
    hits = _retriever().search(Query(query_id="q2", text="quần jean xanh", top_k=1))
    assert hits[0].product_id == "P2"


def test_filters_exclude_every_non_matching_product() -> None:
    hits = _retriever().search(
        Query(query_id="q3", text="áo trắng", top_k=5, filters={"category": "áo thun"})
    )

    assert [hit.product_id for hit in hits] == ["P5"]


def test_filter_with_no_match_returns_nothing() -> None:
    hits = _retriever().search(
        Query(query_id="q4", text="áo trắng", top_k=5, filters={"category": "không có"})
    )
    assert hits == []


def test_search_without_indexing_returns_nothing() -> None:
    assert BM25Retriever().search(Query(query_id="q5", text="áo sơ mi")) == []


def test_index_can_be_rebuilt_with_an_empty_catalog() -> None:
    retriever = _retriever()
    retriever.index([])
    assert retriever.search(Query(query_id="q6", text="áo sơ mi")) == []


def test_image_only_query_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="lexical text index"):
        _retriever().search(Query(query_id="q7", image_path="anh.jpg"))
