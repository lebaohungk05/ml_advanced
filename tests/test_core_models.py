"""Core domain invariants — these guard every adapter downstream."""

from __future__ import annotations

import pytest

from src.core.models import Product, Query, RelevanceLabel, SearchResult


def test_query_requires_text_or_image() -> None:
    with pytest.raises(ValueError, match="needs text or an image"):
        Query(query_id="q1")


def test_query_rejects_blank_text_only() -> None:
    with pytest.raises(ValueError, match="needs text or an image"):
        Query(query_id="q1", text="   ")


def test_query_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        Query(query_id="q1", text="áo khoác", top_k=0)


def test_query_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="query_id must not be empty"):
        Query(query_id=" ", text="áo khoác")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"text": "váy hai dây"}, "text"),
        ({"image_bytes": b"jpeg"}, "image"),
        ({"image_path": "a.jpg"}, "image"),
        ({"text": "váy", "image_bytes": b"jpeg"}, "multimodal"),
    ],
)
def test_query_modality(kwargs: dict[str, object], expected: str) -> None:
    assert Query(query_id="q1", **kwargs).modality == expected  # type: ignore[arg-type]


def test_query_normalizes_text_but_keeps_diacritics() -> None:
    query = Query(query_id="q1", text="  Áo Khoác Bomber  ")
    assert query.normalized_text() == "áo khoác bomber"


def test_product_to_text_includes_attributes_and_skips_none() -> None:
    product = Product(
        product_id="P1",
        title="Áo khoác bomber nam màu be",
        category="áo khoác",
        color="be",
        attributes={"style": "streetwear"},
    )
    text = product.to_text()
    assert "bomber" in text
    assert "streetwear" in text
    assert "None" not in text


def test_product_rejects_empty_title() -> None:
    with pytest.raises(ValueError, match="title must not be empty"):
        Product(product_id="P1", title="", category="áo")


def test_search_result_rank_is_one_based() -> None:
    product = Product(product_id="P1", title="Áo thun", category="áo")
    assert SearchResult(product=product, score=0.5, rank=1).product_id == "P1"
    with pytest.raises(ValueError, match="rank must be >= 1"):
        SearchResult(product=product, score=0.5, rank=0)


@pytest.mark.parametrize("grade", [0, 1, 2])
def test_relevance_label_accepts_valid_grades(grade: int) -> None:
    label = RelevanceLabel(query_id="q1", product_id="P1", grade=grade)
    assert label.is_relevant is (grade > 0)


@pytest.mark.parametrize("grade", [-1, 3])
def test_relevance_label_rejects_out_of_range_grades(grade: int) -> None:
    with pytest.raises(ValueError, match=r"grade must be within 0\.\.2"):
        RelevanceLabel(query_id="q1", product_id="P1", grade=grade)
