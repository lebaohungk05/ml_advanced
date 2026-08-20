"""Domain objects shared by every adapter.

Standard library only on purpose: the core must stay importable without
numpy/torch/pydantic so that adapters, tests and the API can all depend on it.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

Vector = list[float]
Matrix = list[Vector]

Modality = Literal["text", "image", "multimodal"]

MAX_GRADE = 2
"""Highest relevance grade used by the annotation guideline (0 / 1 / 2)."""


@dataclass(frozen=True, slots=True)
class Product:
    """A catalog item as the retrieval systems see it."""

    product_id: str
    title: str
    category: str
    color: str | None = None
    material: str | None = None
    price_vnd: int | None = None
    image_path: str | None = None
    description: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.product_id.strip():
            raise ValueError("product_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if self.price_vnd is not None and self.price_vnd < 0:
            raise ValueError("price_vnd must not be negative")

    def to_text(self) -> str:
        """Flatten the item into the document text used by text-based indexes.

        Shared by the dense retriever and (later) BM25 so both index exactly the
        same surface form. Vietnamese diacritics are preserved deliberately —
        stripping them makes queries seriously ambiguous.
        """
        parts = [self.title, self.category, self.color, self.material, self.description]
        parts.extend(self.attributes.values())
        return " ".join(unicodedata.normalize("NFC", p) for p in parts if p)


@dataclass(frozen=True, slots=True)
class Query:
    """One search request: text, image, or both."""

    query_id: str
    text: str | None = None
    image_bytes: bytes | None = None
    image_path: str | None = None
    top_k: int = 10
    filters: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must not be empty")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not self.has_text and not self.has_image:
            raise ValueError("a query needs text or an image")

    @property
    def has_text(self) -> bool:
        return self.text is not None and bool(self.text.strip())

    @property
    def has_image(self) -> bool:
        return self.image_bytes is not None or self.image_path is not None

    @property
    def modality(self) -> Modality:
        if self.has_text and self.has_image:
            return "multimodal"
        return "text" if self.has_text else "image"

    def normalized_text(self) -> str:
        """NFC-normalized, lowercased query text (empty for image-only queries)."""
        if self.text is None:
            return ""
        return unicodedata.normalize("NFC", self.text).strip().lower()


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single ranked hit."""

    product: Product
    score: float
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be >= 1 (1-based)")

    @property
    def product_id(self) -> str:
        return self.product.product_id


@dataclass(frozen=True, slots=True)
class RelevanceLabel:
    """One annotator's judgement of (query, product) on the 0/1/2 scale."""

    query_id: str
    product_id: str
    grade: int
    annotator: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.grade <= MAX_GRADE:
            raise ValueError(f"grade must be within 0..{MAX_GRADE}, got {self.grade}")

    @property
    def is_relevant(self) -> bool:
        """Grades 1 and 2 both count as relevant for Recall/MRR (nDCG uses the grade)."""
        return self.grade > 0
