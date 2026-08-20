"""REST API for the multimodal fashion search demo.

Boots with zero external services: the default config wires the fake embedder to
the in-memory store, so ``uvicorn app.api:app --reload`` works on a laptop with
no GPU and no Qdrant. Point ``FASHION_SEARCH_CONFIG`` at another config file to
swap in a real backbone once Sprint 2 lands one.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from src.core.models import Product, Query, SearchResult
from src.core.ports import Retriever
from src.registry import build_retriever_from_config, load_config, products_from_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
API_KEY_ENV = "FASHION_SEARCH_API_KEY"
MAX_IMAGE_BYTES = 8 * 1024 * 1024


class ProductOut(BaseModel):
    """One catalog item as returned by the API."""

    product_id: str
    title: str
    category: str
    color: str | None = None
    material: str | None = None
    price_vnd: int | None = None
    image_path: str | None = None
    description: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, product: Product) -> ProductOut:
        return cls(
            product_id=product.product_id,
            title=product.title,
            category=product.category,
            color=product.color,
            material=product.material,
            price_vnd=product.price_vnd,
            image_path=product.image_path,
            description=product.description,
            attributes=dict(product.attributes),
        )


class HitOut(BaseModel):
    """A ranked search hit."""

    rank: int = Field(ge=1)
    score: float
    product: ProductOut

    @classmethod
    def from_domain(cls, result: SearchResult) -> HitOut:
        return cls(
            rank=result.rank,
            score=result.score,
            product=ProductOut.from_domain(result.product),
        )


class SearchRequest(BaseModel):
    """Text search request."""

    text: str = Field(min_length=1, max_length=512)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, str] = Field(
        default_factory=dict,
        description="Exact-match payload filters, e.g. {\"category\": \"áo khoác\"}",
    )


class SearchResponse(BaseModel):
    """Ranked results plus the system that produced them."""

    retriever: str
    query_id: str
    modality: str
    hits: list[HitOut]


class HealthResponse(BaseModel):
    """Liveness plus which retriever and how many products are loaded."""

    status: str
    retriever: str
    indexed_products: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build and index the retriever once at startup, never per request."""
    config_path = Path(os.environ.get("FASHION_SEARCH_CONFIG", str(DEFAULT_CONFIG_PATH)))
    config = load_config(config_path)
    retriever = build_retriever_from_config(config)
    products = products_from_config(config)
    retriever.index(products)
    app.state.retriever = retriever
    app.state.indexed_products = len(products)
    yield


app = FastAPI(
    title="Multimodal Fashion Product Search",
    description="Skeleton API for comparing interchangeable retrieval systems.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_retriever(request: Request) -> Retriever:
    """Inject the retriever built at startup."""
    retriever: Retriever = request.app.state.retriever
    return retriever


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Authenticate the caller against the ``FASHION_SEARCH_API_KEY`` env var.

    PUBLIC BY DEFAULT: when ``FASHION_SEARCH_API_KEY`` is unset this dependency
    is a deliberate no-op, because the demo is meant to run on localhost for an
    academic project. Setting the env var turns on a shared-secret check on the
    ``X-API-Key`` header — do that before exposing this service on any network.
    It is not a substitute for real per-user auth.
    """
    expected = os.environ.get(API_KEY_ENV)
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-API-Key header",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
AuthDep = Annotated[None, Depends(require_api_key)]


@app.get("/health", response_model=HealthResponse)
def health(retriever: RetrieverDep) -> HealthResponse:
    """Liveness probe. PUBLIC — no authentication, exposes no catalog data."""
    return HealthResponse(
        status="ok",
        retriever=retriever.name,
        indexed_products=app.state.indexed_products,
    )


@app.post("/search", response_model=SearchResponse)
def search_text(
    payload: SearchRequest,
    retriever: RetrieverDep,
    _: AuthDep,
) -> SearchResponse:
    """Text-to-image search over the indexed catalog.

    Auth: requires the ``X-API-Key`` header when ``FASHION_SEARCH_API_KEY`` is
    set in the environment; public otherwise (see ``require_api_key``).
    """
    query = Query(
        query_id=str(uuid.uuid4()),
        text=payload.text,
        top_k=payload.top_k,
        filters=dict(payload.filters),
    )
    return _run(retriever, query)


@app.post("/search/image", response_model=SearchResponse)
async def search_image(
    retriever: RetrieverDep,
    _: AuthDep,
    image: Annotated[UploadFile, File(description="Query image")],
    top_k: Annotated[int, Form(ge=1, le=100)] = 10,
    category: Annotated[str | None, Form()] = None,
) -> SearchResponse:
    """Image-to-image search over the indexed catalog.

    Auth: requires the ``X-API-Key`` header when ``FASHION_SEARCH_API_KEY`` is
    set in the environment; public otherwise (see ``require_api_key``).
    """
    data = await image.read()
    # Numeric codes: starlette renamed the 413/422 constants and both spellings
    # emit a DeprecationWarning on one version or the other.
    if not data:
        raise HTTPException(status_code=422, detail="empty image upload")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413, detail=f"image exceeds {MAX_IMAGE_BYTES} bytes"
        )
    filters: Mapping[str, str] = {"category": category} if category else {}
    query = Query(
        query_id=str(uuid.uuid4()),
        image_bytes=data,
        top_k=top_k,
        filters=dict(filters),
    )
    return _run(retriever, query)


def _run(retriever: Retriever, query: Query) -> SearchResponse:
    try:
        results = retriever.search(query)
    except NotImplementedError as exc:  # a stub adapter is wired in the config
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)
        ) from exc
    return SearchResponse(
        retriever=retriever.name,
        query_id=query.query_id,
        modality=query.modality,
        hits=[HitOut.from_domain(result) for result in results],
    )
