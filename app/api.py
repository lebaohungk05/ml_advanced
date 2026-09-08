"""REST API for the multimodal fashion search demo.

Boots with zero external services: the default config wires the fake embedder to
the in-memory store, so ``uvicorn app.api:app --reload`` works on a laptop with
no GPU and no Qdrant. Point ``FASHION_SEARCH_CONFIG`` at another config file to
swap in a real backbone once Sprint 2 lands one.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.core.models import Product, Query, SearchResult
from src.core.ports import Retriever
from src.registry import build_retriever_from_config, load_config, products_from_config

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
API_KEY_ENV = "FASHION_SEARCH_API_KEY"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})

# Base classes of "the server was not reachable", by qualified name — see
# _is_store_unavailable for why these are strings and not imports. Their
# subclasses (httpx.ConnectError, httpx.ReadTimeout, ...) match via the MRO.
_TRANSPORT_ERRORS = frozenset(
    {
        "httpx.TransportError",
        "httpcore.NetworkError",
        "httpcore.TimeoutException",
    }
)


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
            # Normalised here, once: the prepared catalog stores Windows-style
            # paths, and the browser needs a URL path for /image/{path}.
            image_path=product.image_path.replace("\\", "/") if product.image_path else None,
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
    """Build the retriever once at startup, never per request.

    The inline ``products:`` block is only indexed when the config actually has
    one. A config backed by a persistent store (``demo_siglip2_qdrant.yaml``) has
    none: its catalog was written once by ``python -m src.index`` and re-indexing
    on every boot would be wasted GPU work at best, a wiped collection at worst.
    """
    config_path = Path(os.environ.get("FASHION_SEARCH_CONFIG", str(DEFAULT_CONFIG_PATH)))
    config = load_config(config_path)
    retriever = build_retriever_from_config(config)
    products = products_from_config(config)
    if products:
        retriever.index(products)
    demo_cfg = config.get("demo") or {}
    images_root = Path(str(demo_cfg.get("images_root", "data/raw/fashionpedia")))
    if not images_root.is_absolute():
        images_root = REPO_ROOT / images_root
    app.state.retriever = retriever
    app.state.images_root = images_root
    yield


def _indexed_count(retriever: Retriever) -> int | None:
    """How many products the store holds right now, or ``None`` if it cannot say.

    Asked fresh on every ``/health`` call rather than frozen at startup, so a
    Qdrant that died mid-demo shows up immediately instead of reporting a stale
    number from boot time.

    A retriever with no countable store (``BM25Retriever`` holds its index in
    itself) reports ``0``: there is nothing to count, which is healthy. ``None``
    is reserved for a store that exists but refused to answer.
    """
    count = getattr(getattr(retriever, "store", None), "count", None)
    if count is None:
        return 0
    try:
        return int(count())
    except Exception:  # store unreachable — reported as "degraded", not a 500
        return None


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
    """Liveness probe. PUBLIC — no authentication, exposes no catalog data.

    ``degraded`` means the vector store did not answer a count: the API is up but
    search will fail until it comes back.
    """
    count = _indexed_count(retriever)
    return HealthResponse(
        status="ok" if count is not None else "degraded",
        retriever=retriever.name,
        indexed_products=count if count is not None else 0,
    )


@app.get("/image/{path:path}")
def catalog_image(request: Request, path: str) -> FileResponse:
    """Serve one catalog image by its relative path.

    PUBLIC endpoint, intentionally no auth: the demo page is a plain ``<img>``
    grid and browsers cannot attach the ``X-API-Key`` header to image requests.
    It only ever exposes files already destined for the public demo.

    Everything outside ``images_root`` is rejected, as is anything that is not a
    known image extension, so this cannot be turned into a file-read primitive.
    """
    images_root: Path = request.app.state.images_root
    relative = path.replace("\\", "/")
    if ".." in relative or relative.startswith("/") or "\x00" in relative:
        raise HTTPException(status_code=404, detail="image not found")
    try:
        full = (images_root / relative).resolve()
        inside_root = full.is_relative_to(images_root.resolve())
        servable = full.suffix.lower() in IMAGE_SUFFIXES and full.is_file()
    except (ValueError, OSError) as exc:
        # A path the filesystem refuses to even stat (NUL byte, illegal name,
        # too long) is a miss, not a 500 on a public route.
        raise HTTPException(status_code=404, detail="image not found") from exc
    if not inside_root or not servable:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(full)


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    """Single-page search demo. PUBLIC — no auth; it ships no data of its own.

    The page asks for an API key only to forward it to ``/search``, which is the
    endpoint that actually enforces ``FASHION_SEARCH_API_KEY``.
    """
    return _DEMO_PAGE


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


# PUBLIC like /image: static assets only, the key is forwarded to /search client-side.
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


def _causes(exc: BaseException) -> Iterator[BaseException]:
    """Walk an exception and everything it explicitly wraps, without cycles.

    ``qdrant_client`` hides the real failure inside
    ``ResponseHandlingException.source``, so the chain has to be followed rather
    than just inspecting the outermost type. ``__context__`` is deliberately not
    followed: a failure that merely happened while another was being handled is
    not evidence about this one.
    """
    seen: set[int] = set()
    queue: list[BaseException] = [exc]
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        wrapped = (getattr(current, "source", None), current.__cause__)
        queue.extend(item for item in wrapped if isinstance(item, BaseException))


def _is_store_unavailable(exc: BaseException) -> bool:
    """True only for transport-level failures: the store could not be reached.

    Deliberately narrow. A ``qdrant_client`` error is *not* enough on its own —
    a missing collection or a rejected API key also surfaces as one, and calling
    those "store unavailable — start qdrant.exe" would send the operator chasing
    the wrong problem.

    Matched by qualified class name rather than by importing the exception types,
    which would drag ``qdrant_client`` into every import of this module — the
    zero-infrastructure default config must stay installable without it.
    """
    for current in _causes(exc):
        if isinstance(current, ConnectionError | TimeoutError):
            return True
        names = {f"{klass.__module__}.{klass.__qualname__}" for klass in type(current).__mro__}
        if names & _TRANSPORT_ERRORS:
            return True
    return False


def _run(retriever: Retriever, query: Query) -> SearchResponse:
    try:
        results = retriever.search(query)
    except NotImplementedError as exc:  # a stub adapter is wired in the config
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)
        ) from exc
    except Exception as exc:
        # Detail is a fixed string on purpose: the underlying message can carry
        # the store URL and, with a managed Qdrant, the API key.
        if not _is_store_unavailable(exc):
            raise
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="vector store unavailable — start qdrant.exe",
        ) from exc
    return SearchResponse(
        retriever=retriever.name,
        query_id=query.query_id,
        modality=query.modality,
        hits=[HitOut.from_domain(result) for result in results],
    )


# Self-contained on purpose: no build step, no CDN, no framework. Every piece of
# server-returned text is written with .textContent, never innerHTML.
_DEMO_PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tìm kiếm thời trang đa mô thức</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 24px auto;
         padding: 0 16px; background:#111; color:#eee; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  #status { color:#999; font-size:14px; margin-bottom:16px; }
  #status.bad { color:#ff8080; }
  form { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
  #q { flex:1; min-width:260px; padding:10px; font-size:16px; }
  #k, #key { padding:10px; font-size:15px; }
  #k { width:90px; }
  #key { width:170px; }
  button { padding:10px 20px; font-size:16px; cursor:pointer; }
  #error { display:none; background:#7a2020; color:#fff; padding:10px 14px;
           border-radius:6px; margin-top:14px; }
  #grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(190px, 1fr));
          gap:14px; margin-top:20px; }
  .card { background:#1c1c1c; border-radius:8px; overflow:hidden; }
  .card img { width:100%; aspect-ratio:3/4; object-fit:cover; display:block; background:#333; }
  .meta { padding:8px 10px; font-size:13px; }
  .title { font-weight:600; margin-bottom:4px; overflow-wrap:anywhere; }
  .sub { color:#9a9a9a; }
</style>
</head>
<body>
  <h1>Tìm kiếm thời trang đa mô thức</h1>
  <div id="status">đang kiểm tra hệ thống...</div>
  <form id="form">
    <input id="q" placeholder="vd: váy đầm dự tiệc màu đỏ" autofocus>
    <input id="k" type="number" min="1" max="100" value="12" title="số kết quả">
    <input id="key" type="password" placeholder="API key (nếu có)">
    <button type="submit">Tìm</button>
  </form>
  <div id="error"></div>
  <div id="grid"></div>

<script>
const grid = document.getElementById("grid");
const errorBox = document.getElementById("error");
const statusBox = document.getElementById("status");
const keyInput = document.getElementById("key");
// sessionStorage, not localStorage: the key dies with the tab.
keyInput.value = sessionStorage.getItem("fashion_api_key") || "";

function showError(text) {
  errorBox.textContent = text;
  errorBox.style.display = "block";
}

async function refreshStatus() {
  try {
    const res = await fetch("/health");
    const h = await res.json();
    statusBox.textContent =
      `hệ thống: ${h.retriever} · ${h.indexed_products} sản phẩm đã index · ${h.status}`;
    statusBox.className = h.status === "ok" ? "" : "bad";
  } catch (e) {
    statusBox.textContent = "không gọi được /health";
    statusBox.className = "bad";
  }
}

function card(hit) {
  const el = document.createElement("div");
  el.className = "card";
  const p = hit.product;
  if (p.image_path) {
    const img = document.createElement("img");
    // Per segment: '/' must survive, but '#', '?' and '%' in a filename would
    // otherwise cut the URL short or decode into something else.
    img.src = "/image/" + p.image_path.replace(/\\\\/g, "/").split("/")
      .map((segment) => encodeURIComponent(segment)).join("/");
    img.loading = "lazy";
    img.alt = "";
    el.appendChild(img);
  }
  const meta = document.createElement("div");
  meta.className = "meta";
  const title = document.createElement("div");
  title.className = "title";
  title.textContent = p.title;
  const sub = document.createElement("div");
  sub.className = "sub";
  sub.textContent = `#${hit.rank} · ${p.category} · ${hit.score.toFixed(3)}`;
  meta.appendChild(title);
  meta.appendChild(sub);
  el.appendChild(meta);
  return el;
}

document.getElementById("form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = document.getElementById("q").value.trim();
  if (!text) return;
  errorBox.style.display = "none";
  grid.replaceChildren();
  const apiKey = keyInput.value.trim();
  sessionStorage.setItem("fashion_api_key", apiKey);
  const headers = {"Content-Type": "application/json"};
  if (apiKey) headers["X-API-Key"] = apiKey;

  let res;
  try {
    res = await fetch("/search", {
      method: "POST",
      headers: headers,
      body: JSON.stringify({text: text, top_k: Number(document.getElementById("k").value)}),
    });
  } catch (e) {
    showError("không kết nối được tới API");
    return;
  }
  if (!res.ok) {
    let detail = `lỗi ${res.status}`;
    try {
      const body = await res.json();
      if (body && body.detail) detail = `${res.status}: ${JSON.stringify(body.detail)}`;
    } catch (e) { /* body không phải JSON */ }
    showError(detail);
    refreshStatus();
    return;
  }
  const data = await res.json();
  if (!data.hits.length) showError("không có kết quả nào");
  for (const hit of data.hits) grid.appendChild(card(hit));
  refreshStatus();
});

refreshStatus();
</script>
</body>
</html>
"""
