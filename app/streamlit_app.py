"""Streamlit demo UI (Sprint 5).

Layout reference only: ``reference_repos/Multimodal-Image-Search-Engine/app.py``
shows the shape we want (modality radio -> text box or image uploader, a results
count slider, a grid of result images). That file is Gradio and we ship
Streamlit per the proposal, so read it for the layout, do not port the code.

Talk to ``app/api.py`` over HTTP (``POST /search``, ``POST /search/image``) rather
than importing the retriever — the API already owns config loading and indexing,
and keeping one boundary means the latency numbers in the report describe the
same path users hit.

Feature set is kept in step with the inline HTML demo page in ``app/api.py``
(same query flow, same card fields) so the report describes one demo, not two.

Everything above ``main`` is plain data in / plain data out and is unit tested;
Streamlit is imported inside ``main`` only, so the tests never need it.

Run it (API must already be up on API_BASE_URL):

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote

import httpx

API_BASE_URL = "http://localhost:8000"
API_URL_ENV = "FASHION_SEARCH_API_URL"
REQUEST_TIMEOUT = 30.0
# Mirrors app.api.MAX_IMAGE_BYTES so an oversized upload is refused here instead
# of costing a round trip that can only come back as 413.
MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_TOP_K = 12
MAX_TOP_K = 50
MAX_QUERY_CHARS = 512

BannerLevel = Literal["ok", "warn", "error"]


@dataclass(frozen=True)
class ResultCard:
    """One search hit, flattened for rendering."""

    rank: int
    score: float
    title: str
    category: str
    product_id: str
    image_url: str | None


@dataclass(frozen=True)
class HealthBanner:
    """Sidebar system status: what to say and how loudly."""

    level: BannerLevel
    text: str


@dataclass(frozen=True)
class SearchOutcome:
    """Either the ranked cards or a Vietnamese error message, never both."""

    cards: list[ResultCard] = field(default_factory=list)
    retriever: str = ""
    error: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def api_base_url() -> str:
    """Configured API root, env var winning over the module default."""
    return (os.environ.get(API_URL_ENV) or API_BASE_URL).rstrip("/")


def auth_headers(api_key: str) -> dict[str, str]:
    """``X-API-Key`` header when a key was typed in, empty otherwise.

    The key only ever travels in a header — never in a URL, never in a log line.
    """
    key = api_key.strip()
    return {"X-API-Key": key} if key else {}


def build_search_payload(text: str, top_k: int, category: str = "") -> dict[str, Any]:
    """Body for ``POST /search``, matching ``app.api.SearchRequest``."""
    payload: dict[str, Any] = {"text": text.strip()[:MAX_QUERY_CHARS], "top_k": int(top_k)}
    if category.strip():
        payload["filters"] = {"category": category.strip()}
    return payload


def image_url(base_url: str, image_path: str | None) -> str | None:
    """Absolute URL for the API's ``/image/{path:path}`` endpoint.

    Quoted per segment: ``/`` must survive, but ``#``, ``?`` and ``%`` in a
    filename would otherwise truncate the URL or decode into something else.
    """
    if not image_path:
        return None
    segments = image_path.replace("\\", "/").split("/")
    quoted = "/".join(quote(segment, safe="") for segment in segments if segment)
    if not quoted:
        return None
    return f"{base_url.rstrip('/')}/image/{quoted}"


def result_cards(base_url: str, body: Any) -> list[ResultCard]:
    """Map a ``SearchResponse`` body into cards, skipping malformed hits."""
    if not isinstance(body, dict):
        return []
    hits = body.get("hits")
    if not isinstance(hits, list):
        return []
    cards: list[ResultCard] = []
    for index, hit in enumerate(hits, start=1):
        if not isinstance(hit, dict):
            continue
        product = hit.get("product")
        if not isinstance(product, dict):
            continue
        raw_path = product.get("image_path")
        cards.append(
            ResultCard(
                rank=int(hit.get("rank", index)),
                score=float(hit.get("score", 0.0)),
                title=str(product.get("title", "(không có tiêu đề)")),
                category=str(product.get("category", "")),
                product_id=str(product.get("product_id", "")),
                image_url=image_url(base_url, raw_path if isinstance(raw_path, str) else None),
            )
        )
    return cards


def detail_text(body: Any) -> str:
    """The server's ``detail`` as flat text.

    Returned as text on purpose: it is rendered with ``st.text``, so a detail
    containing markup is shown, never interpreted.
    """
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return str(detail)
    if isinstance(body, str):
        return body
    return ""


def error_message(status_code: int) -> str:
    """Vietnamese explanation for an API status code."""
    known = {
        401: "Sai hoặc thiếu API key — nhập lại key ở thanh bên.",
        403: "API từ chối yêu cầu (403).",
        404: "Không tìm thấy endpoint — kiểm tra lại địa chỉ API.",
        413: "Ảnh quá lớn, API chỉ nhận tối đa 8 MB.",
        422: "Dữ liệu truy vấn không hợp lệ (422).",
        501: "Cấu hình hiện tại chưa hỗ trợ truy vấn này (adapter chưa cài đặt).",
        503: "Vector store không phản hồi — kiểm tra Qdrant đã chạy chưa.",
    }
    if status_code in known:
        return known[status_code]
    if status_code >= 500:
        return f"API gặp lỗi nội bộ (HTTP {status_code})."
    return f"API trả về lỗi HTTP {status_code}."


CONNECTION_ERROR = "Không kết nối được tới API — kiểm tra `uvicorn app.api:app` đã chạy chưa."


def health_banner(payload: Any) -> HealthBanner:
    """Sidebar banner from a ``/health`` body, or ``None`` when unreachable."""
    if payload is None:
        return HealthBanner("error", "Không gọi được /health — API chưa chạy?")
    if not isinstance(payload, dict):
        return HealthBanner("error", "/health trả về dữ liệu không hợp lệ.")
    retriever = str(payload.get("retriever", "?"))
    count = payload.get("indexed_products", 0)
    status = str(payload.get("status", ""))
    summary = f"{retriever} · {count} sản phẩm đã index"
    if status == "ok":
        return HealthBanner("ok", f"Hệ thống: {summary}")
    if status == "degraded":
        return HealthBanner(
            "error",
            f"SUY GIẢM: {summary} — vector store không đếm được, tìm kiếm sẽ lỗi.",
        )
    return HealthBanner("warn", f"Trạng thái lạ '{status}': {summary}")


def open_client(base_url: str, timeout: float = REQUEST_TIMEOUT) -> httpx.Client:
    """HTTP client bound to the API root."""
    return httpx.Client(base_url=base_url, timeout=timeout)


def fetch_health(client: httpx.Client, api_key: str = "") -> HealthBanner:
    """Ask ``/health`` and turn the answer into a sidebar banner."""
    try:
        response = client.get("/health", headers=auth_headers(api_key))
    except httpx.RequestError:
        return health_banner(None)
    if response.status_code != 200:
        return HealthBanner("error", f"/health trả về HTTP {response.status_code}.")
    try:
        return health_banner(response.json())
    except ValueError:
        return health_banner("not json")


def _outcome_from_response(client: httpx.Client, response: httpx.Response) -> SearchOutcome:
    try:
        body: Any = response.json()
    except ValueError:
        body = None
    if response.status_code != 200:
        return SearchOutcome(error=error_message(response.status_code), detail=detail_text(body))
    base_url = str(client.base_url).rstrip("/")
    retriever = str(body.get("retriever", "")) if isinstance(body, dict) else ""
    return SearchOutcome(cards=result_cards(base_url, body), retriever=retriever)


def run_text_search(
    client: httpx.Client, text: str, top_k: int, api_key: str = "", category: str = ""
) -> SearchOutcome:
    """``POST /search`` with a text query."""
    if not text.strip():
        return SearchOutcome(error="Nhập câu truy vấn trước khi tìm.")
    try:
        response = client.post(
            "/search",
            json=build_search_payload(text, top_k, category),
            headers=auth_headers(api_key),
        )
    except httpx.RequestError:
        return SearchOutcome(error=CONNECTION_ERROR)
    return _outcome_from_response(client, response)


def run_image_search(
    client: httpx.Client,
    data: bytes,
    filename: str,
    top_k: int,
    api_key: str = "",
    category: str = "",
) -> SearchOutcome:
    """``POST /search/image`` with an uploaded image as the query."""
    if not data:
        return SearchOutcome(error="Chưa chọn ảnh truy vấn.")
    if len(data) > MAX_IMAGE_BYTES:
        return SearchOutcome(error=error_message(413))
    form: dict[str, str] = {"top_k": str(int(top_k))}
    if category.strip():
        form["category"] = category.strip()
    try:
        response = client.post(
            "/search/image",
            files={"image": (filename or "query.jpg", data)},
            data=form,
            headers=auth_headers(api_key),
        )
    except httpx.RequestError:
        return SearchOutcome(error=CONNECTION_ERROR)
    return _outcome_from_response(client, response)


def main() -> None:
    """Render the demo page. All Streamlit calls live here."""
    import streamlit as st

    st.set_page_config(page_title="Tìm kiếm thời trang đa mô thức", layout="wide")
    st.title("Tìm kiếm thời trang đa mô thức")

    with st.sidebar:
        st.header("Kết nối")
        base_url = st.text_input("Địa chỉ API", value=api_base_url()).rstrip("/")
        api_key = st.text_input(
            "API key (nếu server bật xác thực)",
            type="password",
            help="Chỉ được gửi trong header X-API-Key, không ghi ra log.",
        )
        top_k = st.slider("Số kết quả", min_value=1, max_value=MAX_TOP_K, value=DEFAULT_TOP_K)
        category = st.text_input("Lọc theo danh mục (tuỳ chọn)", value="")
        st.header("Tình trạng hệ thống")
        with open_client(base_url) as client:
            banner = fetch_health(client, api_key)
        if banner.level == "ok":
            st.success(banner.text)
        elif banner.level == "warn":
            st.warning(banner.text)
        else:
            st.error(banner.text)

    text_tab, image_tab = st.tabs(["Truy vấn văn bản", "Truy vấn bằng ảnh"])
    outcome: SearchOutcome | None = None

    with text_tab:
        with st.form("text_search"):
            text = st.text_input("Câu truy vấn", placeholder="vd: váy đầm dự tiệc màu đỏ")
            submitted = st.form_submit_button("Tìm")
        if submitted:
            with st.spinner("Đang tìm..."), open_client(base_url) as client:
                outcome = run_text_search(client, text, top_k, api_key, category)

    with image_tab:
        st.caption(
            "Tìm bằng ảnh dùng ảnh làm truy vấn duy nhất. Kết hợp đồng thời ảnh + văn bản "
            "chưa hỗ trợ, API sẽ trả 501."
        )
        upload = st.file_uploader("Ảnh truy vấn", type=["jpg", "jpeg", "png", "webp"])
        if st.button("Tìm bằng ảnh"):
            data = upload.getvalue() if upload is not None else b""
            filename = upload.name if upload is not None else ""
            with st.spinner("Đang tìm..."), open_client(base_url) as client:
                outcome = run_image_search(client, data, filename, top_k, api_key, category)

    if outcome is None:
        return
    if not outcome.ok:
        st.error(outcome.error)
        if outcome.detail:
            st.caption("Chi tiết từ server:")
            st.text(outcome.detail)
        return
    if not outcome.cards:
        st.warning("Không có kết quả nào.")
        return
    st.caption(f"{len(outcome.cards)} kết quả từ hệ thống: {outcome.retriever}")
    columns = st.columns(4)
    for index, card in enumerate(outcome.cards):
        with columns[index % len(columns)]:
            if card.image_url:
                st.image(card.image_url, use_container_width=True)
            # st.text, not st.markdown: catalog titles are data, not markup.
            st.text(card.title)
            st.caption(f"#{card.rank} · {card.category} · {card.score:.3f}")


if __name__ == "__main__":
    main()
