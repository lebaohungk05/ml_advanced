"""Unit tests for the Streamlit demo's HTTP/data layer.

Only the pure functions and the httpx calls are covered — the rendering lives in
``main()`` and needs a running Streamlit script runner, which is out of scope
here. ``app.streamlit_app`` imports Streamlit inside ``main()``, so importing
this module never requires the package.
"""

from __future__ import annotations

import httpx
import pytest

from app.streamlit_app import (
    API_BASE_URL,
    API_URL_ENV,
    CONNECTION_ERROR,
    MAX_IMAGE_BYTES,
    api_base_url,
    auth_headers,
    build_search_payload,
    detail_text,
    error_message,
    fetch_health,
    health_banner,
    image_url,
    result_cards,
    run_image_search,
    run_text_search,
)

BASE = "http://api.test"


def _client(handler: object) -> httpx.Client:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return httpx.Client(base_url=BASE, transport=transport)


def _search_body() -> dict[str, object]:
    return {
        "retriever": "fake-inmemory",
        "query_id": "q1",
        "modality": "text",
        "hits": [
            {
                "rank": 1,
                "score": 0.9123,
                "product": {
                    "product_id": "p1",
                    "title": "Váy đầm dự tiệc",
                    "category": "váy",
                    "image_path": "train/a b#1.jpg",
                },
            }
        ],
    }


def test_api_base_url_defaults_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_URL_ENV, raising=False)
    assert api_base_url() == API_BASE_URL

    monkeypatch.setenv(API_URL_ENV, "http://elsewhere:9000/")
    assert api_base_url() == "http://elsewhere:9000"


def test_auth_headers_omitted_when_no_key() -> None:
    assert auth_headers("  ") == {}
    assert auth_headers(" secret ") == {"X-API-Key": "secret"}


def test_build_search_payload_matches_the_api_schema() -> None:
    assert build_search_payload("  áo khoác  ", 5) == {"text": "áo khoác", "top_k": 5}
    assert build_search_payload("áo", 3, "áo khoác")["filters"] == {"category": "áo khoác"}


def test_build_search_payload_truncates_to_the_api_max_length() -> None:
    payload = build_search_payload("a" * 600, 10)

    assert len(str(payload["text"])) == 512


def test_image_url_quotes_each_segment_but_keeps_the_separators() -> None:
    assert image_url(BASE, "train\\a b#1.jpg") == f"{BASE}/image/train/a%20b%231.jpg"
    assert image_url(BASE, None) is None
    assert image_url(BASE, "") is None


def test_result_cards_flattens_hits() -> None:
    cards = result_cards(BASE, _search_body())

    assert len(cards) == 1
    card = cards[0]
    assert (card.rank, card.title, card.category) == (1, "Váy đầm dự tiệc", "váy")
    assert card.score == pytest.approx(0.9123)
    assert card.image_url == f"{BASE}/image/train/a%20b%231.jpg"


def test_result_cards_tolerates_garbage_bodies() -> None:
    assert result_cards(BASE, None) == []
    assert result_cards(BASE, {"hits": "nope"}) == []
    assert result_cards(BASE, {"hits": [{"product": None}, 7]}) == []


def test_result_cards_handles_a_product_without_an_image() -> None:
    body = {"hits": [{"rank": 2, "score": 0.1, "product": {"product_id": "p", "title": "t"}}]}

    assert result_cards(BASE, body)[0].image_url is None


def test_error_message_is_specific_for_the_codes_the_api_can_return() -> None:
    assert "API key" in error_message(401)
    assert "chưa hỗ trợ" in error_message(501)
    assert "Qdrant" in error_message(503)
    assert "8 MB" in error_message(413)
    assert "418" in error_message(418)
    assert "500" in error_message(500)


def test_detail_text_returns_flat_text() -> None:
    assert detail_text({"detail": "<b>boom</b>"}) == "<b>boom</b>"
    assert detail_text({"detail": [{"loc": ["body"]}]}).startswith("[")
    assert detail_text(None) == ""


def test_health_banner_states() -> None:
    ok = health_banner({"status": "ok", "retriever": "fake-inmemory", "indexed_products": 20})
    assert ok.level == "ok"
    assert "fake-inmemory" in ok.text and "20" in ok.text

    degraded = health_banner({"status": "degraded", "retriever": "siglip2", "indexed_products": 0})
    assert degraded.level == "error"
    assert "SUY GIẢM" in degraded.text

    assert health_banner(None).level == "error"
    assert health_banner("nonsense").level == "error"
    assert health_banner({"status": "weird", "retriever": "r"}).level == "warn"


def test_fetch_health_reports_a_dead_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with _client(handler) as client:
        assert fetch_health(client).level == "error"


def test_fetch_health_sends_the_key_and_reads_the_payload() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-api-key", "")
        return httpx.Response(
            200, json={"status": "ok", "retriever": "fake-inmemory", "indexed_products": 20}
        )

    with _client(handler) as client:
        banner = fetch_health(client, "s3cret")

    assert banner.level == "ok"
    assert seen["key"] == "s3cret"


def test_run_text_search_returns_cards_and_never_puts_the_key_in_the_url() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key", "")
        return httpx.Response(200, json=_search_body())

    with _client(handler) as client:
        outcome = run_text_search(client, "váy đỏ", 5, "s3cret")

    assert outcome.ok
    assert outcome.retriever == "fake-inmemory"
    assert len(outcome.cards) == 1
    assert seen["url"] == f"{BASE}/search"
    assert "s3cret" not in seen["url"]
    assert seen["key"] == "s3cret"


def test_run_text_search_rejects_an_empty_query_without_a_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request expected")

    with _client(handler) as client:
        outcome = run_text_search(client, "   ", 5)

    assert not outcome.ok


@pytest.mark.parametrize("status_code", [401, 501, 503, 429])
def test_run_text_search_maps_error_statuses_with_the_server_detail(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": "chi tiết <script>"})

    with _client(handler) as client:
        outcome = run_text_search(client, "váy", 5)

    assert not outcome.ok
    assert outcome.error == error_message(status_code)
    assert outcome.detail == "chi tiết <script>"
    assert outcome.cards == []


def test_run_text_search_survives_a_non_json_error_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>bad gateway</html>")

    with _client(handler) as client:
        outcome = run_text_search(client, "váy", 5)

    assert not outcome.ok
    assert outcome.detail == ""


def test_run_text_search_reports_a_refused_connection_in_vietnamese() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with _client(handler) as client:
        outcome = run_text_search(client, "váy", 5)

    assert outcome.error == CONNECTION_ERROR


def test_run_image_search_posts_multipart_with_top_k() -> None:
    seen: dict[str, str] = {}
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        bodies.append(request.content)
        return httpx.Response(200, json=_search_body())

    with _client(handler) as client:
        outcome = run_image_search(client, b"\x89PNG-ish", "q.png", 7, category="váy")

    assert outcome.ok
    assert seen["url"] == f"{BASE}/search/image"
    body = bodies[0]
    assert b"q.png" in body and b"\x89PNG-ish" in body
    assert b'name="top_k"' in body and b"7" in body
    assert b'name="category"' in body


def test_run_image_search_refuses_an_oversized_upload_client_side() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request expected")

    with _client(handler) as client:
        outcome = run_image_search(client, b"x" * (MAX_IMAGE_BYTES + 1), "big.png", 5)

    assert outcome.error == error_message(413)


def test_run_image_search_refuses_an_empty_upload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request expected")

    with _client(handler) as client:
        assert not run_image_search(client, b"", "", 5).ok


def test_main_is_importable_when_streamlit_is_installed() -> None:
    pytest.importorskip("streamlit")
    from app.streamlit_app import main

    assert callable(main)
