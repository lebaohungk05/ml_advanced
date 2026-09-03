"""End-to-end: config -> registry -> retriever -> index -> HTTP search.

This is the test that proves the frame is wired together. It uses the real
``configs/default.yaml``, so a broken config fails here.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import (
    API_KEY_ENV,
    MAX_IMAGE_BYTES,
    _indexed_count,
    _is_store_unavailable,
    app,
)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_retriever_and_catalog_size(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["retriever"] == "fake-inmemory"
    assert body["indexed_products"] >= 20


def test_health_reports_degraded_when_the_store_cannot_be_counted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> int:
        raise ConnectionError("qdrant is down")

    monkeypatch.setattr(app.state.retriever.store, "count", boom)

    body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["indexed_products"] == 0


def test_health_is_ok_for_a_retriever_that_has_no_vector_store() -> None:
    class StorelessRetriever:  # BM25Retriever keeps its index in itself
        name = "bm25"

    assert _indexed_count(StorelessRetriever()) == 0  # type: ignore[arg-type]


def test_demo_page_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>" in response.text


def test_catalog_image_is_served_from_the_images_root(
    client: TestClient, tmp_path: Path
) -> None:
    image = tmp_path / "train" / "images" / "000001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\xff\xd8\xff-not-a-real-jpeg")
    app.state.images_root = tmp_path

    response = client.get("/image/train/images/000001.jpg")

    assert response.status_code == 200
    assert response.content == b"\xff\xd8\xff-not-a-real-jpeg"


@pytest.mark.parametrize(
    "path",
    [
        # Percent-encoded so httpx does not collapse the '..' before sending:
        # these reach the handler and must be rejected by its own guard.
        "%2e%2e/%2e%2e/pyproject.toml",
        "train/%2e%2e/%2e%2e/%2e%2e/pyproject.toml",
        "..\\..\\pyproject.toml",
        "train/images/000001.txt",  # inside the root, but not an image extension
        "train/images/missing.jpg",
        # NUL bytes: the filesystem refuses to stat these, which used to escape
        # the handler as a 500 on a public route.
        "%00",
        "train/images/000001.jpg%00.txt",
    ],
)
def test_catalog_image_rejects_paths_outside_the_images_root(
    client: TestClient, tmp_path: Path, path: str
) -> None:
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "images" / "000001.txt").write_text("secret", encoding="utf-8")
    app.state.images_root = tmp_path

    response = client.get(f"/image/{path}")

    assert response.status_code == 404
    assert response.json()["detail"] == "image not found"


def test_catalog_image_never_leaks_a_file_outside_the_root(
    client: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "train").mkdir()
    app.state.images_root = tmp_path

    for path in ("../../pyproject.toml", "train/../../../pyproject.toml", "/etc/passwd"):
        response = client.get(f"/image/{path}")
        assert response.status_code == 404
        assert b"[project]" not in response.content


def test_text_search_returns_the_matching_product_first(client: TestClient) -> None:
    response = client.post("/search", json={"text": "váy hai dây đi biển", "top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["modality"] == "text"
    assert len(body["hits"]) == 5
    assert body["hits"][0]["product"]["product_id"] == "P003"
    assert [hit["rank"] for hit in body["hits"]] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("text", "expected_id"),
    [
        ("áo khoác bomber nam màu be", "P001"),
        ("quần baggy jean rộng ống", "P005"),
        ("áo dài gấm truyền thống màu đỏ", "P013"),
    ],
)
def test_text_search_top_hit(client: TestClient, text: str, expected_id: str) -> None:
    response = client.post("/search", json={"text": text, "top_k": 3})

    assert response.status_code == 200
    assert response.json()["hits"][0]["product"]["product_id"] == expected_id


def test_search_applies_payload_filter(client: TestClient) -> None:
    response = client.post(
        "/search", json={"text": "màu đỏ", "top_k": 10, "filters": {"category": "áo dài"}}
    )

    assert response.status_code == 200
    categories = {hit["product"]["category"] for hit in response.json()["hits"]}
    assert categories == {"áo dài"}


def test_search_rejects_empty_text(client: TestClient) -> None:
    assert client.post("/search", json={"text": ""}).status_code == 422


def test_search_rejects_out_of_range_top_k(client: TestClient) -> None:
    assert client.post("/search", json={"text": "áo", "top_k": 0}).status_code == 422


def test_image_search_returns_ranked_hits(client: TestClient) -> None:
    # The fake embedder's image space is unrelated to its text space, so only the
    # plumbing is asserted here — ranking quality needs a real backbone.
    response = client.post(
        "/search/image",
        files={"image": ("query.jpg", b"fake-jpeg-bytes", "image/jpeg")},
        data={"top_k": "3"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["modality"] == "image"
    assert [hit["rank"] for hit in body["hits"]] == [1, 2, 3]


def test_image_search_rejects_empty_upload(client: TestClient) -> None:
    response = client.post(
        "/search/image", files={"image": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 422


def test_image_search_rejects_an_upload_over_the_size_cap(client: TestClient) -> None:
    oversized = b"\xff\xd8\xff" + b"x" * MAX_IMAGE_BYTES

    response = client.post(
        "/search/image", files={"image": ("huge.jpg", oversized, "image/jpeg")}
    )

    assert response.status_code == 413


def test_a_stub_adapter_surfaces_as_501_not_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DenseRetriever raises NotImplementedError for a multimodal query, and an
    # unimplemented adapter can be wired in by config. Either way the caller
    # deserves "not implemented", not an opaque crash.
    def not_built_yet(query: object) -> None:
        raise NotImplementedError("fusing text and image vectors is not implemented")

    monkeypatch.setattr(app.state.retriever, "search", not_built_yet)

    response = client.post("/search", json={"text": "áo khoác"})

    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"]


def test_a_non_transport_failure_is_not_disguised_as_an_unreachable_store(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bug in the store must not be relabelled "start qdrant.exe": that sends
    # the operator to restart a service that is already running fine.
    def boom(query: object) -> None:
        raise RuntimeError("payload schema mismatch")

    monkeypatch.setattr(app.state.retriever, "search", boom)

    with pytest.raises(RuntimeError, match="payload schema mismatch"):
        client.post("/search", json={"text": "áo khoác"})


def test_a_builtin_connection_error_counts_as_an_unreachable_store() -> None:
    # Not every adapter wraps httpx: a plain socket-level ConnectionError from
    # any future store must map to the same 503.
    assert _is_store_unavailable(ConnectionError("connection refused")) is True
    assert _is_store_unavailable(TimeoutError("timed out")) is True


def test_search_requires_api_key_when_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "s3cret")

    assert client.post("/search", json={"text": "áo thun"}).status_code == 401
    assert (
        client.post(
            "/search", json={"text": "áo thun"}, headers={"X-API-Key": "wrong"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/search", json={"text": "áo thun"}, headers={"X-API-Key": "s3cret"}
        ).status_code
        == 200
    )
    # /health stays public by design.
    assert client.get("/health").status_code == 200


def test_image_search_requires_api_key_when_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "s3cret")
    upload = {"image": ("query.jpg", b"fake-jpeg-bytes", "image/jpeg")}

    assert client.post("/search/image", files=upload).status_code == 401
    assert (
        client.post(
            "/search/image", files=upload, headers={"X-API-Key": "s3cret"}
        ).status_code
        == 200
    )


def test_demo_page_and_images_stay_public_when_an_api_key_is_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Deliberate, not an oversight: a browser cannot attach X-API-Key to an <img>
    # request, so the page and its images must not require one.
    image = tmp_path / "train" / "images" / "000001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\xff\xd8\xff-not-a-real-jpeg")
    app.state.images_root = tmp_path
    monkeypatch.setenv(API_KEY_ENV, "s3cret")

    assert client.get("/").status_code == 200
    assert client.get("/image/train/images/000001.jpg").status_code == 200


def test_search_reports_an_unreachable_store_as_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What qdrant_client raises underneath when the server is not listening.
    message = "All connection attempts failed to http://localhost:6333 (api_key=sup3r-s3cret)"

    def boom(query: object) -> None:
        raise httpx.ConnectError(message)

    monkeypatch.setattr(app.state.retriever, "search", boom)

    response = client.post("/search", json={"text": "áo khoác"})

    assert response.status_code == 503
    assert response.json()["detail"] == "vector store unavailable — start qdrant.exe"
    assert "localhost:6333" not in response.text
    assert "sup3r-s3cret" not in response.text


def test_search_reports_a_wrapped_qdrant_transport_failure_as_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    exceptions = pytest.importorskip("qdrant_client.http.exceptions")

    def boom(query: object) -> None:
        raise exceptions.ResponseHandlingException(httpx.ConnectError("connection refused"))

    monkeypatch.setattr(app.state.retriever, "search", boom)

    assert client.post("/search", json={"text": "áo khoác"}).status_code == 503


def test_a_qdrant_api_error_is_not_disguised_as_an_unreachable_store() -> None:
    # A missing collection or a rejected API key is a qdrant_client exception too,
    # but "start qdrant.exe" would be a lie: the server answered.
    exceptions = pytest.importorskip("qdrant_client.http.exceptions")
    unexpected = exceptions.UnexpectedResponse(
        status_code=404, reason_phrase="Not Found", content=b"collection not found", headers={}
    )

    assert _is_store_unavailable(unexpected) is False
