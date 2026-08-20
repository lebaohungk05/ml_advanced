"""End-to-end: config -> registry -> retriever -> index -> HTTP search.

This is the test that proves the frame is wired together. It uses the real
``configs/default.yaml``, so a broken config fails here.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import API_KEY_ENV, app


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
