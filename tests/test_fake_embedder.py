"""The fake embedder is the substitute for a real backbone in every other test,
so its determinism and normalization are load-bearing."""

from __future__ import annotations

import math

import pytest

from src.adapters.embedders.fake_embedder import FakeEmbedder


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def test_same_text_gives_identical_vector_across_instances() -> None:
    first = FakeEmbedder(dim=32).encode_text(["áo khoác bomber nam màu be"])
    second = FakeEmbedder(dim=32).encode_text(["áo khoác bomber nam màu be"])
    assert first == second


def test_different_text_gives_different_vector() -> None:
    embedder = FakeEmbedder(dim=32)
    vectors = embedder.encode_text(["váy hai dây đi biển", "quần baggy jean rộng ống"])
    assert vectors[0] != vectors[1]


def test_vectors_are_l2_normalized() -> None:
    embedder = FakeEmbedder(dim=16)
    rows = embedder.encode_text(["áo thun cotton", "", "   "])
    rows += embedder.encode_image([b"\x00\x01"])
    for row in rows:
        assert len(row) == 16
        assert _norm(row) == pytest.approx(1.0, abs=1e-6)


def test_batch_order_is_preserved() -> None:
    embedder = FakeEmbedder(dim=16)
    texts = ["áo dài gấm đỏ", "mũ bucket", "giày sneaker trắng"]
    batch = embedder.encode_text(texts)
    assert batch == [embedder.encode_text([text])[0] for text in texts]


def test_shared_tokens_score_higher_than_unrelated_text() -> None:
    embedder = FakeEmbedder(dim=128)
    query, related, unrelated = embedder.encode_text(
        [
            "váy hai dây đi biển",
            "Váy maxi hai dây họa tiết hoa nhí đi biển",
            "quần short thể thao nam màu đen",
        ]
    )
    similar = sum(a * b for a, b in zip(query, related, strict=False))
    different = sum(a * b for a, b in zip(query, unrelated, strict=False))
    assert similar > different


def test_identical_image_bytes_give_identical_vector() -> None:
    embedder = FakeEmbedder(dim=16)
    vectors = embedder.encode_image([b"fake-jpeg-bytes", b"fake-jpeg-bytes", b"other"])
    assert vectors[0] == vectors[1]
    assert vectors[0] != vectors[2]


def test_rejects_degenerate_dim() -> None:
    with pytest.raises(ValueError, match="dim must be >= 2"):
        FakeEmbedder(dim=1)
