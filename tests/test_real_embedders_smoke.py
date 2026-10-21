"""End-to-end smoke test for the real backbones (CLIP, SigLIP 2, ResNet-50).

Marked ``slow`` because it downloads real model weights and runs a forward pass,
so plain ``pytest -q`` skips it. Run explicitly:

    pytest tests/test_real_embedders_smoke.py -m slow -v
"""

from __future__ import annotations

import io
import math

import pytest

from src.adapters.embedders.clip_adapter import ClipEmbedder
from src.adapters.embedders.resnet_adapter import ResnetImageEmbedder
from src.adapters.embedders.siglip2_adapter import Siglip2Embedder
from src.adapters.embedders.visiglip_adapter import ViSiglipEmbedder
from src.core.models import Matrix

QUERY = "áo sơ mi nam trắng"
NORM_TOLERANCE = 1e-3


def _tiny_image_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(200, 120, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def _assert_unit_rows(matrix: Matrix, dim: int) -> None:
    assert len(matrix) == 1
    assert len(matrix[0]) == dim
    norm = math.sqrt(sum(value * value for value in matrix[0]))
    assert abs(norm - 1.0) < NORM_TOLERANCE, f"expected unit norm, got {norm}"


@pytest.mark.slow
def test_clip_embeds_text_and_image_as_unit_vectors() -> None:
    embedder = ClipEmbedder()

    assert embedder.dim == 512
    _assert_unit_rows(embedder.encode_text([QUERY]), embedder.dim)
    _assert_unit_rows(embedder.encode_image([_tiny_image_bytes()]), embedder.dim)


@pytest.mark.slow
def test_siglip2_embeds_text_and_image_as_unit_vectors() -> None:
    embedder = Siglip2Embedder()

    assert embedder.dim == 768
    _assert_unit_rows(embedder.encode_text([QUERY]), embedder.dim)
    _assert_unit_rows(embedder.encode_image([_tiny_image_bytes()]), embedder.dim)


@pytest.mark.slow
def test_siglip_class_is_reusable_for_the_sprint4_ablation() -> None:
    embedder = Siglip2Embedder(model_id="google/siglip-base-patch16-256")

    assert embedder.name == "siglip2:google/siglip-base-patch16-256"
    _assert_unit_rows(embedder.encode_text([QUERY]), embedder.dim)


@pytest.mark.slow
def test_resnet_embeds_images_as_unit_vectors_and_refuses_text() -> None:
    embedder = ResnetImageEmbedder()

    assert embedder.dim == 2048
    _assert_unit_rows(embedder.encode_image([_tiny_image_bytes()]), embedder.dim)

    with pytest.raises(NotImplementedError, match="no text tower"):
        embedder.encode_text([QUERY])


@pytest.mark.slow
def test_visiglip_ot_embeds_text_and_image_as_unit_vectors() -> None:
    embedder = ViSiglipEmbedder()

    assert embedder.dim == 768
    _assert_unit_rows(embedder.encode_text([QUERY]), embedder.dim)
    _assert_unit_rows(embedder.encode_image([_tiny_image_bytes()]), embedder.dim)


@pytest.mark.slow
def test_batching_preserves_input_order() -> None:
    embedder = ClipEmbedder(batch_size=2)
    texts = ["áo sơ mi nam trắng", "váy hai dây", "quần jean xanh", "giày thể thao", "mũ lưỡi trai"]

    rows = embedder.encode_text(texts)

    assert len(rows) == len(texts)
    assert rows[0] == pytest.approx(embedder.encode_text([texts[0]])[0], abs=1e-5)
