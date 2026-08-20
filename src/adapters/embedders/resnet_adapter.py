"""STUB — ResNet-50 + KNN image-only baseline.

DELIBERATELY NOT A FULL Embedder: there is no shared text/image space here, so
``encode_text`` raises ``NotImplementedError`` permanently, not as a TODO. This
adapter only serves image-to-image queries (baseline #5 in DeCuong Mục 5.2,
included to show how much the vision-language alignment actually buys us).
``isinstance(x, Embedder)`` still passes — the Protocol only checks that the
methods exist — so keep image-only systems out of text-query evaluation runs.

Pattern to port: ``reference_repos/fashion-clip/fashion_clip/fashion_clip.py``
(``encode_images``: batched preprocessing then a forward pass; drop the text
branch and take the pooled conv features instead of a projection head).

Implementation notes for whoever picks this up:
* Import ``torch`` / ``torchvision`` INSIDE ``__init__`` (never at module top).
* Take the penultimate pooled layer (2048-d for ResNet-50), not the 1000-way
  ImageNet logits, and L2-normalize so cosine search behaves.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.models import Matrix

_TODO = (
    "TODO(Sprint 2, Hiệp): port from reference_repos/fashion-clip/fashion_clip/"
    "fashion_clip.py (encode_images), pooled features instead of logits"
)


class ResnetImageEmbedder:
    """ResNet-50 pooled-feature embedder. Image queries only."""

    def __init__(
        self,
        model_id: str = "resnet50",
        weights: str = "IMAGENET1K_V2",
        device: str = "cuda",
        batch_size: int = 64,
        image_size: int = 224,
    ) -> None:
        self.model_id = model_id
        self.weights = weights
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        # Backbone loading belongs here, behind a local import.

    @property
    def name(self) -> str:
        return f"resnet:{self.model_id}"

    @property
    def dim(self) -> int:
        raise NotImplementedError(_TODO)

    def encode_text(self, texts: Sequence[str]) -> Matrix:
        raise NotImplementedError(
            "ResNet-50 has no text tower; this baseline answers image queries only"
        )

    def encode_image(self, images: Sequence[bytes]) -> Matrix:
        raise NotImplementedError(_TODO)
