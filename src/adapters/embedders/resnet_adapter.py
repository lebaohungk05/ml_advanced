"""ResNet-50 + KNN image-only baseline (partial Embedder port).

DELIBERATELY NOT A FULL Embedder: there is no shared text/image space here, so
``encode_text`` raises ``NotImplementedError`` permanently, not as a TODO. This
adapter only serves image-to-image queries (baseline #5 in DeCuong Mục 5.2,
included to show how much the vision-language alignment actually buys us).
``isinstance(x, Embedder)`` still passes — the Protocol only checks that the
methods exist — so keep image-only systems out of text-query evaluation runs.

Implementation notes:
* ``torch`` / ``torchvision`` are imported INSIDE the loader, never at module
  top — this file must stay importable with only ``requirements.txt`` installed,
  and constructing the adapter must not download anything.
* The backbone is the ImageNet classifier with its ``fc`` head replaced by an
  identity, so the forward pass returns the 2048-d globally-pooled conv
  features, not the 1000-way logits.
* Preprocessing comes from the weights enum's own ``transforms()``: the resize /
  crop / normalization constants belong to the checkpoint, not to us.
* Rows are L2-normalized: the Embedder port contracts dot product == cosine.
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Sequence
from typing import Any

from src.core.models import Matrix

FEATURE_DIM = 2048
"""Width of ResNet-50's pooled conv features (the input of the dropped ``fc``)."""


class ResnetImageEmbedder:
    """ResNet-50 pooled-feature embedder. Image queries only."""

    def __init__(
        self,
        model_id: str = "resnet50",
        weights: str = "IMAGENET1K_V2",
        device: str | None = None,
        batch_size: int = 64,
        image_size: int = 224,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_id = model_id
        self.weights = weights
        self.device = device
        self.batch_size = batch_size
        # Informational only: the real resize/crop lives in the weights' own
        # transforms(), which is already 224 for every ResNet-50 checkpoint.
        self.image_size = image_size
        self._model: Any | None = None
        self._preprocess: Any | None = None

    @property
    def name(self) -> str:
        return f"resnet:{self.model_id}"

    @property
    def dim(self) -> int:
        return FEATURE_DIM

    def encode_text(self, texts: Sequence[str]) -> Matrix:
        raise NotImplementedError(
            "ResNet-50 has no text tower; this baseline answers image queries only"
        )

    def encode_image(self, images: Sequence[bytes]) -> Matrix:
        if not images:
            return []
        import torch

        model, preprocess = self._load()
        rows: Matrix = []
        for chunk in _chunks(list(images), self.batch_size):
            batch = torch.stack([preprocess(_decode_image(data)) for data in chunk])
            with torch.no_grad():
                features = model(batch.to(self.device))
            features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
            rows.extend([float(value) for value in row] for row in features.cpu())
        return rows

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._preprocess is None:
            import torch
            from torchvision import models

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            weights = models.get_weight(f"ResNet50_Weights.{self.weights}")
            model = models.resnet50(weights=weights)
            # Dropping the classifier head turns forward() into "return the
            # pooled features", keeping the rest of the graph untouched.
            model.fc = torch.nn.Identity()
            self._model = model.to(self.device).eval()
            self._preprocess = weights.transforms()
        return self._model, self._preprocess


def _decode_image(data: bytes) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def _chunks(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
