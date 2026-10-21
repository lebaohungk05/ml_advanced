"""OpenAI CLIP zero-shot baseline (Embedder port).

Backed by ``sentence-transformers`` (``clip-ViT-B-32``, 512-dim), which bundles
the image tower and the text tower behind a single ``encode`` call and handles
the batching/preprocessing for both modalities.

Implementation notes:
* ``torch`` / ``sentence_transformers`` are imported INSIDE the loader, never at
  module top — this file must stay importable with only ``requirements.txt``
  installed, and constructing the adapter must not download anything.
* Rows come back L2-normalized: the Embedder port contracts dot product ==
  cosine.
* CLIP's text tower caps at 77 tokens; Vietnamese queries tokenize long, so
  check for truncation before blaming the model for bad recall.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

from src.core.models import Matrix

DEFAULT_MODEL_ID = "clip-ViT-B-32"
"""Sentence-Transformers alias for OpenAI CLIP ViT-B/32 (512-dim)."""


class ClipEmbedder:
    """CLIP zero-shot embedder. Baseline #1 in DeCuong Mục 5.2."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 77,
        normalize: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        # CLIP's text tower is fixed at 77 positions; kept as config so callers
        # can assert against it when checking Vietnamese queries for truncation.
        self.max_length = max_length
        self.normalize = normalize
        self._model: Any | None = None
        self._dim: int | None = None

    @property
    def name(self) -> str:
        return f"clip:{self.model_id}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            # The CLIP module does not report its output size, so fall back to
            # measuring one probe encoding.
            reported = self._load().get_sentence_embedding_dimension()
            self._dim = int(reported) if reported else len(self._encode(["dim probe"])[0])
        return self._dim

    def encode_text(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return []
        return self._encode(list(texts))

    def encode_image(self, images: Sequence[bytes]) -> Matrix:
        if not images:
            return []
        return self._encode([_decode_image(data) for data in images])

    def _load(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    def _encode(self, inputs: list[Any]) -> Matrix:
        # SentenceTransformer.encode does the mini-batching itself, so the whole
        # list goes in at once and batch_size only caps GPU memory.
        encoded = self._load().encode(
            inputs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in encoded]


def _decode_image(data: bytes) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")
