"""STUB — OpenAI CLIP zero-shot baseline (Embedder port).

Pattern to port: ``reference_repos/fashion-clip/fashion_clip/fashion_clip.py``
(see ``encode_images`` / ``encode_text``: batched processor call, then L2
normalize). ``reference_repos/marqo-FashionCLIP/models/hf_models.py`` shows the
same thing in ~15 lines if you prefer the shorter version.

Implementation notes for whoever picks this up:
* Import ``torch`` / ``transformers`` INSIDE ``__init__`` (never at module top) —
  this file must stay importable with only ``requirements.txt`` installed.
* Return L2-normalized rows: the Embedder port contracts dot product == cosine.
* CLIP's text tower caps at 77 tokens; Vietnamese queries tokenize long, so
  check for truncation before blaming the model for bad recall.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.models import Matrix

_TODO = (
    "TODO(Sprint 2, Hùng): port from reference_repos/fashion-clip/fashion_clip/"
    "fashion_clip.py (encode_images / encode_text)"
)


class ClipEmbedder:
    """CLIP zero-shot embedder. Baseline #1 in DeCuong Mục 5.2."""

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 77,
        normalize: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        # Model/processor loading belongs here, behind a local import.

    @property
    def name(self) -> str:
        return f"clip:{self.model_id}"

    @property
    def dim(self) -> int:
        raise NotImplementedError(_TODO)

    def encode_text(self, texts: Sequence[str]) -> Matrix:
        raise NotImplementedError(_TODO)

    def encode_image(self, images: Sequence[bytes]) -> Matrix:
        raise NotImplementedError(_TODO)
