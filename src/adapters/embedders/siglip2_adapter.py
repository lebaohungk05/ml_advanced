"""STUB — SigLIP 2 backbone, the project's primary system (Embedder port).

Pattern to port: ``reference_repos/marqo-FashionCLIP/models/hf_models.py`` — the
``HFCLIP`` wrapper (``AutoModel`` + ``AutoProcessor``, ``get_image_features`` /
``get_text_features``, ``F.normalize``). Note that file already special-cases
SigLIP: ``max_length=64`` and ``padding="max_length"`` are required, SigLIP was
trained with fixed-length padding and gives degraded vectors without it.

Implementation notes for whoever picks this up:
* Import ``torch`` / ``transformers`` INSIDE ``__init__`` (never at module top).
* Keep Vietnamese diacritics (DeCuong Mục 4.3) — the multilingual tokenizer
  handles them and stripping them is ambiguous.
* ``adapter_path`` is where a LoRA/DoRA checkpoint from Sprint 3 gets loaded
  (``peft.PeftModel.from_pretrained``); leave it ``None`` for the zero-shot run.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.models import Matrix

_TODO = (
    "TODO(Sprint 2, Hùng): port from reference_repos/marqo-FashionCLIP/models/"
    "hf_models.py (HFCLIP + load_model, note the siglip max_length=64 branch)"
)


class Siglip2Embedder:
    """SigLIP 2 embedder, optionally with a LoRA/DoRA adapter on top."""

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-224",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 64,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        normalize: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.normalize = normalize
        # Model/processor loading belongs here, behind a local import.

    @property
    def name(self) -> str:
        suffix = "+adapter" if self.adapter_path else ""
        return f"siglip2:{self.model_id}{suffix}"

    @property
    def dim(self) -> int:
        raise NotImplementedError(_TODO)

    def encode_text(self, texts: Sequence[str]) -> Matrix:
        raise NotImplementedError(_TODO)

    def encode_image(self, images: Sequence[bytes]) -> Matrix:
        raise NotImplementedError(_TODO)
