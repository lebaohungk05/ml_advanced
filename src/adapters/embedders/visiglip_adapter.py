"""STUB — ViSigLIP-OT, the Vietnamese-native backbone (Embedder port).

This is the counterpart in the central comparison of the project ("native
Vietnamese backbone vs. multilingual SigLIP 2", DeCuong Mục 2.5 / ablation
axis 3), so it must be wired exactly like ``siglip2_adapter.py`` — same
normalization, same truncation policy — or the comparison is confounded.

Pattern to port: ``reference_repos/marqo-FashionCLIP/models/hf_models.py`` if the
checkpoint is on the HF hub (likely; needs ``trust_remote_code=True``), otherwise
``reference_repos/marqo-FashionCLIP/models/open_clip_model.py`` for the
open_clip-style loading path.

Implementation notes for whoever picks this up:
* Import ``torch`` / ``transformers`` INSIDE ``__init__`` (never at module top).
* Confirm the tokenizer's max length from the model card instead of assuming 64.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.models import Matrix

_TODO = (
    "TODO(Sprint 2, Hùng): port from reference_repos/marqo-FashionCLIP/models/"
    "hf_models.py (or open_clip_model.py if the checkpoint is open_clip-style)"
)


class ViSiglipEmbedder:
    """ViSigLIP-OT (0.2B) embedder — Vietnamese-native baseline."""

    def __init__(
        self,
        model_id: str = "visiglip/visiglip-ot-0.2b",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 64,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        normalize: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.normalize = normalize
        # Model/processor loading belongs here, behind a local import.

    @property
    def name(self) -> str:
        suffix = "+adapter" if self.adapter_path else ""
        return f"visiglip:{self.model_id}{suffix}"

    @property
    def dim(self) -> int:
        raise NotImplementedError(_TODO)

    def encode_text(self, texts: Sequence[str]) -> Matrix:
        raise NotImplementedError(_TODO)

    def encode_image(self, images: Sequence[bytes]) -> Matrix:
        raise NotImplementedError(_TODO)
