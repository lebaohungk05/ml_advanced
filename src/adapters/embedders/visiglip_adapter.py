"""ViSigLIP-OT, the Vietnamese-native backbone (Embedder port).

Counterpart in the central comparison of the project ("native Vietnamese
backbone vs. multilingual SigLIP 2", DeCuong Mục 2.5 / ablation axis 3).

Real checkpoint: ``minhnguyent546/ViSigLIP-OT`` (confirmed from its HF model
card, not the placeholder id this stub used to carry) — a ``trust_remote_code``
model exposing its own ``encode_text``/``encode_image`` convenience methods
(dual-encoder: Vietnamese-SBERT text tower + DINOv3-ViT-B/16 image tower,
768-d, 221M params total). It does its own batching internally, so this
adapter's ``batch_size`` is just forwarded, not re-implemented.

Two numerical gotchas found by actually running this on real hardware, not by
guessing:
* ``dtype="bfloat16"`` produces NaN image embeddings on this checkpoint —
  defaulted to float32 instead.
* Even at float32, image encoding returns NaN on CUDA on this dev machine
  (Quadro T2000, an older/small GPU) while the exact same call is correct on
  CPU. Root cause not isolated further (likely a kernel/precision issue in the
  DINOv3 image tower on this compute capability) — device defaults to CPU for
  this adapter specifically, overriding the "auto-detect cuda" convention used
  by the other embedders in this package. If you're on a newer/bigger GPU and
  confirm CUDA gives non-NaN output there, pass ``device="cuda"`` explicitly.
"""

from __future__ import annotations

import io
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from src.core.models import Matrix

_DIM = 768


class ViSiglipEmbedder:
    """ViSigLIP-OT (221M) embedder — Vietnamese-native baseline."""

    def __init__(
        self,
        model_id: str = "minhnguyent546/ViSigLIP-OT",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 256,
        adapter_path: str | None = None,
        dtype: str = "float32",
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
        self._model = None

    @property
    def name(self) -> str:
        suffix = "+adapter" if self.adapter_path else ""
        return f"visiglip:{self.model_id}{suffix}"

    @property
    def dim(self) -> int:
        return _DIM

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel

        # NOT "cuda if available" like the other embedders in this package —
        # see the module docstring: this checkpoint returns NaN on CUDA on at
        # least one real dev GPU. CPU is the safe default; override explicitly
        # if you've confirmed CUDA is clean on your hardware.
        device = self.device or "cpu"
        dtype = getattr(torch, self.dtype)
        model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code, dtype=dtype
        )
        model.to(device)
        model.eval()
        if self.adapter_path:
            # TODO(Sprint 3): no fine-tuned checkpoint exists yet to test this
            # path against — same situation as siglip2_adapter.py.
            raise NotImplementedError(
                "adapter_path loading for ViSigLIP-OT has no checkpoint to test "
                "against yet (Sprint 3 work)"
            )
        self._model = model
        self._device = device

    def encode_text(self, texts: Sequence[str]) -> Matrix:
        self._load()
        assert self._model is not None
        emb = self._model.encode_text(
            sentences=list(texts),
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize=self.normalize,
        )
        return self._to_matrix(emb)

    def encode_image(self, images: Sequence[bytes]) -> Matrix:
        self._load()
        assert self._model is not None
        from PIL import Image

        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        try:
            emb = self._model.encode_image(
                images=pil_images,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize=self.normalize,
            )
        except (TypeError, AttributeError, ValueError):
            # Some trust_remote_code revisions only accept path-like image
            # sources, not PIL.Image objects directly — fall back to temp files.
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for i, img in enumerate(pil_images):
                    p = Path(tmp) / f"{i}.png"
                    img.save(p)
                    paths.append(str(p))
                emb = self._model.encode_image(
                    images=paths,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                    normalize=self.normalize,
                )
        return self._to_matrix(emb)

    def _to_matrix(self, tensor) -> Matrix:  # type: ignore[no-untyped-def]
        import torch

        t = tensor.detach().to(torch.float32).cpu()
        if self.normalize:
            norms = t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            t = t / norms
        return cast(Matrix, t.tolist())
