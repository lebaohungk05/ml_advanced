"""SigLIP 2 backbone, the project's primary system (Embedder port).

Generic ``AutoModel`` / ``AutoProcessor`` wrapper: nothing here is specific to
SigLIP 2, so the Sprint 4 ablation can point ``model_id`` at SigLIP 1
(``google/siglip-base-patch16-256``) and compare the two through the same class.

Implementation notes:
* ``torch`` / ``transformers`` are imported INSIDE the loader, never at module
  top — this file must stay importable with only ``requirements.txt`` installed,
  and constructing the adapter must not download anything.
* SigLIP was trained with fixed-length padding: ``padding="max_length"`` with
  ``max_length=64`` is required, vectors degrade without it.
* The towers do not self-normalize, so rows are L2-normalized here (in float32,
  because a bf16 norm is only accurate to ~1e-2).
* Keep Vietnamese diacritics (DeCuong Mục 4.3) — the multilingual tokenizer
  handles them and stripping them is ambiguous.
* ``adapter_path`` is where a LoRA/DoRA checkpoint from Sprint 3 gets loaded;
  leave it ``None`` for the zero-shot run.
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Sequence
from typing import Any

from src.core.models import Matrix

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"


class Siglip2Embedder:
    """SigLIP 2 embedder, optionally with a LoRA/DoRA adapter on top."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 64,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        normalize: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.normalize = normalize
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def name(self) -> str:
        suffix = "+adapter" if self.adapter_path else ""
        return f"siglip2:{self.model_id}{suffix}"

    @property
    def dim(self) -> int:
        config = self._load()[0].config
        projection_dim = getattr(config, "projection_dim", None)
        if projection_dim:
            return int(projection_dim)
        return int(config.text_config.hidden_size)

    def encode_text(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return []
        model, processor = self._load()
        rows: Matrix = []
        for chunk in _chunks(list(texts), self.batch_size):
            inputs = processor(
                text=chunk,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            rows.extend(self._features(model.get_text_features, inputs))
        return rows

    def encode_image(self, images: Sequence[bytes]) -> Matrix:
        if not images:
            return []
        model, processor = self._load()
        rows: Matrix = []
        for chunk in _chunks(list(images), self.batch_size):
            inputs = processor(
                images=[_decode_image(data) for data in chunk],
                return_tensors="pt",
            ).to(device=self.device, dtype=model.dtype)
            rows.extend(self._features(model.get_image_features, inputs))
        return rows

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            import torch
            from transformers import AutoModel, AutoProcessor

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModel.from_pretrained(self.model_id, dtype=getattr(torch, self.dtype))
            if self.adapter_path is not None:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter_path)
            self._model = model.to(self.device).eval()
            # AutoProcessor.from_pretrained is unannotated in transformers.
            self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                self.model_id
            )
        return self._model, self._processor

    def _features(self, tower: Any, inputs: Any) -> Matrix:
        import torch

        with torch.no_grad():
            output = tower(**inputs)
        # transformers >= 5 returns a ModelOutput from get_*_features; the
        # projected embedding is its pooler_output. Older versions return the
        # tensor directly.
        features = output if isinstance(output, torch.Tensor) else output.pooler_output
        features = features.float()
        if self.normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return [[float(value) for value in row] for row in features.cpu()]


def _decode_image(data: bytes) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def _chunks(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
