"""STUB — LoRA/DoRA fine-tuning of the frozen backbone (Sprint 3).

``TrainConfig`` below is real and complete: every default is the value agreed in
DeCuong Mục 3.3's hyperparameter table, so a run is reproducible from the config
alone and ``runs.csv`` rows can be filled straight from it. The functions are
signatures only.

Reference for the loading/normalizing side:
``reference_repos/marqo-FashionCLIP/models/hf_models.py``. Neither cloned repo
does PEFT training, so the LoRA/DoRA and GradCache parts come from the PEFT and
GradCache docs, not from a repo we can copy.

Implementation notes for whoever picks this up:
* No heavy imports at module level — ``torch``/``peft``/``transformers`` go inside
  the functions, so ``TrainConfig`` stays importable without the ML extras.
* DoRA needs ``peft >= 0.18`` (``use_dora=True`` in ``LoraConfig``).
* Backbone stays FULLY FROZEN; only adapter params train. Log the trainable
  parameter count and share — the report quotes it (expected 1-2%).
* Physical batch 32 vs. effective 256 is not a typo: GradCache does two passes to
  simulate the large contrastive batch on 12 GB VRAM.
* Checkpoint selection is Recall@5 on validation, not training loss.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.models import Product


@dataclass(slots=True)
class TrainConfig:
    """Hyperparameters for one fine-tuning run (DeCuong Mục 3.3)."""

    # backbone
    model_id: str = "google/siglip2-base-patch16-224"
    freeze_backbone: bool = True

    # optimizer / schedule
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    lr_schedule: str = "cosine"
    warmup_steps: int = 500
    weight_decay: float = 0.01
    epochs: int = 10
    early_stopping_patience: int = 2

    # batching — effective batch is reached via GradCache, not real VRAM
    physical_batch_size: int = 32
    effective_batch_size: int = 256
    use_grad_cache: bool = True

    # precision / reproducibility
    bf16: bool = True
    seed: int = 42

    # LoRA / DoRA adapter
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    use_dora: bool = True
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")
    adapt_image_tower: bool = True
    adapt_text_tower: bool = True

    # loss / negatives
    loss: str = "sigmoid_pairwise"
    use_hard_negatives: bool = True
    hard_negatives_per_sample: int = 4
    hard_negative_start_epoch: int = 5

    # checkpointing / logging
    checkpoint_metric: str = "recall@5"
    output_dir: Path = Path("checkpoints")
    run_id: str | None = None
    wandb_project: str | None = "fashion-multimodal-search"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.effective_batch_size % self.physical_batch_size != 0:
            raise ValueError(
                "effective_batch_size must be a multiple of physical_batch_size "
                f"(got {self.effective_batch_size} / {self.physical_batch_size})"
            )
        if not self.adapt_image_tower and not self.adapt_text_tower:
            raise ValueError("at least one tower must be adapted")

    @property
    def grad_accumulation_steps(self) -> int:
        return self.effective_batch_size // self.physical_batch_size


def build_peft_model(config: TrainConfig) -> Any:
    """Load the frozen backbone and attach the LoRA/DoRA adapter."""
    # TODO(Sprint 3, Hùng): peft.LoraConfig(r=..., lora_alpha=..., use_dora=True,
    # target_modules=config.target_modules) -> get_peft_model; import torch/peft here.
    raise NotImplementedError("TODO(Sprint 3, Hùng): PEFT >= 0.18 LoraConfig(use_dora=True)")


def sigmoid_pairwise_loss(
    image_embeddings: Any,
    text_embeddings: Any,
    logit_scale: Any,
    logit_bias: Any,
) -> Any:
    """SigLIP sigmoid pairwise loss over the in-batch pair matrix."""
    # TODO(Sprint 3, Hùng): -logsigmoid(labels * (scale * img @ txt.T + bias)).mean(),
    # labels = +1 on the diagonal and -1 elsewhere (SigLIP paper eq. 2).
    raise NotImplementedError("TODO(Sprint 3, Hùng): SigLIP sigmoid loss")


def mine_hard_negatives(
    products: Sequence[Product],
    embeddings: Any,
    config: TrainConfig,
) -> dict[str, list[str]]:
    """Pick same-category, different-colour/style negatives per product."""
    # TODO(Sprint 3, Hiệp): nearest neighbours within the same category, excluding
    # same colour; returns product_id -> negative product_ids.
    raise NotImplementedError("TODO(Sprint 3, Hiệp): category-constrained hard negatives")


def train(
    config: TrainConfig,
    train_products: Sequence[Product],
    val_products: Sequence[Product],
) -> Path:
    """Run fine-tuning; returns the path of the best checkpoint by Recall@5."""
    # TODO(Sprint 3, Hùng): GradCache two-pass loop, cosine schedule with warmup,
    # eval each epoch, early stopping, append the run to experiments/runs.csv.
    raise NotImplementedError("TODO(Sprint 3, Hùng): training loop with GradCache")
