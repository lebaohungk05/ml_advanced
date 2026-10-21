"""LoRA/DoRA fine-tuning of the frozen backbone (Sprint 3).

``TrainConfig`` is the agreed hyperparameter table (DeCuong Mục 3.3), so a run is
reproducible from the config alone and ``runs.csv`` rows can be filled straight
from it.

Reference for the loading/normalizing side:
``reference_repos/marqo-FashionCLIP/models/hf_models.py``. Neither cloned repo
does PEFT training, so the LoRA/DoRA and GradCache parts follow the PEFT and
GradCache papers/docs rather than a repo we can copy.

Implementation notes:
* No heavy imports at module level — ``torch``/``peft``/``transformers`` go inside
  the functions, so ``TrainConfig`` stays importable without the ML extras.
* DoRA needs ``peft >= 0.18`` (``use_dora=True`` in ``LoraConfig``).
* Backbone stays FULLY FROZEN; only adapter params train. The trainable
  parameter count and share are logged — the report quotes them.
* Physical batch 32 vs. effective 256 is not a typo: the GradCache two-pass loop
  in ``_grad_cache_step`` simulates the large contrastive batch on 12 GB VRAM.
  Pass 1 embeds every micro-batch under ``no_grad`` and caches the vectors, the
  sigmoid loss is taken over the whole effective batch, then pass 2 re-runs each
  micro-batch WITH grad and seeds ``autograd.backward`` with the cached
  representation gradients. Only one micro-batch of activations is ever alive.
  Hand-rolled rather than the ``grad-cache`` PyPI package: that package wants to
  own the forward call, which fights PEFT's wrapped ``get_*_features``.
* Checkpoint selection is Recall@5 on validation, not training loss.

Two documented simplifications, both forced by the dataset we actually have
(Fashionpedia's HF mirror kept category + bbox but none of the paper's 294
fine-grained attributes — see DeCuong Mục 6.1):
* Training captions are ``Product.to_text()``, i.e. category names only. No
  colour/material text signal exists, so the contrastive text side is coarser
  than Mục 3.3 assumed.
* Hard negatives are same-category nearest neighbours in embedding space (the
  "different colour" half of the original rule is not expressible).
* Validation Recall@5 for checkpoint selection uses each val product's own
  caption as a synthetic self-query (no labelled query set exists at training
  time). Captions repeat across products of the same category, so this number is
  a pessimistic proxy for ranking quality, useful only for comparing epochs of
  the same run — never quote it as a headline retrieval metric.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.models import Product

logger = logging.getLogger(__name__)

TEXT_TOWER_PREFIX = "text_model"
VISION_TOWER_PREFIX = "vision_model"
MAX_TEXT_LENGTH = 64
"""SigLIP was trained with fixed-length padding; 64 matches ``Siglip2Embedder``."""

DEFAULT_IMAGE_ROOT = Path("data/raw/fashionpedia")
"""``Product.image_path`` is stored relative to this; override via ``extra['image_root']``."""

DEFAULT_RUNS_CSV = Path("experiments/runs.csv")
"""Experiment log (DeCuong Mục 4.4); override via ``extra['runs_csv']`` (tests do)."""


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


def lora_target_modules(model: Any, config: TrainConfig) -> list[str]:
    """Fully-qualified names of the modules LoRA should wrap on ``model``.

    Qualified rather than suffix-matched because ``adapt_image_tower`` /
    ``adapt_text_tower`` have to be honoured and SigLIP's two towers use the same
    leaf names (``text_model.encoder…q_proj`` vs ``vision_model.encoder…q_proj``).
    Restricted to ``<tower>.encoder.layers`` on purpose: the attention-pooling
    head also owns an ``out_proj``, but it belongs to ``nn.MultiheadAttention``,
    which PEFT cannot wrap cleanly.
    """
    prefixes = []
    if config.adapt_image_tower:
        prefixes.append(f"{VISION_TOWER_PREFIX}.encoder.layers.")
    if config.adapt_text_tower:
        prefixes.append(f"{TEXT_TOWER_PREFIX}.encoder.layers.")

    leaves = set(config.target_modules)
    targets = [
        name
        for name, _ in model.named_modules()
        if name.rsplit(".", 1)[-1] in leaves and any(name.startswith(p) for p in prefixes)
    ]
    if not targets:
        raise ValueError(
            f"no module of {sorted(leaves)} found under {prefixes} in "
            f"{type(model).__name__}; the backbone's module names changed"
        )
    return targets


def build_peft_model(config: TrainConfig) -> Any:
    """Load the frozen backbone and attach the LoRA/DoRA adapter."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel

    model = AutoModel.from_pretrained(config.model_id, dtype=_torch_dtype(config))
    if config.freeze_backbone:
        model.requires_grad_(False)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        use_dora=config.use_dora,
        target_modules=lora_target_modules(model, config),
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    trainable, total = trainable_parameters(peft_model)
    logger.info(
        "%s %s: %d trainable / %d total params (%.2f%%), towers=%s",
        "DoRA" if config.use_dora else "LoRA",
        config.model_id,
        trainable,
        total,
        100.0 * trainable / total,
        "+".join(
            name
            for name, on in (
                ("image", config.adapt_image_tower),
                ("text", config.adapt_text_tower),
            )
            if on
        ),
    )
    return peft_model


def trainable_parameters(model: Any) -> tuple[int, int]:
    """``(trainable, total)`` parameter counts — the report quotes the share."""
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def sigmoid_pairwise_loss(
    image_embeddings: Any,
    text_embeddings: Any,
    logit_scale: Any,
    logit_bias: Any,
) -> Any:
    """SigLIP sigmoid pairwise loss over the in-batch pair matrix (paper eq. 2).

    ``logits = logit_scale * image @ text.T + logit_bias``; the label is +1 on the
    diagonal (the matched pair) and -1 everywhere else, and the loss is
    ``-mean(logsigmoid(label * logit))``. Unlike softmax InfoNCE this is a sum of
    independent binary problems, which is exactly why GradCache can split the
    batch without changing the objective.
    """
    import torch

    logits = logit_scale * image_embeddings @ text_embeddings.transpose(0, 1) + logit_bias
    labels = -torch.ones_like(logits)
    labels.fill_diagonal_(1.0)
    loss = -torch.nn.functional.logsigmoid(labels * logits).mean()
    return loss


def mine_hard_negatives(
    products: Sequence[Product],
    embeddings: Any,
    config: TrainConfig,
) -> dict[str, list[str]]:
    """Pick same-category nearest neighbours as hard negatives per product.

    ``embeddings`` is aligned index-to-index with ``products`` (numpy array, torch
    tensor or list of rows). For each product the negatives are the
    ``config.hard_negatives_per_sample`` other members of the SAME category with
    the highest cosine similarity — i.e. the items a retriever is most likely to
    confuse it with.

    Deviation from the original design: Mục 3.3 asked for "same category,
    different colour", but the dataset has no colour/attribute field at all (see
    the module docstring), so category + embedding proximity is the whole signal.
    Categories with fewer members than requested simply yield a shorter list, and
    a category with a single member yields no entry at all.
    """
    import numpy as np

    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().to("cpu").float().numpy()
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(products):
        raise ValueError(
            f"embeddings must have shape ({len(products)}, dim), got {matrix.shape}"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)

    by_category: dict[str, list[int]] = {}
    for index, product in enumerate(products):
        by_category.setdefault(product.category, []).append(index)

    wanted = config.hard_negatives_per_sample
    negatives: dict[str, list[str]] = {}
    for group in by_category.values():
        if len(group) < 2:
            continue
        group_matrix = matrix[group]
        for chunk_start in range(0, len(group), _MINING_CHUNK):
            chunk = group[chunk_start : chunk_start + _MINING_CHUNK]
            scores = matrix[chunk] @ group_matrix.T
            # A product is its own nearest neighbour; mask the diagonal out.
            for row in range(len(chunk)):
                scores[row, chunk_start + row] = -np.inf
            keep = min(wanted, len(group) - 1)
            top = np.argpartition(-scores, keep - 1, axis=1)[:, :keep]
            for row, index in enumerate(chunk):
                ranked = sorted(top[row], key=lambda pos: -scores[row, pos])
                negatives[products[index].product_id] = [
                    products[group[int(pos)]].product_id for pos in ranked
                ]
    return negatives


_MINING_CHUNK = 512
"""Similarity is computed in row chunks: the biggest category has 12.5k members."""


def train(
    config: TrainConfig,
    train_products: Sequence[Product],
    val_products: Sequence[Product],
) -> Path:
    """Run fine-tuning; returns the path of the best checkpoint by Recall@5."""
    import torch
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup

    # Reuses the runs.csv schema and writer of the evaluation CLI (DeCuong Mục
    # 4.4) rather than defining a second, drifting set of columns.
    from src.evaluate import append_run_row, git_commit

    if not train_products:
        raise ValueError("train_products must not be empty")
    if not val_products:
        raise ValueError("val_products must not be empty")

    started = time.perf_counter()
    kind = "dora" if config.use_dora else "lora"
    run_id = config.run_id or f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}"
    _seed_everything(config.seed)

    device = _select_device()
    dtype = _torch_dtype(config)
    model = build_peft_model(config).to(device)
    # AutoProcessor.from_pretrained is unannotated in transformers.
    processor = AutoProcessor.from_pretrained(config.model_id)  # type: ignore[no-untyped-call]
    base_model = model.get_base_model()
    image_root = Path(config.extra.get("image_root", DEFAULT_IMAGE_ROOT))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_products) / config.effective_batch_size)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = min(config.warmup_steps, max(1, total_steps // 2))
    if warmup_steps != config.warmup_steps:
        logger.warning(
            "warmup_steps clipped %d -> %d: only %d optimizer steps in this run",
            config.warmup_steps,
            warmup_steps,
            total_steps,
        )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    run = _init_wandb(config, run_id)

    rng = random.Random(config.seed)
    best_metric = -1.0
    best_path: Path | None = None
    epochs_without_gain = 0
    epochs_run = 0

    for epoch in range(1, config.epochs + 1):
        order = list(range(len(train_products)))
        rng.shuffle(order)
        mined = 0
        if config.use_hard_negatives and epoch >= config.hard_negative_start_epoch:
            order, mined = _hard_negative_order(
                train_products, order, model, processor, config, device, dtype, image_root
            )

        model.train()
        losses: list[float] = []
        for group in _chunks(order, config.effective_batch_size):
            micro_batches = list(_chunks(group, config.physical_batch_size))
            loss = _grad_cache_step(
                model,
                base_model,
                processor,
                train_products,
                micro_batches,
                device,
                dtype,
                image_root,
            )
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss)

        train_loss = sum(losses) / len(losses)
        recall = _recall_at_k_self_query(
            model, processor, val_products, config, device, dtype, image_root, k=5
        )
        epochs_run = epoch
        logger.info(
            "epoch %d/%d: loss=%.4f val_%s=%.4f lr=%.2e hard_negatives=%d",
            epoch,
            config.epochs,
            train_loss,
            config.checkpoint_metric,
            recall,
            scheduler.get_last_lr()[0],
            mined,
        )
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    f"val/{config.checkpoint_metric}": recall,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        if recall > best_metric:
            best_metric = recall
            best_path = Path(config.output_dir) / run_id / "best"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_path))
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.early_stopping_patience:
                logger.info(
                    "early stopping after %d epochs without a %s gain",
                    epochs_without_gain,
                    config.checkpoint_metric,
                )
                break

    if best_path is None:  # pragma: no cover - only if every epoch raised
        raise RuntimeError("training produced no checkpoint")

    elapsed = time.perf_counter() - started
    trainable, total = trainable_parameters(model)
    append_run_row(
        Path(config.extra.get("runs_csv", DEFAULT_RUNS_CSV)),
        {
            "run_id": run_id,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_by": config.extra.get("run_by", ""),
            "commit": git_commit(),
            "config": f"{config.model_id} dora={config.use_dora}",
            "lora_rank": config.lora_rank,
            "learning_rate": config.learning_rate,
            "effective_batch": config.effective_batch_size,
            "epochs": epochs_run,
            f"{config.checkpoint_metric}": f"{best_metric:.4f}",
            "runtime_s": f"{elapsed:.1f}",
            "notes": (
                f"LoRA/DoRA fine-tune, trainable {trainable}/{total} "
                f"({100.0 * trainable / total:.2f}%), val recall is self-query proxy, "
                f"checkpoint={best_path}"
            ),
        },
    )
    if run is not None:
        run.summary[f"best_val_{config.checkpoint_metric}"] = best_metric
        run.finish()
    logger.info(
        "best val %s=%.4f after %d epoch(s) in %.1fs -> %s",
        config.checkpoint_metric,
        best_metric,
        epochs_run,
        elapsed,
        best_path,
    )
    return best_path


# --- internals --------------------------------------------------------------


def _chunks(items: Sequence[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_dtype(config: TrainConfig) -> Any:
    """bf16 when asked for AND the GPU really supports it.

    The project is specced for a 12 GB RTX 4070 (Ada, native bf16). On older
    Turing cards bf16 matmuls are emulated and both slow and numerically worse
    than fp32, so we fall back and say so instead of honouring the flag blindly.
    """
    import torch

    if not config.bf16:
        return torch.float32
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        logger.warning(
            "bf16 requested but %s has no native bf16 — using fp32",
            torch.cuda.get_device_name(0),
        )
        return torch.float32
    return torch.bfloat16


def _init_wandb(config: TrainConfig, run_id: str) -> Any:
    """Start a W&B run, or return ``None`` if this machine has no usable login."""
    if not config.wandb_project:
        return None
    try:
        import wandb

        wandb.Api()
        return wandb.init(
            project=config.wandb_project,
            name=run_id,
            config={
                "model_id": config.model_id,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "use_dora": config.use_dora,
                "learning_rate": config.learning_rate,
                "effective_batch_size": config.effective_batch_size,
                "physical_batch_size": config.physical_batch_size,
                "epochs": config.epochs,
                "seed": config.seed,
            },
        )
    # Broad on purpose: an offline/expired W&B must not kill a training run.
    except Exception as error:
        logger.warning("W&B disabled (%s: %s)", type(error).__name__, error)
        return None


def _image(product: Product, image_root: Path) -> Any:
    from PIL import Image

    if product.image_path is None:
        raise ValueError(f"product {product.product_id} has no image_path")
    path = Path(product.image_path)
    if not path.is_absolute() and not path.exists():
        path = image_root / product.image_path
    with Image.open(path) as image:
        return image.convert("RGB")


def _prepare_batch(
    processor: Any,
    products: Sequence[Product],
    indices: Sequence[int],
    image_root: Path,
) -> Any:
    chosen = [products[i] for i in indices]
    return processor(
        text=[p.to_text() for p in chosen],
        images=[_image(p, image_root) for p in chosen],
        padding="max_length",
        max_length=MAX_TEXT_LENGTH,
        truncation=True,
        return_tensors="pt",
    )


def _pooled(output: Any) -> Any:
    import torch

    # transformers >= 5 returns a ModelOutput from get_*_features; older versions
    # return the tensor directly.
    return output if isinstance(output, torch.Tensor) else output.pooler_output


def _encode(model: Any, inputs: Any, device: str, dtype: Any) -> tuple[Any, Any]:
    """Embed one micro-batch through both towers, L2-normalized in fp32."""
    import torch

    image_features = _pooled(
        model.get_image_features(pixel_values=inputs["pixel_values"].to(device=device, dtype=dtype))
    )
    text_features = _pooled(
        model.get_text_features(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=(
                inputs["attention_mask"].to(device) if "attention_mask" in inputs else None
            ),
        )
    )
    normalize = torch.nn.functional.normalize
    return normalize(image_features.float(), dim=-1), normalize(text_features.float(), dim=-1)


def _grad_cache_step(
    model: Any,
    base_model: Any,
    processor: Any,
    products: Sequence[Product],
    micro_batches: Sequence[Sequence[int]],
    device: str,
    dtype: Any,
    image_root: Path,
) -> float:
    """One optimizer step's worth of gradient, GradCache-style (two passes).

    Returns the loss of the whole effective batch. Gradients are left on the
    parameters; the caller steps the optimizer.
    """
    import torch

    prepared = [_prepare_batch(processor, products, batch, image_root) for batch in micro_batches]

    image_cache = []
    text_cache = []
    # The RNG state of each micro-batch is kept so that pass 2 replays the exact
    # LoRA dropout masks of pass 1 — otherwise the cached representation
    # gradients belong to a different sampled sub-network than the one being
    # differentiated (the same trick the GradCache reference implementation uses).
    rng_states = []
    with torch.no_grad():
        for inputs in prepared:
            rng_states.append(_rng_state(device))
            image_features, text_features = _encode(model, inputs, device, dtype)
            image_cache.append(image_features)
            text_cache.append(text_features)

    images = torch.cat(image_cache).detach().requires_grad_(True)
    texts = torch.cat(text_cache).detach().requires_grad_(True)
    loss = sigmoid_pairwise_loss(
        images, texts, base_model.logit_scale.exp().float(), base_model.logit_bias.float()
    )
    loss.backward()
    if images.grad is None or texts.grad is None:  # pragma: no cover - defensive
        raise RuntimeError("no representation gradient: the cached pass was detached")
    sizes = [len(batch) for batch in micro_batches]
    image_grads = torch.split(images.grad, sizes)
    text_grads = torch.split(texts.grad, sizes)

    for inputs, image_grad, text_grad, state in zip(
        prepared, image_grads, text_grads, rng_states, strict=True
    ):
        _restore_rng_state(state, device)
        image_features, text_features = _encode(model, inputs, device, dtype)
        torch.autograd.backward([image_features, text_features], [image_grad, text_grad])

    return float(loss.detach())


def _rng_state(device: str) -> tuple[Any, Any]:
    import torch

    cuda_state = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None
    return torch.get_rng_state(), cuda_state


def _restore_rng_state(state: tuple[Any, Any], device: str) -> None:
    import torch

    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _encode_images(
    model: Any,
    processor: Any,
    products: Sequence[Product],
    batch_size: int,
    device: str,
    dtype: Any,
    image_root: Path,
) -> Any:
    import torch

    model.eval()
    rows = []
    with torch.no_grad():
        for chunk in _chunks(list(range(len(products))), batch_size):
            inputs = _prepare_batch(processor, products, chunk, image_root)
            rows.append(_encode(model, inputs, device, dtype)[0])
    return torch.cat(rows)


def _hard_negative_order(
    products: Sequence[Product],
    shuffled: Sequence[int],
    model: Any,
    processor: Any,
    config: TrainConfig,
    device: str,
    dtype: Any,
    image_root: Path,
) -> tuple[list[int], int]:
    """Reorder an epoch so each anchor sits next to its mined hard negatives.

    The sigmoid loss only sees negatives that share a batch, so "using" hard
    negatives means batch composition: an anchor and its same-category nearest
    neighbours land in the same physical batch. Still a permutation of the epoch,
    so every product is seen exactly once.
    """
    embeddings = _encode_images(
        model, processor, products, config.physical_batch_size, device, dtype, image_root
    )
    negatives = mine_hard_negatives(products, embeddings, config)
    index_of = {product.product_id: i for i, product in enumerate(products)}

    used = [False] * len(products)
    order: list[int] = []
    for anchor in shuffled:
        if used[anchor]:
            continue
        used[anchor] = True
        order.append(anchor)
        for negative_id in negatives.get(products[anchor].product_id, []):
            index = index_of[negative_id]
            if not used[index]:
                used[index] = True
                order.append(index)
    return order, len(negatives)


def _recall_at_k_self_query(
    model: Any,
    processor: Any,
    products: Sequence[Product],
    config: TrainConfig,
    device: str,
    dtype: Any,
    image_root: Path,
    k: int,
) -> float:
    """Share of val products whose own image is in the top-k for its own caption.

    A stand-in for the labelled query set, which does not exist at training time
    (see the module docstring): captions repeat within a category, so identical
    captions compete and the number is pessimistic. Only used to pick between
    epochs of the same run.
    """
    import torch

    model.eval()
    image_rows = []
    text_rows = []
    with torch.no_grad():
        for chunk in _chunks(list(range(len(products))), config.physical_batch_size):
            inputs = _prepare_batch(processor, products, chunk, image_root)
            image_features, text_features = _encode(model, inputs, device, dtype)
            image_rows.append(image_features)
            text_rows.append(text_features)
    images = torch.cat(image_rows)
    texts = torch.cat(text_rows)

    top_k = min(k, images.shape[0])
    hits = 0
    for start in range(0, texts.shape[0], _MINING_CHUNK):
        scores = texts[start : start + _MINING_CHUNK] @ images.transpose(0, 1)
        ranked = scores.topk(top_k, dim=1).indices
        gold = torch.arange(start, start + scores.shape[0], device=ranked.device).unsqueeze(1)
        hits += int((ranked == gold).any(dim=1).sum())
    return hits / len(products)
