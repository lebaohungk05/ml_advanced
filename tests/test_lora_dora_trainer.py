"""LoRA/DoRA trainer: loss + hard-negative unit tests, and one real smoke run.

The two unit tests need torch/numpy but no weights, so they run in a couple of
seconds. ``test_train_runs_end_to_end_on_a_tiny_real_slice`` is marked ``slow``
because it fine-tunes the real SigLIP 2 backbone on real Fashionpedia images:

    pytest tests/test_lora_dora_trainer.py -m slow -v -s
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.adapters.training.lora_dora_trainer import (
    TrainConfig,
    mine_hard_negatives,
    sigmoid_pairwise_loss,
    train,
)
from src.core.models import Product

ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT = ROOT / "data" / "raw" / "fashionpedia"
TRAIN_JSON = ROOT / "data" / "processed" / "train.json"


def test_sigmoid_loss_is_lower_when_the_diagonal_is_the_match() -> None:
    torch = pytest.importorskip("torch")

    identity = torch.eye(4)
    swapped = identity.roll(1, dims=0)

    matched = sigmoid_pairwise_loss(identity, identity, torch.tensor(10.0), torch.tensor(-10.0))
    mismatched = sigmoid_pairwise_loss(identity, swapped, torch.tensor(10.0), torch.tensor(-10.0))

    assert matched < mismatched
    # Hand-checked: logits are 0 everywhere, so every pair costs -log(0.5).
    zeroed = sigmoid_pairwise_loss(
        torch.zeros(3, 8), torch.zeros(3, 8), torch.tensor(1.0), torch.tensor(0.0)
    )
    assert zeroed == pytest.approx(0.6931472, abs=1e-6)


def test_sigmoid_loss_gradient_flows_to_both_towers() -> None:
    torch = pytest.importorskip("torch")

    images = torch.randn(4, 8, requires_grad=True)
    texts = torch.randn(4, 8, requires_grad=True)

    sigmoid_pairwise_loss(images, texts, torch.tensor(4.0), torch.tensor(-1.0)).backward()

    assert images.grad is not None and images.grad.abs().sum() > 0
    assert texts.grad is not None and texts.grad.abs().sum() > 0


def _product(product_id: str, category: str) -> Product:
    return Product(product_id=product_id, title=f"item {product_id}", category=category)


def test_hard_negatives_are_the_nearest_neighbours_inside_the_category() -> None:
    products = [
        _product("a1", "váy"),
        _product("a2", "váy"),
        _product("a3", "váy"),
        _product("b1", "quần"),
        _product("b2", "quần"),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.99, 0.14],  # closest to a1
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    config = TrainConfig(hard_negatives_per_sample=1)

    negatives = mine_hard_negatives(products, embeddings, config)

    assert negatives["a1"] == ["a2"]
    assert negatives["a2"] == ["a1"]
    assert negatives["a3"] == ["a2"]
    # never crosses the category boundary
    assert negatives["b1"] == ["b2"]


def test_hard_negatives_degrade_gracefully_in_thin_categories() -> None:
    products = [_product("a1", "váy"), _product("a2", "váy"), _product("solo", "mũ")]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    negatives = mine_hard_negatives(products, embeddings, TrainConfig())

    # Asked for 4 negatives, the category only has one other member.
    assert negatives["a1"] == ["a2"]
    # A category of one yields no entry rather than a cross-category negative.
    assert "solo" not in negatives


def test_mine_hard_negatives_rejects_misaligned_embeddings() -> None:
    with pytest.raises(ValueError, match="embeddings must have shape"):
        mine_hard_negatives([_product("a1", "váy")], [[1.0], [2.0]], TrainConfig())


def _tiny_real_slice(count: int) -> list[Product]:
    with TRAIN_JSON.open(encoding="utf-8") as handle:
        raw: list[dict[str, Any]] = json.load(handle)
    products: list[Product] = []
    for record in raw:
        if len(products) == count:
            break
        if not (IMAGE_ROOT / record["image_path"]).exists():
            continue
        products.append(
            Product(
                product_id=record["product_id"],
                title=record["title"],
                category=record["category"],
                image_path=record["image_path"],
                attributes=record.get("attributes") or {},
            )
        )
    return products


@pytest.mark.slow
def test_train_runs_end_to_end_on_a_tiny_real_slice(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    if not TRAIN_JSON.exists():
        pytest.skip("data/processed/train.json not built (run scripts/prepare_catalog.py)")

    products = _tiny_real_slice(16)
    assert len(products) == 16, "need 16 real products with images on disk"

    config = TrainConfig(
        epochs=1,
        physical_batch_size=4,
        effective_batch_size=8,
        warmup_steps=1,
        use_hard_negatives=False,
        wandb_project=None,
        output_dir=tmp_path / "checkpoints",
        run_id="smoke",
        extra={
            "image_root": str(IMAGE_ROOT),
            # never pollute the real experiment log from a test run
            "runs_csv": str(tmp_path / "runs.csv"),
        },
    )

    checkpoint = train(config, products[:12], products[12:])

    assert checkpoint.exists()
    assert (tmp_path / "runs.csv").read_text(encoding="utf-8").count("\n") == 2
    assert (checkpoint / "adapter_config.json").exists()
    assert (checkpoint / "adapter_model.safetensors").exists()
