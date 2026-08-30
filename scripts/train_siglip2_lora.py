"""Full SigLIP2 + LoRA/DoRA fine-tuning run (Sprint 3) on the real dataset.

Loads data/processed/train.json + val.json (built by prepare_catalog.py) as
Product lists and calls src.adapters.training.lora_dora_trainer.train() with
the agreed default hyperparameters (DeCuong Mục 3.3) -- nothing overridden
here except output_dir/run_id, so the resulting runs.csv row and checkpoint
are directly comparable to every other run using the same config.

Usage:
    python scripts/train_siglip2_lora.py
    python scripts/train_siglip2_lora.py --run-id hung-run1 \
        --wandb-project fashion-multimodal-search
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.adapters.training.lora_dora_trainer import TrainConfig, train  # noqa: E402
from src.core.models import Product  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
IMAGE_ROOT = ROOT / "data" / "raw" / "fashionpedia"


def load_products(path: Path) -> list[Product]:
    with open(path, encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)

    products: list[Product] = []
    skipped = 0
    for record in raw:
        image_path = record.get("image_path")
        if not image_path or not (IMAGE_ROOT / image_path).exists():
            skipped += 1
            continue
        products.append(
            Product(
                product_id=record["product_id"],
                title=record["title"],
                category=record["category"],
                image_path=image_path,
                attributes=record.get("attributes") or {},
            )
        )
    if skipped:
        print(f"[{path.name}] bỏ qua {skipped} sản phẩm thiếu ảnh trên đĩa")
    return products


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--wandb-project", default="fashion-multimodal-search")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()

    if not (PROCESSED / "train.json").exists():
        raise SystemExit(
            "data/processed/train.json chưa có -- chạy trước:\n"
            "  python scripts/download_fashionpedia.py\n"
            "  python scripts/prepare_catalog.py"
        )

    train_products = load_products(PROCESSED / "train.json")
    val_products = load_products(PROCESSED / "val.json")
    print(f"train: {len(train_products)} sản phẩm | val: {len(val_products)} sản phẩm")

    config = TrainConfig(
        run_id=args.run_id,
        output_dir=args.output_dir,
        wandb_project=None if args.no_wandb else args.wandb_project,
        extra={"image_root": str(IMAGE_ROOT)},
    )

    checkpoint = train(config, train_products, val_products)
    print(f"\nCheckpoint tốt nhất (theo Recall@5 trên val): {checkpoint}")


if __name__ == "__main__":
    main()
