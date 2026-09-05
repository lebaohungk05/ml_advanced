"""Loading ``data/processed/*.json`` for training runs.

Separate from ``src.index``'s loaders on purpose: training additionally needs the
image to exist ON DISK (the trainer opens every image), so records whose
``image_path`` is missing are dropped instead of raising. Shared by
``scripts/train_siglip2_lora.py`` and ``scripts/run_ablation.py`` so both runs
see the exact same product lists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.models import Product


def load_training_products(path: Path, image_root: Path) -> tuple[list[Product], int]:
    """Products from a prepared split, dropping those with no image on disk.

    Returns ``(products, skipped)`` so the caller can report the drop count.
    """
    with path.open(encoding="utf-8") as handle:
        raw: list[dict[str, Any]] = json.load(handle)

    products: list[Product] = []
    skipped = 0
    for record in raw:
        image_path = record.get("image_path")
        if not image_path or not (image_root / image_path).exists():
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
    return products, skipped
