"""One-off data-prep script (Sprint 2): dedup + split + build Product catalog.

Builds data/processed/{train,val,test}.json from data/raw/fashionpedia/*_manifest.json
+ data/category_mapping.json. Not part of src/ on purpose — same reasoning as
scripts/download_fashionpedia.py.

Caption limitation (see docs/DeCuong Mục 6.1): this HF mirror of Fashionpedia
keeps category + bbox only, not the 294 fine-grained attributes from the paper.
Product.to_text() is therefore built from detected category names only (no
color/material) — coarser than the attribute-template caption originally
planned.
"""

from __future__ import annotations

import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

import imagehash
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "fashionpedia"
OUT = ROOT / "data" / "processed"
SEED = 42


def load_category_mapping() -> dict[int, dict]:
    with open(ROOT / "data" / "category_mapping.json", encoding="utf-8") as f:
        data = json.load(f)
    return {c["id"] - 1: c for c in data["categories"]}  # dataset ids are 0-indexed


def build_product(record: dict, cat_map: dict[int, dict], split_prefix: str) -> dict | None:
    objects = record.get("objects") or {}
    cat_ids = objects.get("category", [])
    if not cat_ids:
        return None

    primary_group_order = ["main_garment", "accessory", "garment_part_or_detail"]
    by_group: dict[str, list[str]] = {g: [] for g in primary_group_order}
    for cid in cat_ids:
        c = cat_map.get(cid)
        if c is None:
            continue
        vi = c["vi"]
        if vi not in by_group[c["group"]]:
            by_group[c["group"]].append(vi)

    # Primary category: prefer a main_garment; fall back to accessory.
    primary = (
        by_group["main_garment"]
        or by_group["accessory"]
        or by_group["garment_part_or_detail"]
    )
    if not primary:
        return None
    category = primary[0]

    title_parts = by_group["main_garment"] + by_group["accessory"]
    title = ", ".join(title_parts) if title_parts else category

    attributes = {}
    if by_group["garment_part_or_detail"]:
        attributes["chi_tiet"] = ", ".join(by_group["garment_part_or_detail"])

    return {
        "product_id": f"{split_prefix}-{record['image_id']}",
        "title": title,
        "category": category,
        "image_path": record["image_file"],
        "attributes": attributes,
    }


def phash_of(image_path: Path) -> str | None:
    try:
        with Image.open(image_path) as img:
            return str(imagehash.phash(img.convert("RGB")))
    except Exception:
        return None


def main() -> None:
    cat_map = load_category_mapping()
    OUT.mkdir(parents=True, exist_ok=True)

    all_products: list[dict] = []
    for split_file, prefix in [("train_manifest.json", "fp"), ("val_manifest.json", "fp")]:
        path = RAW / split_file
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            p = build_product(r, cat_map, prefix)
            if p is not None:
                all_products.append(p)

    print(f"Sản phẩm hợp lệ (có >=1 danh mục nhận dạng được): {len(all_products)}")

    print("Đang kiểm tra ảnh trùng lặp (perceptual hash)...")
    seen_hashes: dict[str, str] = {}
    deduped: list[dict] = []
    dup_count = 0
    for i, p in enumerate(all_products):
        h = phash_of(RAW / p["image_path"])
        if h is None:
            continue
        if h in seen_hashes:
            dup_count += 1
            continue
        seen_hashes[h] = p["product_id"]
        deduped.append(p)
        if (i + 1) % 5000 == 0:
            print(f"  ... {i + 1}/{len(all_products)}")

    print(f"Loại {dup_count} ảnh trùng lặp -> còn {len(deduped)} sản phẩm")

    rng = random.Random(SEED)
    rng.shuffle(deduped)
    n = len(deduped)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    splits = {
        "train": deduped[:n_train],
        "val": deduped[n_train : n_train + n_val],
        "test": deduped[n_train + n_val :],
    }

    for name, items in splits.items():
        with open(OUT / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
        cat_counts = Counter(p["category"] for p in items)
        print(f"[{name}] {len(items)} sản phẩm — top 5 danh mục: {cat_counts.most_common(5)}")


if __name__ == "__main__":
    main()
