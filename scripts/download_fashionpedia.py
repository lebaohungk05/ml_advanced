"""One-off script: download Fashionpedia (CC BY 4.0) from Hugging Face into data/raw/.

Not part of the hexagonal src/ tree on purpose — this is a data-prep utility run
once per machine, not a runtime adapter. See docs/DeCuong Mục 3.2 for why this
dataset was chosen.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from datasets import load_dataset  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "fashionpedia"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("detection-datasets/fashionpedia")

    manifest = {}
    for split_name, split in ds.items():
        split_dir = OUT_DIR / split_name / "images"
        split_dir.mkdir(parents=True, exist_ok=True)
        n = len(split)
        print(f"[{split_name}] {n} sản phẩm — bắt đầu lưu ảnh + nhãn")
        records = []
        for i, row in enumerate(split):
            img = row["image"]
            img_path = split_dir / f"{i:06d}.jpg"
            if not img_path.exists():
                img.convert("RGB").save(img_path, quality=90)
            records.append(
                {
                    "image_id": row.get("image_id", i),
                    "image_file": str(img_path.relative_to(OUT_DIR)),
                    "width": row.get("width"),
                    "height": row.get("height"),
                    "objects": row.get("objects"),
                }
            )
            if (i + 1) % 2000 == 0:
                print(f"  ... {i + 1}/{n}")
        with open(OUT_DIR / f"{split_name}_manifest.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        manifest[split_name] = n
        print(f"[{split_name}] xong — {n} ảnh lưu tại {split_dir}")

    with open(OUT_DIR / "SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("Xong toàn bộ:", manifest)


if __name__ == "__main__":
    main()
