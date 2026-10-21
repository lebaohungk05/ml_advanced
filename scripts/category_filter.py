"""The category-mismatch auto-zero filter, shared by every pool builder.

Extracted verbatim from scripts/build_final_pool.py so that the fine-tuned
system's pool extension (scripts/build_ft_pool_extension.py) is filtered by
exactly the same rule as the 4-baseline pool. If the two pools were filtered
differently the headline comparison table would not be apples-to-apples.

Deliberately free of prints and of the UTF-8 stdout wrapper: this module is
imported by tests, and rebinding sys.stdout at import time breaks pytest's
capture.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATEGORY_MAPPING_PATH = ROOT / "data" / "category_mapping.json"


def load_garment_names(path: Path = CATEGORY_MAPPING_PATH) -> list[str]:
    with open(path, encoding="utf-8") as f:
        cats = json.load(f)["categories"]
    return [
        c["vi"].split("/")[0].strip().split("(")[0].strip()
        for c in cats
        if c["group"] in ("main_garment", "accessory")
    ]


def garment_keyword_in_query(qtext: str, garment_names: list[str]) -> str | None:
    qtext = qtext.lower()
    for name in garment_names:
        for word in name.lower().split():
            if len(word) > 2 and word in qtext:
                return name
    return None


def is_category_mismatch(keyword: str | None, category: str) -> bool:
    """True when the pair can be auto-graded 0 without a human looking at it."""
    if keyword is None:
        return False
    return keyword.lower().split()[0] not in category.lower()
