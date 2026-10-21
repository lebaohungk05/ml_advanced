"""Build the reduced labeling pool: top-10 (not top-20) + auto-zero filter.

Two cuts applied, both disclosed in the report (DeCuong Mục 6.2's own fallback
plan, executed proactively instead of after running out of time):
1. Pool depth top-20 -> top-10 per system before dedup.
2. Auto-zero: a (query, product) pair where the query names a specific garment
   type (group A/B/C, via a keyword match against data/category_mapping.json)
   and the candidate's category clearly does not match gets grade=0 WITHOUT a
   human looking at the image. This is a transparent, disclosed use of
   already-existing catalog metadata (not an AI judgment call) — it only
   short-circuits the "obviously wrong type" cases; every genuinely ambiguous
   pair (including all of group D, which has no specific garment keyword)
   still goes to human blind labeling.

Outputs:
- data/eval/relevance_template.json   -- rows that STILL need a human grade
  (grade: null), this is what scripts/labeling_server.py serves.
- data/eval/auto_zero_labels.json     -- rows already graded 0 by the filter
  above, to be merged back in before computing Recall/MRR/nDCG later.
"""

from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.category_filter import (  # noqa: E402
    garment_keyword_in_query,
    is_category_mismatch,
    load_garment_names,
)

POOL_PATH = ROOT / "data" / "eval" / "pool_top20.json"
QUERIES_PATH = ROOT / "data" / "eval" / "queries.json"
CATALOG_PATH = ROOT / "data" / "processed" / "test.json"
OUT_NEEDS_HUMAN = ROOT / "data" / "eval" / "relevance_template.json"
OUT_AUTO_ZERO = ROOT / "data" / "eval" / "auto_zero_labels.json"
TOP_K = 10
SEED = 42


def main() -> None:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = {str(q["id"]): q for q in json.load(f)["queries"]}
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = {p["product_id"]: p for p in json.load(f)}
    garment_names = load_garment_names()

    rng = random.Random(SEED)
    systems = sorted(pool.keys())
    needs_human: list[dict] = []
    auto_zero: list[dict] = []

    for qid, q in queries.items():
        seen: set[str] = set()
        for system in systems:
            for hit in pool.get(system, {}).get(qid, [])[:TOP_K]:
                seen.add(hit["product_id"])

        candidates = list(seen)
        rng.shuffle(candidates)

        kw = None if q["group"] == "D" else garment_keyword_in_query(q["text"], garment_names)

        for pid in candidates:
            product = catalog.get(pid)
            if product is None:
                continue
            row = {
                "query_id": qid,
                "query_text": q["text"],
                "query_group": q["group"],
                "product_id": pid,
                "image_path": product["image_path"],
                "category": product["category"],
            }
            if is_category_mismatch(kw, product["category"]):
                row["grade"] = 0
                row["graded_by"] = "auto-filter:category-mismatch"
                auto_zero.append(row)
            else:
                row["grade"] = None
                row["graded_by"] = None
                needs_human.append(row)

    with open(OUT_NEEDS_HUMAN, "w", encoding="utf-8") as f:
        json.dump(needs_human, f, ensure_ascii=False, indent=1)
    with open(OUT_AUTO_ZERO, "w", encoding="utf-8") as f:
        json.dump(auto_zero, f, ensure_ascii=False, indent=1)

    print(f"top-{TOP_K}, sau khi gộp {len(systems)} hệ thống ({systems}):")
    print(f"  Cần người chấm thật : {len(needs_human)} dòng -> {OUT_NEEDS_HUMAN}")
    print(f"  Auto-zero (sai loại rõ ràng, đã loại) : {len(auto_zero)} dòng -> {OUT_AUTO_ZERO}")


if __name__ == "__main__":
    main()
