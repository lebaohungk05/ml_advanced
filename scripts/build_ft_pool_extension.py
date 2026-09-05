"""Extend the relevance pool to the fine-tuned system's top-10.

Why this exists: data/eval/pool_top20.json's pool was built from the top-20 of
the 4 Sprint 2 BASELINE systems only (bm25, clip, siglip2, visiglip_ot), so all
4 read 0 unlabelled hits at every K. The Sprint 3 fine-tuned system
(``siglip2_lora``) retrieves almost entirely different products — 1245 of its
1550 top-10 (query, product) pairs carry no label at all and are currently
scored as grade 0. That makes its measured Recall@1 a hard LOWER BOUND, not an
estimate, and it cannot go in the headline comparison table as-is.

This script collects those unlabelled pairs so they can be graded, which makes
Recall@1/5/10 and nDCG@10 defensible for all 5 systems.

Two properties matter and are covered by tests/test_ft_pool_extension.py:

1. The SAME category-mismatch auto-zero filter as scripts/build_final_pool.py
   (via scripts/category_filter.py) is applied. The baseline pool auto-zeroed
   category mismatches; if the fine-tuned pool did not, the comparison would be
   unfair to the baselines.
2. Nothing is silently dropped: the auto-zero output and the human template are
   disjoint and together cover every unlabelled top-10 pair. A pair falling
   through the gap would quietly score as 0 and corrupt the final metrics.

Rows handed to humans carry no system-identifying field and are shuffled within
each query with a fixed seed, so a grader cannot infer the retrieval rank
(blind pooling, DeCuong Mục 3.4).

Outputs:
- data/eval/relevance_template_ft.json -- rows still needing a human grade
- data/eval/auto_zero_labels_ft.json   -- pairs pre-graded 0 by the filter

Usage:
    python scripts/build_ft_pool_extension.py
"""

from __future__ import annotations

import io
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.category_filter import (  # noqa: E402
    garment_keyword_in_query,
    is_category_mismatch,
    load_garment_names,
)

EVAL_DIR = ROOT / "data" / "eval"
POOL_PATH = EVAL_DIR / "pool_top20.json"
QUERIES_PATH = EVAL_DIR / "queries.json"
CATALOG_PATH = ROOT / "data" / "processed" / "test.json"
OUT_NEEDS_HUMAN = EVAL_DIR / "relevance_template_ft.json"
OUT_AUTO_ZERO = EVAL_DIR / "auto_zero_labels_ft.json"

SYSTEM = "siglip2_lora"
TOP_K = 10
SEED = 42

Pair = tuple[str, str]
Row = dict[str, Any]


@dataclass(frozen=True)
class ExtensionCounts:
    total_pairs: int
    already_labelled: int
    auto_zeroed: int
    needs_human: int
    missing_from_catalog: int


def load_labelled_pairs(eval_dir: Path = EVAL_DIR) -> set[Pair]:
    """Every (query_id, product_id) that already carries a grade from any source.

    Globs both the auto-zero label files and every collected grades_*.jsonl so
    that re-running this script after another round of labeling never re-asks
    for a grade somebody already gave.
    """
    labelled: set[Pair] = set()

    for path in sorted(eval_dir.glob("auto_zero_labels*.json")):
        with open(path, encoding="utf-8") as f:
            for row in json.load(f):
                labelled.add((str(row["query_id"]), str(row["product_id"])))

    for path in sorted(eval_dir.glob("grades_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                labelled.add((str(rec["query_id"]), str(rec["product_id"])))

    return labelled


def build_extension(
    pool: dict[str, dict[str, list[Row]]],
    queries: dict[str, Row],
    catalog: dict[str, Row],
    labelled: set[Pair],
    garment_names: list[str],
    system: str = SYSTEM,
    top_k: int = TOP_K,
    seed: int = SEED,
) -> tuple[list[Row], list[Row], ExtensionCounts]:
    """Split the fine-tuned system's unlabelled top-k pairs into auto-zero vs human."""
    run = pool[system]
    rng = random.Random(seed)
    needs_human: list[Row] = []
    auto_zero: list[Row] = []
    total_pairs = 0
    already_labelled = 0
    missing_from_catalog = 0

    for qid, q in queries.items():
        hits = run.get(qid, [])[:top_k]
        pids = list(dict.fromkeys(str(hit["product_id"]) for hit in hits))
        total_pairs += len(pids)

        pending = []
        for pid in pids:
            if (qid, pid) in labelled:
                already_labelled += 1
                continue
            pending.append(pid)

        # Randomised within the query so the grader cannot read off the rank.
        rng.shuffle(pending)

        kw = None if q["group"] == "D" else garment_keyword_in_query(q["text"], garment_names)

        for pid in pending:
            product = catalog.get(pid)
            if product is None:
                missing_from_catalog += 1
                continue
            row: Row = {
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

    counts = ExtensionCounts(
        total_pairs=total_pairs,
        already_labelled=already_labelled,
        auto_zeroed=len(auto_zero),
        needs_human=len(needs_human),
        missing_from_catalog=missing_from_catalog,
    )
    return needs_human, auto_zero, counts


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = {str(q["id"]): q for q in json.load(f)["queries"]}
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = {p["product_id"]: p for p in json.load(f)}

    needs_human, auto_zero, counts = build_extension(
        pool, queries, catalog, load_labelled_pairs(), load_garment_names()
    )

    with open(OUT_NEEDS_HUMAN, "w", encoding="utf-8") as f:
        json.dump(needs_human, f, ensure_ascii=False, indent=1)
    with open(OUT_AUTO_ZERO, "w", encoding="utf-8") as f:
        json.dump(auto_zero, f, ensure_ascii=False, indent=1)

    print(f"Mở rộng pool cho '{SYSTEM}' đến top-{TOP_K}:")
    print(f"  Tổng cặp (truy vấn, sản phẩm) trong top-{TOP_K} : {counts.total_pairs}")
    print(f"  Đã có nhãn từ trước (bỏ qua)                    : {counts.already_labelled}")
    print(f"  Auto-zero (sai loại rõ ràng)                    : {counts.auto_zeroed}"
          f" -> {OUT_AUTO_ZERO}")
    print(f"  Cần người chấm thật                             : {counts.needs_human}"
          f" -> {OUT_NEEDS_HUMAN}")
    if counts.missing_from_catalog:
        print(f"  CẢNH BÁO: {counts.missing_from_catalog} sản phẩm không có trong catalog")


if __name__ == "__main__":
    main()
