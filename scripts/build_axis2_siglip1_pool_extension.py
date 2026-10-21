"""Extend the relevance pool to the top-10 of ``axis2_no_hard_negatives`` and ``siglip1``.

Same problem as scripts/build_ft_pool_extension.py, one round later: the pool in
data/eval/pool_top20.json was built from the top-20 of the 4 Sprint 2 baselines,
then extended once for ``siglip2_lora``. The axis-2 ablation checkpoint and the
zero-shot ``siglip1`` baseline retrieve mostly other products, so most of their
top-10 pairs carry no label and score as grade 0 — their Recall/nDCG are
lower bounds, not values, and cannot go in the headline table.

This script covers BOTH systems in a single labeling round. It reuses
``build_ft_pool_extension.build_extension`` unchanged (same category-mismatch
auto-zero filter, same row shape, same blind-pooling shuffle) and feeds the
first system's freshly-collected pairs back into the ``labelled`` set before
running the second, so a pair wanted by both systems is asked for once.

Outputs:
- data/eval/relevance_template_axis2_siglip1.json -- rows needing a human grade
- data/eval/auto_zero_labels_axis2_siglip1.json   -- pairs pre-graded 0 by the filter

Usage:
    python scripts/build_axis2_siglip1_pool_extension.py
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

from scripts.build_ft_pool_extension import (  # noqa: E402
    CATALOG_PATH,
    EVAL_DIR,
    POOL_PATH,
    QUERIES_PATH,
    SEED,
    TOP_K,
    build_extension,
    load_labelled_pairs,
)
from scripts.category_filter import load_garment_names  # noqa: E402

OUT_NEEDS_HUMAN = EVAL_DIR / "relevance_template_axis2_siglip1.json"
OUT_AUTO_ZERO = EVAL_DIR / "auto_zero_labels_axis2_siglip1.json"

SYSTEMS = ("axis2_no_hard_negatives", "siglip1")

Pair = tuple[str, str]
Row = dict[str, Any]


@dataclass(frozen=True)
class CombinedCounts:
    total_pairs: int
    already_labelled: int
    auto_zeroed: int
    needs_human: int
    missing_from_catalog: int
    per_system_new: dict[str, int]


def unlabelled_pairs(
    pool: dict[str, dict[str, list[Row]]],
    queries: dict[str, Row],
    labelled: set[Pair],
    systems: tuple[str, ...] = SYSTEMS,
    top_k: int = TOP_K,
) -> set[Pair]:
    """Union of the systems' top-k pairs that carry no label anywhere."""
    pairs: set[Pair] = set()
    for system in systems:
        run = pool[system]
        for qid in queries:
            for hit in run.get(qid, [])[:top_k]:
                pair = (qid, str(hit["product_id"]))
                if pair not in labelled:
                    pairs.add(pair)
    return pairs


def build_combined_extension(
    pool: dict[str, dict[str, list[Row]]],
    queries: dict[str, Row],
    catalog: dict[str, Row],
    labelled: set[Pair],
    garment_names: list[str],
    systems: tuple[str, ...] = SYSTEMS,
    top_k: int = TOP_K,
    seed: int = SEED,
) -> tuple[list[Row], list[Row], CombinedCounts]:
    """One pool-extension round over several systems, deduped across them."""
    seen = set(labelled)
    needs_human: list[Row] = []
    auto_zero: list[Row] = []
    total_pairs = 0
    already_labelled = 0
    missing_from_catalog = 0
    per_system_new: dict[str, int] = {}

    for system in systems:
        human, zeros, counts = build_extension(
            pool, queries, catalog, seen, garment_names, system=system,
            top_k=top_k, seed=seed,
        )
        needs_human.extend(human)
        auto_zero.extend(zeros)
        total_pairs += counts.total_pairs
        already_labelled += counts.already_labelled
        missing_from_catalog += counts.missing_from_catalog
        per_system_new[system] = counts.needs_human + counts.auto_zeroed
        seen |= {(str(r["query_id"]), str(r["product_id"])) for r in human + zeros}

    needs_human = _shuffle_within_query(needs_human, queries, seed)

    combined = CombinedCounts(
        total_pairs=total_pairs,
        already_labelled=already_labelled,
        auto_zeroed=len(auto_zero),
        needs_human=len(needs_human),
        missing_from_catalog=missing_from_catalog,
        per_system_new=per_system_new,
    )
    return needs_human, auto_zero, combined


def _shuffle_within_query(rows: list[Row], queries: dict[str, Row], seed: int) -> list[Row]:
    """Regroup per query in query order and reshuffle, so rank stays unreadable."""
    rng = random.Random(seed)
    by_query: dict[str, list[Row]] = {}
    for row in rows:
        by_query.setdefault(str(row["query_id"]), []).append(row)
    out: list[Row] = []
    for qid in queries:
        group = by_query.get(qid, [])
        rng.shuffle(group)
        out.extend(group)
    return out


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = {str(q["id"]): q for q in json.load(f)["queries"]}
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = {p["product_id"]: p for p in json.load(f)}

    labelled = load_labelled_pairs()
    expected = unlabelled_pairs(pool, queries, labelled)

    needs_human, auto_zero, counts = build_combined_extension(
        pool, queries, catalog, labelled, load_garment_names()
    )

    with open(OUT_NEEDS_HUMAN, "w", encoding="utf-8") as f:
        json.dump(needs_human, f, ensure_ascii=False, indent=1)
    with open(OUT_AUTO_ZERO, "w", encoding="utf-8") as f:
        json.dump(auto_zero, f, ensure_ascii=False, indent=1)

    produced = {(str(r["query_id"]), str(r["product_id"])) for r in needs_human + auto_zero}

    print(f"Mở rộng pool cho {', '.join(SYSTEMS)} đến top-{TOP_K}:")
    print(f"  Tổng cặp (truy vấn, sản phẩm) 2 hệ thống      : {counts.total_pairs}")
    print(f"  Đã có nhãn / đã hỏi ở hệ thống trước (bỏ qua) : {counts.already_labelled}")
    print(f"  Cặp CHƯA có nhãn (hợp của 2 hệ thống)         : {len(expected)}")
    for system, n in counts.per_system_new.items():
        print(f"    - mới thêm bởi {system:<24}: {n}")
    print(f"  Auto-zero (sai loại rõ ràng)                  : {counts.auto_zeroed}"
          f" -> {OUT_AUTO_ZERO}")
    print(f"  Cần người chấm thật                           : {counts.needs_human}"
          f" -> {OUT_NEEDS_HUMAN}")
    print(f"  Kiểm tra: {counts.auto_zeroed} + {counts.needs_human} = "
          f"{counts.auto_zeroed + counts.needs_human} / {len(expected)} cặp chưa nhãn")
    if counts.missing_from_catalog:
        print(f"  CẢNH BÁO: {counts.missing_from_catalog} sản phẩm không có trong catalog")

    dropped = expected - produced
    if dropped:
        print(f"  CẢNH BÁO: {len(dropped)} cặp bị bỏ rơi, ví dụ {sorted(dropped)[:5]}")
    if produced & labelled:
        print(f"  CẢNH BÁO: {len(produced & labelled)} cặp đã có nhãn bị hỏi lại")


if __name__ == "__main__":
    main()
