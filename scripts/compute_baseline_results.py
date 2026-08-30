"""Merge collected grades + auto-zero labels into RelevanceLabel objects, score
every Sprint 2 baseline system (data/eval/pool_top20.json) with RankingMetrics,
and compute Cohen's kappa on whatever (query, product) pairs already have 2
independent human grades.

Runs on WHATEVER labels exist right now (labeling doesn't need to be 100% done
first -- see docstring on RankingMetrics._check_coverage: unlabelled queries
just score as a documented 0, not a crash). Re-run this after Hùng finishes his
share of the labeling to get the final numbers.

Usage:
    python scripts/compute_baseline_results.py
"""

from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.adapters.metrics.ranking_metrics import RankingMetrics, cohens_kappa  # noqa: E402
from src.core.models import Product, RelevanceLabel, SearchResult  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"


def load_labels() -> list[RelevanceLabel]:
    labels: list[RelevanceLabel] = []

    with open(EVAL_DIR / "auto_zero_labels.json", encoding="utf-8") as f:
        for row in json.load(f):
            labels.append(
                RelevanceLabel(
                    query_id=str(row["query_id"]),
                    product_id=str(row["product_id"]),
                    grade=int(row["grade"]),
                    annotator=row["graded_by"],
                )
            )

    for path in sorted(EVAL_DIR.glob("grades_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                labels.append(
                    RelevanceLabel(
                        query_id=str(rec["query_id"]),
                        product_id=str(rec["product_id"]),
                        grade=int(rec["grade"]),
                        annotator=rec["annotator"],
                    )
                )
    return labels


def load_runs() -> dict[str, dict[str, list[SearchResult]]]:
    with open(EVAL_DIR / "pool_top20.json", encoding="utf-8") as f:
        raw = json.load(f)

    runs: dict[str, dict[str, list[SearchResult]]] = {}
    for system_name, run in raw.items():
        converted: dict[str, list[SearchResult]] = {}
        for query_id, hits in run.items():
            converted[query_id] = [
                SearchResult(
                    product=Product(
                        product_id=hit["product_id"], title=hit["product_id"], category="n/a"
                    ),
                    score=hit["score"],
                    rank=hit["rank"],
                )
                for hit in hits
            ]
        runs[system_name] = converted
    return runs


def report_kappa(labels: list[RelevanceLabel]) -> None:
    by_pair: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for label in labels:
        if label.annotator is None or label.annotator.startswith("auto-filter"):
            continue
        by_pair[(label.query_id, label.product_id)][label.annotator] = label.grade

    doubly_graded = {k: v for k, v in by_pair.items() if len(v) >= 2}
    print(f"\nDòng có >=2 người chấm (bỏ qua auto-filter): {len(doubly_graded)}")
    if not doubly_graded:
        print("Chưa đủ dữ liệu để tính kappa.")
        return

    # kappa is pairwise -- compute for every pair of annotators that overlap
    pair_grades: dict[tuple[str, str], tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for grades_by_annotator in doubly_graded.values():
        names = sorted(grades_by_annotator)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pair_grades[(a, b)][0].append(grades_by_annotator[a])
                pair_grades[(a, b)][1].append(grades_by_annotator[b])

    for (a, b), (ga, gb) in sorted(pair_grades.items()):
        if len(ga) < 2:
            continue
        kappa = cohens_kappa(ga, gb)
        print(f"  {a} vs {b}: kappa = {kappa:.3f} (trên {len(ga)} dòng chung)")


def main() -> None:
    labels = load_labels()
    print(f"Tổng số nhãn (đã gộp mọi người + auto-zero): {len(labels)}")

    runs = load_runs()
    metrics = RankingMetrics()

    header = (
        f"{'Hệ thống':<14}{'Recall@1':>10}{'Recall@5':>10}"
        f"{'Recall@10':>11}{'MRR':>8}{'nDCG@10':>10}"
    )
    print(f"\n{header}")
    for name, run in runs.items():
        scores = metrics.summary(run, labels)
        print(
            f"{name:<14}"
            f"{scores['recall@1']:>10.4f}"
            f"{scores['recall@5']:>10.4f}"
            f"{scores['recall@10']:>11.4f}"
            f"{scores['mrr']:>8.4f}"
            f"{scores['ndcg@10']:>10.4f}"
        )

    report_kappa(labels)


if __name__ == "__main__":
    main()
