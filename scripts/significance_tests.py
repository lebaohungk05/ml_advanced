"""Sprint 4 deliverable: paired-bootstrap significance tests between systems
(DeCuong Mục 5.5).

Answers "is the gap in the accuracy table real, or could it be sampling noise on
155 queries?". Uses ``paired_bootstrap_test`` from
``src/adapters/metrics/ranking_metrics.py`` — there is exactly one bootstrap
implementation in this repo and this script is only the per-query score
extraction + reporting around it.

The test needs PER-QUERY scores of the same metric, aligned by query id:
* Recall@10 contribution — 0 or 1 per query (hit rate, DeCuong Mục 5.1's
  definition, NOT BEIR's fraction-of-relevant),
* reciprocal rank — 1/rank of the first hit with grade >= 1 over the full
  ranked list, 0 for a query with no hit (the per-query term of MRR).
Both mean out to exactly the aggregates printed by
``scripts/compute_baseline_results.py``, which this script asserts before
running any test — a second, subtly different score extraction would silently
change the p-values.

siglip2_lora was excluded while its top-10 was mostly unlabelled (the pool came
from the 4 baselines' top-20 only), because a bootstrap resamples queries and
cannot repair a systematic labelling gap. The pool extension has since been
graded by 2 annotators — ``unlabelled_hits_at_k`` now reports 0/1550 at K=10 for
every system — so it is included and ``EXCLUDED`` is empty. Re-check that table
before trusting any p-value here.

Usage:
    python -m scripts.significance_tests --limit 20   # fast smoke run
    python -m scripts.significance_tests             # full run, writes JSON
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapters.metrics.ranking_metrics import (  # noqa: E402
    RankingMetrics,
    _grade_lookup,  # same 0/1/2 averaging over annotators as the metrics table
    paired_bootstrap_test,
)
from src.core.models import RelevanceLabel, SearchResult  # noqa: E402

OUT_PATH = ROOT / "data" / "eval" / "significance_results.json"
SYSTEMS = ("bm25", "clip", "siglip2", "visiglip_ot", "siglip2_lora")
EXCLUDED: dict[str, str] = {}
MIN_RELEVANT_GRADE = 1
RECALL_K = 10

Run = Mapping[str, Sequence[SearchResult]]
Grades = Mapping[tuple[str, str], float]


def force_utf8_stdout() -> None:
    """Make the Windows console print Vietnamese instead of raising.

    ``reconfigure``, NOT a fresh ``io.TextIOWrapper`` around ``sys.stdout.buffer``
    like the older scripts do: wrapping a stdout that a previously imported
    script already wrapped closes the inner wrapper as soon as it is collected,
    and every later print dies with "I/O operation on closed file".
    """
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")


def sorted_query_ids(run: Run) -> list[str]:
    """Query ids in a stable order (numeric when the ids are numeric strings)."""
    ids = list(run)
    if all(qid.isdigit() for qid in ids):
        return sorted(ids, key=int)
    return sorted(ids)


def per_query_recall_at_k(
    run: Run, grades: Grades, query_ids: Sequence[str], k: int = RECALL_K
) -> list[float]:
    """One 0/1 hit-rate contribution per query id, in ``query_ids`` order.

    1.0 when any of the query's top-``k`` hits has grade >= 1. A query missing
    from ``run`` contributes 0.0 (the system returned nothing for it), matching
    how ``RankingMetrics.recall_at_k`` treats an empty result list.
    """
    scores: list[float] = []
    for query_id in query_ids:
        hits = run.get(query_id, [])[:k]
        hit = any(
            grades.get((query_id, hit.product_id), 0.0) >= MIN_RELEVANT_GRADE for hit in hits
        )
        scores.append(1.0 if hit else 0.0)
    return scores


def per_query_reciprocal_rank(run: Run, grades: Grades, query_ids: Sequence[str]) -> list[float]:
    """1/rank of the first relevant hit per query id, 0.0 when there is none.

    Rank is the 1-indexed position in the full ranked list, not truncated to any
    K — same definition as ``RankingMetrics.mrr``, whose value is the mean of
    this list.
    """
    scores: list[float] = []
    for query_id in query_ids:
        reciprocal = 0.0
        for hit in run.get(query_id, []):
            if grades.get((query_id, hit.product_id), 0.0) >= MIN_RELEVANT_GRADE:
                reciprocal = 1.0 / hit.rank
                break
        scores.append(reciprocal)
    return scores


def check_against_aggregate(
    per_query: Mapping[str, list[float]], run: Run, labels: Sequence[RelevanceLabel]
) -> None:
    """Fail loudly if the per-query scores don't average to the headline metrics.

    The whole point of the bootstrap is that it decomposes the SAME number the
    accuracy table reports. If this drifts, the p-values describe a metric
    nobody published.
    """
    metrics = RankingMetrics(min_relevant_grade=MIN_RELEVANT_GRADE)
    expected = {
        "recall@1": metrics.recall_at_k(run, labels, k=1),
        f"recall@{RECALL_K}": metrics.recall_at_k(run, labels, k=RECALL_K),
        "mrr": metrics.mrr(run, labels),
    }
    for metric, aggregate in expected.items():
        scores = per_query[metric]
        mean = sum(scores) / len(scores)
        if abs(mean - aggregate) > 1e-9:
            raise AssertionError(
                f"{metric}: mean of per-query scores {mean:.6f} != RankingMetrics "
                f"{aggregate:.6f} — per-query extraction and the metrics module disagree"
            )


def run_pairwise_tests(
    per_query: Mapping[str, Mapping[str, list[float]]],
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """p-value + observed mean difference for every ordered pair of systems."""
    matrix: dict[str, dict[str, dict[str, float]]] = {a: {} for a in per_query}
    for a, b in itertools.permutations(per_query, 2):
        scores_a = per_query[a][metric]
        scores_b = per_query[b][metric]
        p_value = paired_bootstrap_test(scores_a, scores_b, iterations=iterations, seed=seed)
        mean_diff = sum(scores_a) / len(scores_a) - sum(scores_b) / len(scores_b)
        matrix[a][b] = {"p_value": p_value, "mean_diff": mean_diff}
    return matrix


def print_report(
    metric: str,
    means: Mapping[str, float],
    matrix: Mapping[str, Mapping[str, Mapping[str, float]]],
    alpha: float,
) -> None:
    systems = list(matrix)
    print(f"\n=== {metric} ===")
    print("Trung bình mỗi hệ thống: " + ", ".join(f"{s}={means[s]:.4f}" for s in systems))

    print("\nMa trận p-value (paired bootstrap, hai phía):")
    print(f"{'':<14}" + "".join(f"{s:>14}" for s in systems))
    for a in systems:
        cells = []
        for b in systems:
            cells.append("—" if a == b else f"{matrix[a][b]['p_value']:.4f}")
        print(f"{a:<14}" + "".join(f"{c:>14}" for c in cells))

    print("\nTừng cặp (Δ = trung bình A - trung bình B):")
    for a, b in itertools.combinations(systems, 2):
        cell = matrix[a][b]
        verdict = "có ý nghĩa" if cell["p_value"] < alpha else "KHÔNG có ý nghĩa"
        print(
            f"  {a:<12} vs {b:<12} Δ = {cell['mean_diff']:+.4f}  "
            f"p = {cell['p_value']:.4f}  ({verdict} ở alpha={alpha})"
        )


def main() -> None:
    force_utf8_stdout()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations", type=int, default=10_000, help="số lần resample của bootstrap"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed để kết quả tái lập được")
    parser.add_argument("--alpha", type=float, default=0.05, help="ngưỡng ý nghĩa thống kê")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chỉ dùng N truy vấn đầu (smoke run); khi có cờ này KHÔNG ghi file kết quả",
    )
    args = parser.parse_args()

    from scripts.compute_baseline_results import load_labels, load_runs

    labels = load_labels()
    runs = load_runs()
    print(f"Nhãn đã gộp: {len(labels)} | hệ thống trong pool_top20.json: {', '.join(runs)}")

    for key, reason in EXCLUDED.items():
        if key in runs:
            print(f"\nBỎ QUA '{key}' khỏi mọi kiểm định: {reason}.")

    included = [name for name in SYSTEMS if name in runs]
    if len(included) < 2:
        raise SystemExit(f"cần >=2 hệ thống baseline trong pool, chỉ thấy {included}")
    if missing := [name for name in SYSTEMS if name not in runs]:
        print(f"CẢNH BÁO: thiếu hệ thống {missing} trong pool_top20.json — bỏ khỏi ma trận.")

    query_ids = sorted_query_ids(runs[included[0]])
    for name in included[1:]:
        if set(runs[name]) != set(query_ids):
            raise SystemExit(
                f"'{name}' không cùng tập truy vấn với '{included[0]}' — "
                "kiểm định ghép cặp yêu cầu cùng tập truy vấn, cùng thứ tự"
            )
    if args.limit is not None:
        query_ids = query_ids[: args.limit]
    print(f"Kiểm định trên {len(query_ids)} truy vấn, {len(included)} hệ thống: {included}")

    grades = _grade_lookup(labels)
    per_query: dict[str, dict[str, list[float]]] = {}
    for name in included:
        run = {qid: runs[name][qid] for qid in query_ids}
        scores = {
            "recall@1": per_query_recall_at_k(run, grades, query_ids, k=1),
            f"recall@{RECALL_K}": per_query_recall_at_k(run, grades, query_ids, k=RECALL_K),
            "mrr": per_query_reciprocal_rank(run, grades, query_ids),
        }
        check_against_aggregate(scores, run, labels)
        per_query[name] = scores

    results: dict[str, Any] = {
        "systems": included,
        "excluded": {key: reason for key, reason in EXCLUDED.items() if key in runs},
        "n_queries": len(query_ids),
        "iterations": args.iterations,
        "seed": args.seed,
        "alpha": args.alpha,
        "limit": args.limit,
        "metrics": {},
    }
    for metric in ("recall@1", f"recall@{RECALL_K}", "mrr"):
        matrix = run_pairwise_tests(per_query, metric, args.iterations, args.seed)
        means = {
            name: sum(per_query[name][metric]) / len(per_query[name][metric])
            for name in included
        }
        results["metrics"][metric] = {"means": means, "pairs": matrix}
        print_report(metric, means, matrix, args.alpha)

    if args.limit is not None:
        print(
            f"\n--limit {args.limit} => smoke run, KHÔNG ghi {OUT_PATH.name}. "
            "Chạy lại không có --limit để lấy số thật."
        )
    else:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu {OUT_PATH}")


if __name__ == "__main__":
    main()
