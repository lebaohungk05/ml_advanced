"""Per-query score extraction for the paired bootstrap: hand-computed values.

The bootstrap itself is tested in ``tests/test_ranking_metrics.py``; what has to
be right here is that the per-query scores are the decomposition of the SAME
Recall@10 / MRR the accuracy table publishes, and that ``siglip2_lora`` really
cannot sneak into the comparison.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.adapters.metrics.ranking_metrics import RankingMetrics, _grade_lookup
from src.core.models import Product, RelevanceLabel, SearchResult

ROOT = Path(__file__).resolve().parent.parent


def _load_script() -> ModuleType:
    """Import ``scripts/significance_tests.py`` (``scripts`` is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "significance_tests_under_test", ROOT / "scripts" / "significance_tests.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sig = _load_script()

_P = {pid: Product(product_id=pid, title=pid, category="áo") for pid in ("p1", "p2", "p3", "p4")}
QUERY_IDS = ["q1", "q2", "q3"]

# system A puts a grade-2 hit at rank 1 on q1 and a grade-1 hit at rank 1 on q2;
# system B pushes both of those down to rank 2. Nobody hits anything on q3.
RUN_A: dict[str, list[SearchResult]] = {
    "q1": [
        SearchResult(product=_P["p1"], score=0.9, rank=1),
        SearchResult(product=_P["p2"], score=0.8, rank=2),
        SearchResult(product=_P["p3"], score=0.7, rank=3),
    ],
    "q2": [
        SearchResult(product=_P["p3"], score=0.9, rank=1),
        SearchResult(product=_P["p4"], score=0.8, rank=2),
    ],
    "q3": [SearchResult(product=_P["p4"], score=0.9, rank=1)],
}
RUN_B: dict[str, list[SearchResult]] = {
    "q1": [
        SearchResult(product=_P["p3"], score=0.9, rank=1),
        SearchResult(product=_P["p1"], score=0.8, rank=2),
    ],
    "q2": [
        SearchResult(product=_P["p4"], score=0.9, rank=1),
        SearchResult(product=_P["p3"], score=0.8, rank=2),
    ],
    "q3": [SearchResult(product=_P["p4"], score=0.9, rank=1)],
}
LABELS = [
    RelevanceLabel(query_id="q1", product_id="p1", grade=2),
    RelevanceLabel(query_id="q1", product_id="p2", grade=1),
    RelevanceLabel(query_id="q1", product_id="p3", grade=0),
    RelevanceLabel(query_id="q2", product_id="p3", grade=1),
    RelevanceLabel(query_id="q2", product_id="p4", grade=0),
    RelevanceLabel(query_id="q3", product_id="p4", grade=0),
]
GRADES = _grade_lookup(LABELS)


def test_per_query_recall_is_binary_per_query() -> None:
    # A: q1 hits p1 (grade 2), q2 hits p3 (grade 1), q3's only hit is grade 0.
    assert sig.per_query_recall_at_k(RUN_A, GRADES, QUERY_IDS, k=10) == [1.0, 1.0, 0.0]
    # B: same hits, just deeper -- Recall@10 cannot see the rank difference.
    assert sig.per_query_recall_at_k(RUN_B, GRADES, QUERY_IDS, k=10) == [1.0, 1.0, 0.0]


def test_per_query_recall_truncates_at_k() -> None:
    # At k=1 B's rank-1 hits (p3 on q1, p4 on q2) are both grade 0.
    assert sig.per_query_recall_at_k(RUN_A, GRADES, QUERY_IDS, k=1) == [1.0, 1.0, 0.0]
    assert sig.per_query_recall_at_k(RUN_B, GRADES, QUERY_IDS, k=1) == [0.0, 0.0, 0.0]


def test_per_query_reciprocal_rank_sees_the_rank_difference() -> None:
    assert sig.per_query_reciprocal_rank(RUN_A, GRADES, QUERY_IDS) == [1.0, 1.0, 0.0]
    assert sig.per_query_reciprocal_rank(RUN_B, GRADES, QUERY_IDS) == [0.5, 0.5, 0.0]


def test_scores_follow_the_requested_query_order() -> None:
    assert sig.per_query_reciprocal_rank(RUN_B, GRADES, ["q3", "q1"]) == [0.0, 0.5]


def test_a_query_absent_from_the_run_scores_zero_not_dropped() -> None:
    run = {"q1": RUN_A["q1"]}

    assert sig.per_query_recall_at_k(run, GRADES, QUERY_IDS, k=10) == [1.0, 0.0, 0.0]
    assert sig.per_query_reciprocal_rank(run, GRADES, QUERY_IDS) == [1.0, 0.0, 0.0]


def test_two_annotators_averaging_to_one_still_counts_as_relevant() -> None:
    # 2 and 0 average to 1.0, which is >= min_relevant_grade, so q3 becomes a hit.
    labels = [
        *LABELS,
        RelevanceLabel(query_id="q3", product_id="p4", grade=2, annotator="b"),
    ]

    grades = _grade_lookup(labels)

    assert grades[("q3", "p4")] == pytest.approx(1.0)
    assert sig.per_query_recall_at_k(RUN_A, grades, QUERY_IDS, k=10) == [1.0, 1.0, 1.0]


def test_per_query_means_equal_the_published_aggregates() -> None:
    metrics = RankingMetrics()
    scores = {
        "recall@1": sig.per_query_recall_at_k(RUN_B, GRADES, QUERY_IDS, k=1),
        "recall@10": sig.per_query_recall_at_k(RUN_B, GRADES, QUERY_IDS, k=10),
        "mrr": sig.per_query_reciprocal_rank(RUN_B, GRADES, QUERY_IDS),
    }

    assert sum(scores["recall@10"]) / 3 == pytest.approx(metrics.recall_at_k(RUN_B, LABELS, k=10))
    assert sum(scores["mrr"]) / 3 == pytest.approx(metrics.mrr(RUN_B, LABELS))
    sig.check_against_aggregate(scores, RUN_B, LABELS)


def test_check_against_aggregate_rejects_a_drifted_extraction() -> None:
    scores = {
        "recall@1": sig.per_query_recall_at_k(RUN_B, GRADES, QUERY_IDS, k=1),
        "recall@10": [1.0, 1.0, 1.0],  # wrong: q3 has no relevant product
        "mrr": sig.per_query_reciprocal_rank(RUN_B, GRADES, QUERY_IDS),
    }

    with pytest.raises(AssertionError, match="recall@10"):
        sig.check_against_aggregate(scores, RUN_B, LABELS)


def test_pairwise_tests_report_effect_size_and_symmetric_p_values() -> None:
    per_query = {
        "a": {"mrr": sig.per_query_reciprocal_rank(RUN_A, GRADES, QUERY_IDS)},
        "b": {"mrr": sig.per_query_reciprocal_rank(RUN_B, GRADES, QUERY_IDS)},
    }

    matrix = sig.run_pairwise_tests(per_query, "mrr", iterations=200, seed=7)

    # A's mean RR is 2/3, B's is 1/3.
    assert matrix["a"]["b"]["mean_diff"] == pytest.approx(1.0 / 3.0)
    assert matrix["b"]["a"]["mean_diff"] == pytest.approx(-1.0 / 3.0)
    assert matrix["a"]["b"]["p_value"] == pytest.approx(matrix["b"]["a"]["p_value"])
    assert "a" not in matrix["a"]


def test_the_finetuned_system_is_tested_now_that_its_pool_is_labelled() -> None:
    # It was excluded while ~80% of its top-10 was unlabelled; the pool
    # extension has since been graded, so a bootstrap over it is valid.
    assert "siglip2_lora" in sig.SYSTEMS
    assert set(sig.SYSTEMS) == {
        "bm25",
        "clip",
        "siglip2",
        "visiglip_ot",
        "siglip2_lora",
        "axis1_lora_no_dora",
        "axis2_no_hard_negatives",
    }


def test_siglip1_stays_excluded_while_two_thirds_of_its_top_10_is_unlabelled() -> None:
    assert "siglip1" not in sig.SYSTEMS
    assert "siglip1" in sig.EXCLUDED
    assert sig.EXCLUDED["siglip1"].strip()


def test_numeric_query_ids_sort_numerically_not_lexicographically() -> None:
    assert sig.sorted_query_ids({"10": [], "2": [], "1": []}) == ["1", "2", "10"]
    assert sig.sorted_query_ids({"q10": [], "q2": []}) == ["q10", "q2"]
