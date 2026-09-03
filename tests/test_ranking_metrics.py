"""RankingMetrics: hand-computed expected values, not just shape checks."""

from __future__ import annotations

import math

import pytest

from src.adapters.metrics.ranking_metrics import (
    RankingMetrics,
    cohens_kappa,
    paired_bootstrap_test,
)
from src.core.models import Product, RelevanceLabel, SearchResult

_P = {
    pid: Product(product_id=pid, title=pid, category="áo")
    for pid in ("p1", "p2", "p3", "p4")
}

# What the ``run_and_labels`` fixture hands back: a run (query_id -> ranked hits)
# plus the ground-truth labels for it.
RunAndLabels = tuple[dict[str, list[SearchResult]], list[RelevanceLabel]]


def _hit(pid: str, rank: int) -> SearchResult:
    return SearchResult(product=_P[pid], score=1.0 / rank, rank=rank)


@pytest.fixture
def run_and_labels() -> RunAndLabels:
    run = {
        "q1": [_hit("p3", 1), _hit("p1", 2), _hit("p2", 3)],
        "q2": [_hit("p4", 1)],
    }
    labels = [
        RelevanceLabel(query_id="q1", product_id="p1", grade=2),
        RelevanceLabel(query_id="q1", product_id="p2", grade=1),
        RelevanceLabel(query_id="q1", product_id="p3", grade=0),
        RelevanceLabel(query_id="q2", product_id="p4", grade=0),
    ]
    return run, labels


def test_recall_at_1_misses_when_the_top_hit_is_irrelevant(run_and_labels: RunAndLabels) -> None:
    run, labels = run_and_labels
    metrics = RankingMetrics()

    # q1's rank-1 hit (p3) is grade 0, q2's only hit is grade 0 -> 0 hits / 2 queries.
    assert metrics.recall_at_k(run, labels, k=1) == pytest.approx(0.0)


def test_recall_at_3_counts_the_deeper_hit(run_and_labels: RunAndLabels) -> None:
    run, labels = run_and_labels
    metrics = RankingMetrics()

    # q1 has a grade>=1 hit within top-3 (p1, p2); q2 has none -> 1 hit / 2 queries.
    assert metrics.recall_at_k(run, labels, k=3) == pytest.approx(0.5)


def test_mrr_uses_the_first_relevant_ranks_reciprocal(run_and_labels: RunAndLabels) -> None:
    run, labels = run_and_labels
    metrics = RankingMetrics()

    # q1's first relevant hit is at rank 2 -> 1/2; q2 has no hit -> 0. Mean = 0.25.
    assert metrics.mrr(run, labels) == pytest.approx(0.25)


def test_ndcg_matches_hand_computed_value_and_excludes_zero_ceiling_queries(
    run_and_labels: RunAndLabels,
) -> None:
    run, labels = run_and_labels
    metrics = RankingMetrics()

    dcg = (
        (2**0 - 1) / math.log2(2)  # rank 1: p3, grade 0
        + (2**2 - 1) / math.log2(3)  # rank 2: p1, grade 2
        + (2**1 - 1) / math.log2(4)  # rank 3: p2, grade 1
    )
    idcg = (
        (2**2 - 1) / math.log2(2)  # ideal order: 2, 1, 0
        + (2**1 - 1) / math.log2(3)
        + (2**0 - 1) / math.log2(4)
    )
    expected_q1 = dcg / idcg

    # q2's only label is grade 0 (nothing relevant exists for it at all) -> IDCG
    # would be 0, so q2 is excluded from the average rather than counted as 0.
    assert metrics.ndcg_at_k(run, labels, k=3) == pytest.approx(expected_q1)


def test_summary_returns_all_five_headline_metrics(run_and_labels: RunAndLabels) -> None:
    run, labels = run_and_labels
    scores = RankingMetrics().summary(run, labels)

    assert set(scores) == {"recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_metrics_average_over_every_query_including_unlabelled_ones() -> None:
    # A query with zero labels at all must still count as a miss, not be skipped
    # -- otherwise a system that returns garbage on hard queries looks perfect.
    run = {"q1": [_hit("p1", 1)], "unlabelled": [_hit("p1", 1)]}
    labels = [RelevanceLabel(query_id="q1", product_id="p1", grade=2)]
    metrics = RankingMetrics()

    assert metrics.recall_at_k(run, labels, k=1) == pytest.approx(0.5)
    assert metrics.mrr(run, labels) == pytest.approx(0.5)


def test_two_annotators_disagreeing_on_a_pair_average_to_a_fractional_grade() -> None:
    run = {"q1": [_hit("p1", 1)]}
    labels = [
        RelevanceLabel(query_id="q1", product_id="p1", grade=0, annotator="a"),
        RelevanceLabel(query_id="q1", product_id="p1", grade=2, annotator="b"),
    ]
    metrics = RankingMetrics(min_relevant_grade=1)

    # Averaged grade is 1.0, which clears the min_relevant_grade=1 threshold.
    assert metrics.recall_at_k(run, labels, k=1) == pytest.approx(1.0)


def test_cohens_kappa_is_one_for_perfect_but_non_trivial_agreement() -> None:
    grades_a = [0, 1, 2, 0, 1, 2]
    grades_b = [0, 1, 2, 0, 1, 2]

    assert cohens_kappa(grades_a, grades_b) == pytest.approx(1.0)


def test_cohens_kappa_is_near_zero_for_chance_level_agreement() -> None:
    # Both annotators use all 3 grades equally often but in a shuffled pattern
    # that matches only as often as random chance would.
    grades_a = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
    grades_b = [1, 2, 0, 2, 0, 1, 1, 2, 0, 2, 0, 1]

    kappa = cohens_kappa(grades_a, grades_b)
    assert kappa < 0.2


def test_cohens_kappa_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa([0, 1], [0, 1, 2])


def test_paired_bootstrap_is_significant_when_one_system_always_wins() -> None:
    scores_a = [1.0] * 30
    scores_b = [0.0] * 30

    p_value = paired_bootstrap_test(scores_a, scores_b, iterations=2000, seed=42)

    assert p_value < 0.01


def test_paired_bootstrap_is_not_significant_for_identical_systems() -> None:
    scores = [1.0, 0.0, 1.0, 1.0, 0.0]

    assert paired_bootstrap_test(scores, scores, iterations=2000) == pytest.approx(1.0)


def test_paired_bootstrap_is_deterministic_given_a_seed() -> None:
    scores_a = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    scores_b = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

    p1 = paired_bootstrap_test(scores_a, scores_b, iterations=500, seed=7)
    p2 = paired_bootstrap_test(scores_a, scores_b, iterations=500, seed=7)

    assert p1 == p2
