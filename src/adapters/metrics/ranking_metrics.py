"""Recall@K / MRR / nDCG@10 over pooled 0/1/2 labels (MetricsPort).

Pattern to adapt: ``reference_repos/marqo-FashionCLIP/utils/metrics.py``. Take
its ``mrr()`` almost as-is (sort scores desc, first hit with grade > 0, average
reciprocal rank over ALL queries including misses). Do NOT take its
``evaluate_retrieval()``: that delegates to the ``beir`` package, which is a heavy
dependency and — more importantly — defines Recall differently from us (see
below). Reimplement in numpy.

    !! Recall@K here means HIT RATE !!
    DeCuong Mục 5.1 defines Recall@K as "the share of queries with at least one
    relevant product in the top K" — per query it is 0 or 1, then averaged over
    queries. BEIR (and hence marqo's metrics.py) instead reports
    |relevant ∩ top-K| / |relevant|, a fraction of all relevant items. The two
    numbers are NOT comparable, and ours is systematically higher. Any number
    quoted from an external leaderboard must be labelled with which definition
    it uses, otherwise the baseline table in the report is meaningless.

Definitions to implement (grades: 0 = irrelevant, 1 = partly, 2 = on point):
* ``recall_at_k``  : hit rate as above; grade >= 1 counts as a hit.
* ``mrr``          : 1/rank of the first hit with grade >= 1, 0 if none; averaged.
* ``ndcg_at_k``    : graded. DCG = sum over i<k of (2**grade_i - 1) / log2(i + 2);
                     IDCG from the labelled grades of that query sorted desc.
                     Unlabelled (query, product) pairs count as grade 0 — this is
                     exactly why top-20 pooling across all systems matters.

Implementation notes for whoever picks this up:
* Every metric must average over the full query set, not only queries that got a
  hit, or systems that return nothing on hard queries look good.
* Assert that every query in ``run`` has at least one label, and report how many
  labels were missing. Silent zeros are the classic way to fake a good result.
* Paired bootstrap significance test (DeCuong Mục 5.5) also lives here.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from src.core.models import RelevanceLabel
from src.core.ports import RunResults

logger = logging.getLogger(__name__)


def _grade_lookup(labels: Sequence[RelevanceLabel]) -> dict[tuple[str, str], float]:
    """(query_id, product_id) -> grade, averaging when >1 annotator graded a pair.

    DeCuong Mục 3.4 asks for 2 independent annotators per pair; it does not say
    how to reduce two grades into one. Averaging (not requiring exact agreement,
    not taking the max) keeps the disagreement's information — 0-then-2 lands at
    1, distinct from a real 1-1 agreement, but neither is used as ground truth
    without inspecting Cohen's kappa first. A single grade is used as-is.
    """
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for label in labels:
        grouped[(label.query_id, label.product_id)].append(label.grade)
    return {key: sum(grades) / len(grades) for key, grades in grouped.items()}


def _queries_with_zero_ceiling(
    run: RunResults, grades: dict[tuple[str, str], float], min_relevant_grade: int
) -> set[str]:
    """Query ids where no pooled label reaches ``min_relevant_grade`` at all.

    Recall/MRR still average these in as a genuine 0 (a system that finds
    nothing where nothing was relevant contributes correctly). nDCG cannot: its
    denominator (IDCG) would be zero, so these are excluded there instead —
    exactly the Mục 3.2 caveat about categories with no ground truth to find.
    """
    zero_ceiling = set()
    for query_id in run:
        pair_grades = [
            grade
            for (q, _pid), grade in grades.items()
            if q == query_id
        ]
        if not pair_grades or max(pair_grades) < min_relevant_grade:
            zero_ceiling.add(query_id)
    return zero_ceiling


class RankingMetrics:
    """Ranking metrics over pooled graded labels. Satisfies MetricsPort."""

    def __init__(self, min_relevant_grade: int = 1) -> None:
        self.min_relevant_grade = min_relevant_grade

    def _check_coverage(self, run: RunResults, grades: dict[tuple[str, str], float]) -> None:
        labelled_queries = {q for q, _ in grades}
        missing = [q for q in run if q not in labelled_queries]
        if missing:
            logger.warning(
                "%d/%d queries in this run have NO relevance label at all "
                "(will silently score 0 on every metric): %s",
                len(missing),
                len(run),
                missing[:10],
            )

    def recall_at_k(self, run: RunResults, labels: Sequence[RelevanceLabel], k: int) -> float:
        """Hit rate at k — NOT the BEIR fraction-of-relevant definition.

        Recall@K = (1/|Q|) * sum_over_queries( 1 if any of top-K results has
        grade >= min_relevant_grade else 0 )

        Per query the term is binary (0 or 1) — it does NOT divide by the total
        number of relevant items for that query, which is what BEIR/marqo call
        Recall. Average the binary term over ALL queries in ``labels``, including
        ones where every system returns 0 hits.
        """
        grades = _grade_lookup(labels)
        self._check_coverage(run, grades)
        if not run:
            return 0.0
        hits = 0
        for query_id, results in run.items():
            top_k = results[:k]
            is_hit = any(
                grades.get((query_id, hit.product_id), 0.0) >= self.min_relevant_grade
                for hit in top_k
            )
            hits += int(is_hit)
        return hits / len(run)

    def mrr(self, run: RunResults, labels: Sequence[RelevanceLabel]) -> float:
        """Mean Reciprocal Rank.

        MRR = (1/|Q|) * sum_over_queries( 1/rank_of_first_hit ), where
        rank_of_first_hit is the 1-indexed position of the first result with
        grade >= min_relevant_grade, and the term is 0 (not skipped) for a query
        with no hit at all. rank is over the full ranked list returned by ``run``,
        not truncated to any K.
        """
        grades = _grade_lookup(labels)
        self._check_coverage(run, grades)
        if not run:
            return 0.0
        total = 0.0
        for query_id, results in run.items():
            reciprocal = 0.0
            for hit in results:
                if grades.get((query_id, hit.product_id), 0.0) >= self.min_relevant_grade:
                    reciprocal = 1.0 / hit.rank
                    break
            total += reciprocal
        return total / len(run)

    def ndcg_at_k(self, run: RunResults, labels: Sequence[RelevanceLabel], k: int = 10) -> float:
        """Normalized Discounted Cumulative Gain at k, graded 0/1/2.

        DCG@k  = sum_{i=1}^{k} (2**grade_i - 1) / log2(i + 1)
        IDCG@k = DCG@k computed on the SAME query's labelled grades sorted
                 descending (the best possible ordering), not on a fixed
                 upper bound — a query with only one grade-2 label has
                 IDCG@k = (2**2 - 1) / log2(2) = 3, not the value from a
                 hypothetical all-2 ranking.
        nDCG@k = DCG@k / IDCG@k, and is 0 (not undefined) if IDCG@k == 0
                 (no relevant items exist for that query at all — exclude such
                 queries from the average rather than counting them as 0; note
                 this exclusion in the report, per Mục 3.2's category-with-zero-
                 samples caveat).
        i is 1-indexed in the DCG formula above (i.e. rank 1 uses log2(2)=1,
        not log2(1)=0 which would divide by zero).
        """
        grades = _grade_lookup(labels)
        self._check_coverage(run, grades)
        zero_ceiling = _queries_with_zero_ceiling(run, grades, self.min_relevant_grade)
        eligible = [q for q in run if q not in zero_ceiling]
        if not eligible:
            return 0.0

        total = 0.0
        for query_id in eligible:
            results = run[query_id][:k]
            dcg = sum(
                (2 ** grades.get((query_id, hit.product_id), 0.0) - 1) / math.log2(i + 1)
                for i, hit in enumerate(results, start=1)
            )
            all_grades_for_query = sorted(
                (grade for (q, _pid), grade in grades.items() if q == query_id),
                reverse=True,
            )
            idcg = sum(
                (2**grade - 1) / math.log2(i + 1)
                for i, grade in enumerate(all_grades_for_query[:k], start=1)
            )
            total += (dcg / idcg) if idcg > 0 else 0.0
        return total / len(eligible)

    def summary(self, run: RunResults, labels: Sequence[RelevanceLabel]) -> dict[str, float]:
        """All headline metrics for one system: Recall@1/5/10, MRR, nDCG@10."""
        return {
            "recall@1": self.recall_at_k(run, labels, k=1),
            "recall@5": self.recall_at_k(run, labels, k=5),
            "recall@10": self.recall_at_k(run, labels, k=10),
            "mrr": self.mrr(run, labels),
            "ndcg@10": self.ndcg_at_k(run, labels, k=10),
        }


def cohens_kappa(grades_a: Sequence[int], grades_b: Sequence[int]) -> float:
    """Inter-annotator agreement on the 0/1/2 scale (DeCuong Mục 3.4).

    kappa = (p_o - p_e) / (1 - p_e): p_o is the observed fraction of items
    where the two annotators gave the identical grade; p_e is the agreement
    expected by chance alone, from each annotator's own marginal grade
    distribution (sum over grades of P(a=g) * P(b=g)). kappa=1 is perfect
    agreement, 0 is chance-level, negative is worse than chance.
    """
    if len(grades_a) != len(grades_b):
        raise ValueError(f"grades_a and grades_b must be the same length, got "
                          f"{len(grades_a)} and {len(grades_b)}")
    n = len(grades_a)
    if n == 0:
        raise ValueError("need at least one graded pair")

    observed_agreement = sum(a == b for a, b in zip(grades_a, grades_b, strict=True)) / n

    counts_a = Counter(grades_a)
    counts_b = Counter(grades_b)
    all_grades = set(counts_a) | set(counts_b)
    chance_agreement = sum(
        (counts_a.get(g, 0) / n) * (counts_b.get(g, 0) / n) for g in all_grades
    )

    if chance_agreement >= 1.0:
        # Both annotators gave the same single grade to everything — agreement
        # is total but uninformative; report perfect agreement rather than 0/0.
        return 1.0
    return (observed_agreement - chance_agreement) / (1 - chance_agreement)


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    iterations: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided p-value for "system A beats system B" (DeCuong Mục 5.5).

    ``scores_a``/``scores_b`` are PER-QUERY scores of the same metric (e.g. each
    query's own Recall@5 contribution, 0 or 1) for the two systems, aligned by
    query. Resamples query indices with replacement ``iterations`` times, and
    the p-value is the fraction of resamples where the sign of the resampled
    mean difference disagrees with the sign observed on the real data — the
    standard paired bootstrap significance test.
    """
    import numpy as np

    if len(scores_a) != len(scores_b):
        raise ValueError(f"scores_a and scores_b must be the same length, got "
                          f"{len(scores_a)} and {len(scores_b)}")
    n = len(scores_a)
    if n == 0:
        raise ValueError("need at least one paired query score")

    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    observed_diff = float(np.mean(a - b))
    if observed_diff == 0.0:
        return 1.0

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(iterations, n))
    resampled_diff = (a[indices] - b[indices]).mean(axis=1)

    if observed_diff > 0:
        extreme = np.count_nonzero(resampled_diff <= 0)
    else:
        extreme = np.count_nonzero(resampled_diff >= 0)
    p_value = 2.0 * extreme / iterations
    return min(p_value, 1.0)
