"""STUB — Recall@K / MRR / nDCG@10 over pooled 0/1/2 labels (MetricsPort).

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

from collections.abc import Sequence

from src.core.models import RelevanceLabel
from src.core.ports import RunResults

_TODO = (
    "TODO(Sprint 2, Hiếu): reimplement in numpy following reference_repos/"
    "marqo-FashionCLIP/utils/metrics.py, hit-rate Recall per DeCuong Mục 5.1"
)


class RankingMetrics:
    """Ranking metrics over pooled graded labels. Satisfies MetricsPort."""

    def __init__(self, min_relevant_grade: int = 1) -> None:
        self.min_relevant_grade = min_relevant_grade

    def recall_at_k(self, run: RunResults, labels: Sequence[RelevanceLabel], k: int) -> float:
        """Hit rate at k — NOT the BEIR fraction-of-relevant definition."""
        raise NotImplementedError(_TODO)

    def mrr(self, run: RunResults, labels: Sequence[RelevanceLabel]) -> float:
        raise NotImplementedError(_TODO)

    def ndcg_at_k(self, run: RunResults, labels: Sequence[RelevanceLabel], k: int = 10) -> float:
        raise NotImplementedError(_TODO)

    def summary(self, run: RunResults, labels: Sequence[RelevanceLabel]) -> dict[str, float]:
        """All headline metrics for one system: Recall@1/5/10, MRR, nDCG@10."""
        raise NotImplementedError(_TODO)


def cohens_kappa(grades_a: Sequence[int], grades_b: Sequence[int]) -> float:
    """Inter-annotator agreement on the 0/1/2 scale (DeCuong Mục 3.4)."""
    raise NotImplementedError("TODO(Sprint 3, Hiếu): needed when pooling is labelled")


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    iterations: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided p-value for "system A beats system B" (DeCuong Mục 5.5)."""
    raise NotImplementedError("TODO(Sprint 4, Hiếu): resample per-query scores with seed")
