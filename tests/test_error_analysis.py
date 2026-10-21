"""Quy tắc chọn ca lỗi Sprint 4: giá trị tính tay, không chỉ kiểm tra kiểu dữ liệu.

Điều phải đúng ở đây là ca lỗi được chọn ĐÚNG như quy tắc mà báo cáo công bố:
rổ 1 (final trượt, zero-shot đúng) trước rổ 2 (cả hai trượt) trước rổ 3 (tụt
hạng), và trong mỗi rổ ưu tiên truy vấn mà zero-shot từng trả đúng ở hạng cao.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.core.models import Product, SearchResult

ROOT = Path(__file__).resolve().parent.parent


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "error_analysis_under_test", ROOT / "scripts" / "error_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ea = _load_script()

_P = {pid: Product(product_id=pid, title=pid, category="áo") for pid in ("p1", "p2", "p3")}


def _hit(pid: str, rank: int) -> SearchResult:
    return SearchResult(product=_P[pid], score=1.0 / rank, rank=rank)


def _score(recall: float, reciprocal: float) -> object:
    return ea.QueryScore(recall_at_k=recall, reciprocal_rank=reciprocal)


def test_per_query_scores_hand_computed() -> None:
    run = {
        "1": [_hit("p1", 1), _hit("p2", 2)],  # p2 là hit -> rr = 1/2
        "2": [_hit("p1", 1), _hit("p3", 2)],  # không hit nào
        "3": [_hit("p2", 1)],  # hit ngay hạng 1
    }
    grades = {("1", "p2"): 2.0, ("2", "p1"): 0.0, ("3", "p2"): 1.0}

    scores = ea.per_query_scores(run, grades, ["1", "2", "3"], k=2)

    assert scores["1"].recall_at_k == pytest.approx(1.0)
    assert scores["1"].reciprocal_rank == pytest.approx(0.5)
    assert scores["2"].recall_at_k == pytest.approx(0.0)
    assert scores["2"].reciprocal_rank == pytest.approx(0.0)
    assert scores["3"].recall_at_k == pytest.approx(1.0)
    assert scores["3"].reciprocal_rank == pytest.approx(1.0)


def test_recall_ignores_hits_below_k_but_reciprocal_rank_does_not() -> None:
    run = {"1": [_hit("p1", 1), _hit("p2", 2), _hit("p3", 3)]}
    grades = {("1", "p3"): 2.0}

    scores = ea.per_query_scores(run, grades, ["1"], k=2)

    # hit nằm ở hạng 3, ngoài top-2 -> Recall@2 = 0 nhưng rr vẫn là 1/3.
    assert scores["1"].recall_at_k == pytest.approx(0.0)
    assert scores["1"].reciprocal_rank == pytest.approx(1.0 / 3.0)


def test_bucket_of_covers_the_three_buckets_and_the_non_error_case() -> None:
    assert ea.bucket_of(_score(0.0, 0.0), _score(1.0, 0.25)) == ea.BUCKET_FINAL_ONLY
    assert ea.bucket_of(_score(0.0, 0.0), _score(0.0, 0.0)) == ea.BUCKET_BOTH_FAIL
    assert ea.bucket_of(_score(1.0, 0.2), _score(1.0, 1.0)) == ea.BUCKET_REGRESSION
    assert ea.bucket_of(_score(1.0, 1.0), _score(1.0, 0.5)) is None
    assert ea.bucket_of(_score(1.0, 0.5), _score(1.0, 0.5)) is None


def test_bucket_of_ignores_a_deep_zeroshot_hit_when_final_also_fails_at_k() -> None:
    # zero-shot có hit ở hạng 15 (rr>0) nhưng Recall@10 = 0 -> vẫn là "cả hai trượt",
    # không phải rổ 1: rổ 1 dành cho ca zero-shot trả đúng TRONG top-10.
    assert ea.bucket_of(_score(0.0, 0.0), _score(0.0, 1.0 / 15)) == ea.BUCKET_BOTH_FAIL


def test_select_error_cases_orders_buckets_then_zeroshot_rank() -> None:
    final = {
        "1": _score(1.0, 1.0),  # không lỗi
        "2": _score(0.0, 0.0),  # cả hai trượt
        "3": _score(0.0, 0.0),  # rổ 1: zero-shot đúng ở hạng 4
        "4": _score(1.0, 0.25),  # tụt hạng
        "5": _score(0.0, 0.0),  # rổ 1: zero-shot đúng ở hạng 1 -> trước q3
    }
    reference = {
        "1": _score(1.0, 0.5),
        "2": _score(0.0, 0.0),
        "3": _score(1.0, 0.25),
        "4": _score(1.0, 1.0),
        "5": _score(1.0, 1.0),
    }

    cases = ea.select_error_cases(final, reference, max_cases=10)

    assert [(c.query_id, c.bucket) for c in cases] == [
        ("5", ea.BUCKET_FINAL_ONLY),
        ("3", ea.BUCKET_FINAL_ONLY),
        ("2", ea.BUCKET_BOTH_FAIL),
        ("4", ea.BUCKET_REGRESSION),
    ]


def test_select_error_cases_cuts_the_worst_n_and_never_takes_a_clean_query() -> None:
    final = {
        "1": _score(0.0, 0.0),
        "2": _score(0.0, 0.0),
        "3": _score(1.0, 0.5),  # tụt hạng, phải bị cắt vì rổ 3 xếp sau
        "4": _score(1.0, 1.0),  # không lỗi
    }
    reference = {
        "1": _score(0.0, 0.0),
        "2": _score(0.0, 0.0),
        "3": _score(1.0, 1.0),
        "4": _score(1.0, 1.0),
    }

    cases = ea.select_error_cases(final, reference, max_cases=2)

    assert [c.query_id for c in cases] == ["1", "2"]
    assert all(c.bucket == ea.BUCKET_BOTH_FAIL for c in cases)


def test_select_error_cases_ties_break_on_numeric_query_id() -> None:
    final = {str(qid): _score(0.0, 0.0) for qid in (100, 9, 21)}
    reference = dict(final)

    cases = ea.select_error_cases(final, reference, max_cases=3)

    assert [c.query_id for c in cases] == ["9", "21", "100"]


def test_accessory_head_terms_match_only_the_head_noun() -> None:
    queries = {
        "1": {"text": "giày cao gót đen", "group": "A"},
        "2": {"text": "túi tote vải canvas basic", "group": "C"},
        "3": {"text": "quần short kaki túi hộp màu xanh rêu", "group": "B"},
        "4": {"text": "giày cao gót mũi nhọn màu đen da bóng", "group": "B"},
        "5": {"text": "áo khoác nỉ có mũ màu đen", "group": "B"},
    }

    flagged = ea.accessory_target_queries(queries)

    assert set(flagged) == {"1", "2", "4"}
    assert flagged["1"]["fashionpedia_category_id"] == "24"
    assert flagged["2"]["fashionpedia_category_id"] == "25"


def test_zero_ceiling_needs_every_pooled_label_below_the_relevant_grade() -> None:
    grades = {
        ("1", "p1"): 0.0,
        ("1", "p2"): 0.5,  # 2 người chấm 0/1 -> trung bình 0.5, chưa đạt ngưỡng 1
        ("2", "p1"): 1.0,
        ("3", "p9"): 0.0,
    }

    assert ea.queries_with_zero_ceiling(["1", "2", "3"], grades) == frozenset({"1", "3"})


def test_unrepresentable_queries_flags_traditional_garments_only() -> None:
    queries = {
        "1": {"text": "áo dài trắng nữ sinh", "group": "C"},
        "2": {"text": "áo sơ mi nữ trắng", "group": "A"},
    }

    flagged = ea.unrepresentable_queries(queries)

    assert set(flagged) == {"1"}
    assert flagged["1"]["term"] == "áo dài"
