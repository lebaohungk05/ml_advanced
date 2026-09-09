"""Sprint 4 deliverable: chọn 15-20 ca lỗi tệ nhất + thống kê thiên lệch theo
nhóm truy vấn (DeCuong Mục 5.4, Mục 3.2/6.1).

Ba phần, chạy trên đúng bộ nhãn + pool mà bảng kết quả chính dùng
(``scripts/compute_baseline_results.py:load_labels/load_runs``):

1. Recall@10 và reciprocal rank theo TỪNG truy vấn cho hệ thống cuối
   (``siglip2_lora``) và mốc zero-shot (``siglip2``).
2. Chọn ca lỗi theo quy tắc cố định, không chọn tay — 3 rổ, xếp theo mức độ
   nghiêm trọng giảm dần:
     * rổ 1 ``final_truot_zeroshot_dung``: final Recall@10 = 0 nhưng zero-shot
       tìm được — fine-tune làm MẤT kết quả đã có, tín hiệu lỗi rõ nhất;
     * rổ 2 ``ca_hai_truot``: cả hai Recall@10 = 0;
     * rổ 3 ``tut_hang``: final vẫn có hit trong top-10 nhưng thứ hạng hit đầu
       tiên tệ hơn zero-shot.
   Trong mỗi rổ xếp theo reciprocal rank của zero-shot giảm dần (chỗ nào từng có
   kết quả đúng ở hạng cao thì lỗi càng đáng phân tích), rồi theo id.
3. Recall@10 / nDCG@10 tách theo nhóm truy vấn A/B/C/D cho mọi hệ thống trong
   pool, cộng với số truy vấn KHÔNG hệ thống nào tìm được kết quả liên quan
   (trần 0 — Mục 3.2) và số truy vấn đòi loại trang phục mà 46 danh mục
   Fashionpedia không diễn đạt được.

Lưu ý về nDCG theo nhóm: ``RankingMetrics.ndcg_at_k`` loại các truy vấn trần 0
khỏi mẫu trung bình, nên nDCG của một nhóm được tính trên ít truy vấn hơn
Recall của cùng nhóm; script in ra cả 2 cỡ mẫu.

Usage:
    python -m scripts.error_analysis
    python -m scripts.error_analysis --max-cases 20
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapters.metrics.ranking_metrics import (  # noqa: E402
    RankingMetrics,
    _grade_lookup,  # cùng cách gộp điểm 2 người chấm như bảng kết quả chính
)
from src.core.models import RelevanceLabel, SearchResult  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
OUT_PATH = EVAL_DIR / "error_cases.json"
QUERIES_PATH = EVAL_DIR / "queries.json"
CATEGORY_MAPPING_PATH = ROOT / "data" / "category_mapping.json"
PROCESSED_DIR = ROOT / "data" / "processed"

FINAL_SYSTEM = "siglip2_lora"
REFERENCE_SYSTEM = "siglip2"
MIN_RELEVANT_GRADE = 1
RECALL_K = 10
TOP_N_SHOWN = 5
GROUPS = ("A", "B", "C", "D")

BUCKET_FINAL_ONLY = "final_truot_zeroshot_dung"
BUCKET_BOTH_FAIL = "ca_hai_truot"
BUCKET_REGRESSION = "tut_hang"
BUCKET_PRIORITY = {BUCKET_FINAL_ONLY: 0, BUCKET_BOTH_FAIL: 1, BUCKET_REGRESSION: 2}

# Khái niệm trang phục truyền thống/bản địa mà taxonomy Fashionpedia 46 danh mục
# không có ô nào diễn đạt được -> truy vấn chứa các từ này không có đáp án đúng
# trong catalog, không hệ thống nào trả lời được (DeCuong Mục 3.2/6.1).
UNREPRESENTABLE_TERMS = {
    "áo dài": "không có danh mục nào tương ứng (gần nhất là Dress #11, khác hẳn hình thái)",
    "áo bà ba": "không có danh mục tương ứng",
    "nón lá": "Hat #15 quá chung, không phân biệt được nón lá",
    "khăn đóng": "không có danh mục tương ứng",
    "yếm": "không có danh mục tương ứng",
    "áo tứ thân": "không có danh mục tương ứng",
}

# Danh từ trung tâm là phụ kiện -> danh mục Fashionpedia gộp cả loại phụ kiện đó
# vào 1 ô duy nhất, không diễn đạt được kiểu con mà truy vấn hỏi (giày cao gót vs
# giày thể thao đều là Shoe #24). Khớp theo tiền tố để "quần short kaki túi hộp"
# (phụ kiện chỉ là chi tiết) không bị tính là truy vấn phụ kiện.
ACCESSORY_HEAD_TERMS = {
    "kính": 14,
    "mũ": 15,
    "băng đô": 16,
    "cà vạt": 17,
    "găng tay": 18,
    "đồng hồ": 19,
    "dây nịt": 20,
    "thắt lưng": 20,
    "vớ": 23,
    "tất": 23,
    "giày": 24,
    "túi": 25,
    "ví": 25,
    "khăn": 26,
    "dù": 27,
}

Run = Mapping[str, Sequence[SearchResult]]
Grades = Mapping[tuple[str, str], float]


def force_utf8_stdout() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class QueryScore:
    """Điểm của MỘT truy vấn: hit rate ở K và reciprocal rank trên cả danh sách."""

    recall_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True)
class SelectedCase:
    query_id: str
    bucket: str
    final: QueryScore
    reference: QueryScore


def sorted_query_ids(run: Run) -> list[str]:
    ids = list(run)
    if all(qid.isdigit() for qid in ids):
        return sorted(ids, key=int)
    return sorted(ids)


def per_query_scores(
    run: Run, grades: Grades, query_ids: Sequence[str], k: int = RECALL_K
) -> dict[str, QueryScore]:
    """Recall@k (0/1) + reciprocal rank cho từng truy vấn.

    Cùng định nghĩa với ``RankingMetrics.recall_at_k``/``mrr``: truy vấn không có
    trong ``run`` tính 0, reciprocal rank lấy trên toàn danh sách trả về (không
    cắt ở k).
    """
    scores: dict[str, QueryScore] = {}
    for query_id in query_ids:
        hits = run.get(query_id, [])
        recall = 1.0 if any(
            grades.get((query_id, hit.product_id), 0.0) >= MIN_RELEVANT_GRADE for hit in hits[:k]
        ) else 0.0
        reciprocal = 0.0
        for hit in hits:
            if grades.get((query_id, hit.product_id), 0.0) >= MIN_RELEVANT_GRADE:
                reciprocal = 1.0 / hit.rank
                break
        scores[query_id] = QueryScore(recall_at_k=recall, reciprocal_rank=reciprocal)
    return scores


def bucket_of(final: QueryScore, reference: QueryScore) -> str | None:
    """Rổ lỗi của một truy vấn, hoặc None nếu không tính là ca lỗi."""
    if final.recall_at_k == 0.0:
        return BUCKET_BOTH_FAIL if reference.recall_at_k == 0.0 else BUCKET_FINAL_ONLY
    if final.reciprocal_rank < reference.reciprocal_rank:
        return BUCKET_REGRESSION
    return None


def select_error_cases(
    final_scores: Mapping[str, QueryScore],
    reference_scores: Mapping[str, QueryScore],
    max_cases: int,
) -> list[SelectedCase]:
    """Xếp mọi ca lỗi theo (rổ, rr zero-shot giảm dần, rr final tăng dần, id) rồi
    cắt ``max_cases`` ca đầu."""
    candidates: list[SelectedCase] = []
    for query_id, final in final_scores.items():
        reference = reference_scores[query_id]
        bucket = bucket_of(final, reference)
        if bucket is not None:
            candidates.append(SelectedCase(query_id, bucket, final, reference))

    candidates.sort(
        key=lambda case: (
            BUCKET_PRIORITY[case.bucket],
            -case.reference.reciprocal_rank,
            case.final.reciprocal_rank,
            int(case.query_id) if case.query_id.isdigit() else 0,
            case.query_id,
        )
    )
    return candidates[:max_cases]


def load_queries() -> dict[str, dict[str, str]]:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return {
        str(row["id"]): {"text": row["text"], "group": row["group"]}
        for row in payload["queries"]
    }


def load_product_meta() -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for split in ("train", "val", "test"):
        path = PROCESSED_DIR / f"{split}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for row in json.load(f):
                meta[row["product_id"]] = {
                    "category": row.get("category", ""),
                    "title": row.get("title", ""),
                    "image_path": row.get("image_path", ""),
                }
    return meta


def grades_by_annotator(labels: Sequence[RelevanceLabel]) -> dict[tuple[str, str], dict[str, int]]:
    spread: dict[tuple[str, str], dict[str, int]] = {}
    for label in labels:
        annotator = label.annotator or "?"
        spread.setdefault((label.query_id, label.product_id), {})[annotator] = label.grade
    return spread


def describe_top_hits(
    hits: Sequence[SearchResult],
    query_id: str,
    grades: Grades,
    spread: Mapping[tuple[str, str], Mapping[str, int]],
    product_meta: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    described: list[dict[str, Any]] = []
    for hit in hits[:TOP_N_SHOWN]:
        key = (query_id, hit.product_id)
        per_annotator = dict(spread.get(key, {}))
        meta = product_meta.get(hit.product_id, {})
        row: dict[str, Any] = {
            "rank": hit.rank,
            "product_id": hit.product_id,
            "grade": grades.get(key),
            "category": meta.get("category", ""),
            "items_in_image": meta.get("title", ""),
            "image_path": meta.get("image_path", ""),
        }
        if len(set(per_annotator.values())) > 1:
            row["annotator_disagreement"] = per_annotator
        described.append(row)
    return described


def group_breakdown(
    runs: Mapping[str, Run],
    labels: Sequence[RelevanceLabel],
    queries: Mapping[str, Mapping[str, str]],
    zero_ceiling: frozenset[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Recall@10 / nDCG@10 của từng hệ thống, tách theo nhóm A/B/C/D."""
    metrics = RankingMetrics(min_relevant_grade=MIN_RELEVANT_GRADE)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for name, run in runs.items():
        per_group: dict[str, dict[str, float]] = {}
        for group in GROUPS:
            sub = {
                qid: list(hits)
                for qid, hits in run.items()
                if queries.get(qid, {}).get("group") == group
            }
            if not sub:
                continue
            per_group[group] = {
                f"recall@{RECALL_K}": metrics.recall_at_k(sub, labels, k=RECALL_K),
                "ndcg@10": metrics.ndcg_at_k(sub, labels, k=10),
                "n_queries": float(len(sub)),
                "n_ndcg_eligible": float(len([q for q in sub if q not in zero_ceiling])),
            }
        result[name] = per_group
    return result


def accessory_target_queries(
    queries: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    """Truy vấn có danh từ trung tâm là phụ kiện (khớp tiền tố)."""
    flagged: dict[str, dict[str, str]] = {}
    for query_id, row in queries.items():
        text = row["text"].lower().strip()
        for term, category_id in sorted(ACCESSORY_HEAD_TERMS.items(), key=lambda kv: -len(kv[0])):
            if text.startswith(term):
                flagged[query_id] = {
                    "text": row["text"],
                    "group": row["group"],
                    "head_term": term,
                    "fashionpedia_category_id": str(category_id),
                }
                break
    return flagged


def target_axis_breakdown(
    runs: Mapping[str, Run],
    labels: Sequence[RelevanceLabel],
    accessory_ids: frozenset[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Recall@10 / nDCG@10 tách theo trục phụ kiện vs trang phục chính."""
    metrics = RankingMetrics(min_relevant_grade=MIN_RELEVANT_GRADE)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for name, run in runs.items():
        per_axis: dict[str, dict[str, float]] = {}
        for axis, wanted in (("phu_kien", True), ("trang_phuc_chinh", False)):
            sub = {
                qid: list(hits)
                for qid, hits in run.items()
                if (qid in accessory_ids) is wanted
            }
            if not sub:
                continue
            per_axis[axis] = {
                f"recall@{RECALL_K}": metrics.recall_at_k(sub, labels, k=RECALL_K),
                "ndcg@10": metrics.ndcg_at_k(sub, labels, k=10),
                "n_queries": float(len(sub)),
            }
        result[name] = per_axis
    return result


def queries_with_zero_ceiling(
    query_ids: Sequence[str], grades: Grades
) -> frozenset[str]:
    """Truy vấn không có BẤT KỲ nhãn >= 1 trong toàn pool.

    Pool là hợp top-20 của mọi hệ thống, nên đây là các truy vấn mà không hệ
    thống nào tìm được kết quả liên quan — trần bằng 0, đúng cảnh báo Mục 3.2.
    """
    best: dict[str, float] = dict.fromkeys(query_ids, 0.0)
    for (query_id, _pid), grade in grades.items():
        if query_id in best:
            best[query_id] = max(best[query_id], grade)
    return frozenset(qid for qid, top in best.items() if top < MIN_RELEVANT_GRADE)


def unrepresentable_queries(
    queries: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    """Truy vấn đòi loại trang phục mà 46 danh mục Fashionpedia không diễn đạt được."""
    flagged: dict[str, dict[str, str]] = {}
    for query_id, row in queries.items():
        text = row["text"].lower()
        for term, reason in UNREPRESENTABLE_TERMS.items():
            if term in text:
                flagged[query_id] = {
                    "text": row["text"],
                    "group": row["group"],
                    "term": term,
                    "reason": reason,
                }
                break
    return flagged


def needs_review_categories() -> list[dict[str, str]]:
    with open(CATEGORY_MAPPING_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return [
        {"id": str(cat["id"]), "en": cat["en"], "vi": cat["vi"], "note": cat.get("note", "")}
        for cat in payload["categories"]
        if cat.get("needs_review")
    ]


def print_selection_report(
    cases: Sequence[SelectedCase],
    bucket_counts: Mapping[str, int],
    max_cases: int,
    n_queries: int,
) -> None:
    print("\n=== Quy tắc chọn ca lỗi ===")
    print(
        f"Hệ thống cuối = {FINAL_SYSTEM}, mốc zero-shot = {REFERENCE_SYSTEM}, "
        f"K = {RECALL_K}, tối đa {max_cases} ca trên {n_queries} truy vấn."
    )
    print(f"  rổ 1 {BUCKET_FINAL_ONLY}: final Recall@10=0 nhưng zero-shot có hit")
    print(f"  rổ 2 {BUCKET_BOTH_FAIL}: cả hai Recall@10=0")
    print(f"  rổ 3 {BUCKET_REGRESSION}: final có hit nhưng rr(final) < rr(zero-shot)")
    print("\nSố ứng viên mỗi rổ (trước khi cắt):")
    for bucket in (BUCKET_FINAL_ONLY, BUCKET_BOTH_FAIL, BUCKET_REGRESSION):
        print(f"  {bucket:<28} {bucket_counts.get(bucket, 0)}")
    total_failures = bucket_counts.get(BUCKET_FINAL_ONLY, 0) + bucket_counts.get(
        BUCKET_BOTH_FAIL, 0
    )
    print(f"  -> tổng ca final trượt hoàn toàn (Recall@10=0): {total_failures}")
    if total_failures < 15:
        print(
            f"  Ghi chú: chỉ có {total_failures} ca trượt hoàn toàn (< 15), phần còn lại "
            f"lấy từ rổ {BUCKET_REGRESSION} theo đúng thứ tự xếp hạng."
        )
    print(f"\nĐã chọn {len(cases)} ca:")
    for case in cases:
        print(
            f"  q{case.query_id:<5} {case.bucket:<28} "
            f"rr(final)={case.final.reciprocal_rank:.3f} "
            f"rr(zs)={case.reference.reciprocal_rank:.3f}"
        )


def print_group_report(
    breakdown: Mapping[str, Mapping[str, Mapping[str, float]]],
    group_sizes: Mapping[str, int],
) -> None:
    print("\n=== Recall@10 / nDCG@10 theo nhóm truy vấn ===")
    print("Cỡ nhóm: " + ", ".join(f"{g}={group_sizes.get(g, 0)}" for g in GROUPS))
    header = f"{'Hệ thống':<14}" + "".join(f"{f'{g} R@10':>10}{f'{g} nDCG':>10}" for g in GROUPS)
    print(header)
    for name, per_group in breakdown.items():
        cells = ""
        for group in GROUPS:
            scores = per_group.get(group)
            if scores is None:
                cells += f"{'—':>10}{'—':>10}"
            else:
                cells += f"{scores[f'recall@{RECALL_K}']:>10.4f}{scores['ndcg@10']:>10.4f}"
        print(f"{name:<14}{cells}")


def main() -> None:
    force_utf8_stdout()

    parser = argparse.ArgumentParser(description="Phân tích lỗi Sprint 4 (Mục 5.4)")
    parser.add_argument(
        "--max-cases", type=int, default=20, help="số ca lỗi tối đa (đề cương: 15-20)"
    )
    parser.add_argument(
        "--no-write", action="store_true", help="chỉ in ra, không ghi data/eval/error_cases.json"
    )
    args = parser.parse_args()

    from scripts.compute_baseline_results import load_labels, load_runs

    labels = load_labels()
    runs = load_runs()
    queries = load_queries()
    product_meta = load_product_meta()
    grades = _grade_lookup(labels)
    print(f"Nhãn: {len(labels)} | hệ thống: {', '.join(runs)} | truy vấn: {len(queries)}")

    for name in (FINAL_SYSTEM, REFERENCE_SYSTEM):
        if name not in runs:
            raise SystemExit(f"thiếu hệ thống '{name}' trong pool_top20.json")

    query_ids = sorted_query_ids(runs[FINAL_SYSTEM])
    if missing := [qid for qid in query_ids if qid not in queries]:
        raise SystemExit(
            f"{len(missing)} truy vấn trong run không có trong queries.json: {missing[:5]}"
        )

    final_scores = per_query_scores(runs[FINAL_SYSTEM], grades, query_ids)
    reference_scores = per_query_scores(runs[REFERENCE_SYSTEM], grades, query_ids)

    metrics = RankingMetrics(min_relevant_grade=MIN_RELEVANT_GRADE)
    for name, scores in ((FINAL_SYSTEM, final_scores), (REFERENCE_SYSTEM, reference_scores)):
        mean_recall = sum(s.recall_at_k for s in scores.values()) / len(scores)
        aggregate = metrics.recall_at_k(runs[name], labels, k=RECALL_K)
        if abs(mean_recall - aggregate) > 1e-9:
            raise AssertionError(
                f"{name}: Recall@{RECALL_K} theo truy vấn ({mean_recall:.6f}) khác "
                f"RankingMetrics ({aggregate:.6f})"
            )
        print(f"  {name}: Recall@{RECALL_K} = {mean_recall:.4f} (khớp bảng kết quả chính)")

    bucket_counts = Counter(
        bucket
        for qid in query_ids
        if (bucket := bucket_of(final_scores[qid], reference_scores[qid])) is not None
    )
    cases = select_error_cases(final_scores, reference_scores, args.max_cases)
    print_selection_report(cases, bucket_counts, args.max_cases, len(query_ids))

    spread = grades_by_annotator(labels)
    case_rows: list[dict[str, Any]] = []
    for case in cases:
        query = queries[case.query_id]
        case_rows.append(
            {
                "query_id": case.query_id,
                "query": query["text"],
                "group": query["group"],
                "bucket": case.bucket,
                f"final_recall@{RECALL_K}": case.final.recall_at_k,
                "final_reciprocal_rank": case.final.reciprocal_rank,
                f"zeroshot_recall@{RECALL_K}": case.reference.recall_at_k,
                "zeroshot_reciprocal_rank": case.reference.reciprocal_rank,
                "final_top5": describe_top_hits(
                    runs[FINAL_SYSTEM][case.query_id], case.query_id, grades, spread, product_meta
                ),
                "zeroshot_top5": describe_top_hits(
                    runs[REFERENCE_SYSTEM][case.query_id],
                    case.query_id,
                    grades,
                    spread,
                    product_meta,
                ),
            }
        )

    zero_ceiling = queries_with_zero_ceiling(query_ids, grades)
    group_sizes = Counter(queries[qid]["group"] for qid in query_ids)
    breakdown = group_breakdown(runs, labels, queries, zero_ceiling)
    print_group_report(breakdown, group_sizes)

    accessory = accessory_target_queries(queries)
    accessory_ids = frozenset(accessory)
    axis_breakdown = target_axis_breakdown(runs, labels, accessory_ids)
    print("\n=== Recall@10 / nDCG@10 theo trục phụ kiện vs trang phục chính ===")
    n_accessory = len(accessory_ids & set(query_ids))
    print(f"Phụ kiện: {n_accessory} truy vấn | trang phục chính: {len(query_ids) - n_accessory}")
    print(f"{'Hệ thống':<14}{'PK R@10':>10}{'PK nDCG':>10}{'TPC R@10':>10}{'TPC nDCG':>10}")
    for name, per_axis in axis_breakdown.items():
        cells = ""
        for axis in ("phu_kien", "trang_phuc_chinh"):
            axis_scores = per_axis.get(axis)
            cells += (
                f"{'—':>10}{'—':>10}"
                if axis_scores is None
                else f"{axis_scores[f'recall@{RECALL_K}']:>10.4f}{axis_scores['ndcg@10']:>10.4f}"
            )
        print(f"{name:<14}{cells}")

    unanswerable = unrepresentable_queries(queries)
    zero_ceiling_by_group = Counter(queries[qid]["group"] for qid in zero_ceiling)
    print("\n=== Truy vấn không hệ thống nào trả lời được ===")
    print(
        f"Trần 0 (không nhãn nào >= {MIN_RELEVANT_GRADE} trong toàn pool top-20): "
        f"{len(zero_ceiling)}/{len(query_ids)} — theo nhóm: "
        + ", ".join(f"{g}={zero_ceiling_by_group.get(g, 0)}" for g in GROUPS)
    )
    n_zero_ceiling_accessory = len(zero_ceiling & accessory_ids)
    print(
        f"Trong đó danh từ trung tâm là phụ kiện: {n_zero_ceiling_accessory}/{len(zero_ceiling)}"
    )
    print(
        f"Truy vấn phụ kiện hỏi kiểu con mà 46 danh mục gộp vào 1 ô (không diễn đạt được): "
        f"{n_accessory}"
    )
    print(
        f"Truy vấn đòi trang phục truyền thống Việt Nam (áo dài, áo bà ba...): {len(unanswerable)}"
    )
    for query_id, row in sorted(unanswerable.items(), key=lambda kv: int(kv[0])):
        flag = "trần 0" if query_id in zero_ceiling else "vẫn có nhãn >= 1"
        print(f"  q{query_id} [{row['group']}] {row['text']} — {row['term']} ({flag})")
    print(f"Danh mục needs_review trong category_mapping.json: {len(needs_review_categories())}")

    error_by_group = Counter(row["group"] for row in case_rows)
    print("\nPhân bố ca lỗi đã chọn theo nhóm: " + ", ".join(
        f"{g}={error_by_group.get(g, 0)}" for g in GROUPS
    ))

    payload: dict[str, Any] = {
        "_meta": {
            "final_system": FINAL_SYSTEM,
            "reference_system": REFERENCE_SYSTEM,
            "k": RECALL_K,
            "min_relevant_grade": MIN_RELEVANT_GRADE,
            "n_queries": len(query_ids),
            "selection_rule": (
                f"Ưu tiên rổ 1 '{BUCKET_FINAL_ONLY}' (final Recall@{RECALL_K}=0, zero-shot "
                "có hit), "
                f"rổ 2 '{BUCKET_BOTH_FAIL}' (cả hai Recall@{RECALL_K}=0), "
                f"rổ 3 '{BUCKET_REGRESSION}' (final có hit nhưng rr thấp hơn zero-shot); "
                "trong mỗi rổ xếp theo rr(zero-shot) giảm dần rồi rr(final) tăng dần rồi id."
            ),
            "generated_by": "python -m scripts.error_analysis",
        },
        "bucket_counts": dict(bucket_counts),
        "cases": case_rows,
        "cases_by_group": dict(error_by_group),
        "group_sizes": dict(group_sizes),
        "group_breakdown": breakdown,
        "target_axis_breakdown": axis_breakdown,
        "zero_ceiling_queries": {
            "count": len(zero_ceiling),
            "by_group": dict(zero_ceiling_by_group),
            "accessory_head_count": n_zero_ceiling_accessory,
            "query_ids": sorted(zero_ceiling, key=int),
        },
        "accessory_target_queries": {
            "count": n_accessory,
            "queries": accessory,
        },
        "unrepresentable_queries": unanswerable,
        "needs_review_categories": needs_review_categories(),
    }

    if args.no_write:
        print(f"\n--no-write => KHÔNG ghi {OUT_PATH.name}")
    else:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu {OUT_PATH}")


if __name__ == "__main__":
    main()
