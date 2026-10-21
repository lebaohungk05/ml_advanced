"""Build the blind-pooled labeling worksheet from data/eval/pool_top20.json.

DeCuong Mục 3.4 điểm 3-4: gộp top-20 của mỗi hệ thống thành 1 danh sách chung,
xáo trộn và ẨN NGUỒN trước khi chấm — người chấm không được biết hệ thống nào
trả về kết quả nào, để không thiên vị. Output: data/eval/relevance_template.json,
1 dòng / (query, product) cần chấm, "grade" để trống chờ điền 0/1/2.

Run once data/eval/pool_top20.json exists (produced by
scripts/run_sprint2_baselines.py):

    python scripts/build_labeling_pool.py
"""

from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = ROOT / "data" / "eval" / "pool_top20.json"
QUERIES_PATH = ROOT / "data" / "eval" / "queries.json"
CATALOG_PATH = ROOT / "data" / "processed" / "test.json"
OUT_PATH = ROOT / "data" / "eval" / "relevance_template.json"
SEED = 42


def main() -> None:
    if not POOL_PATH.exists():
        raise SystemExit(
            f"{POOL_PATH} chưa tồn tại — chạy scripts/run_sprint2_baselines.py trước"
        )

    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)  # {system: {query_id: [{"product_id","score","rank"}, ...]}}
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = {str(q["id"]): q["text"] for q in json.load(f)["queries"]}
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = {p["product_id"]: p for p in json.load(f)}

    rng = random.Random(SEED)
    systems = sorted(pool.keys())
    template = []

    for qid, qtext in queries.items():
        seen_products: set[str] = set()
        candidates = []
        for system in systems:
            for hit in pool.get(system, {}).get(qid, []):
                pid = hit["product_id"]
                if pid in seen_products:
                    continue
                seen_products.add(pid)
                candidates.append(pid)  # source system deliberately NOT recorded here
        rng.shuffle(candidates)

        for pid in candidates:
            product = catalog.get(pid)
            if product is None:
                continue
            template.append(
                {
                    "query_id": qid,
                    "query_text": qtext,
                    "product_id": pid,
                    "image_path": product["image_path"],
                    "category": product["category"],
                    "grade": None,  # 0/1/2 — điền vào đây (xem research/error-analysis-template.md
                    # để biết tiêu chí, và docs/DeCuong Mục 3.4 cho định nghĩa 3 mức)
                    "annotator": None,
                }
            )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=1)

    n_queries = len(queries)
    n_rows = len(template)
    print(
        f"{n_queries} truy vấn -> {n_rows} dòng cần chấm "
        f"(đã gộp {len(systems)} hệ thống: {systems})"
    )
    print(f"Trung bình {n_rows / max(n_queries, 1):.1f} sản phẩm/truy vấn sau khi gộp trùng")
    print(f"Đã lưu: {OUT_PATH}")
    print(
        "\nGợi ý chấm: mỗi (query, product) cần ít nhất 2 người chấm độc lập (Mục 3.4 điểm 4) "
        "-- có thể copy file này thành relevance_<tên>.json cho từng người chấm riêng, "
        "rồi tính Cohen's kappa để xem có đồng thuận không trước khi gộp lại."
    )


if __name__ == "__main__":
    main()
