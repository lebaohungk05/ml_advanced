"""Sprint 3 deliverable: evaluate the fine-tuned SigLIP2+DoRA checkpoint on the
same 155-query eval set as the Sprint 2 baselines, and merge its top-20 run into
data/eval/pool_top20.json under the key ``siglip2_lora``.

Deliberately a copy of scripts/run_sprint2_baselines.py's mechanics rather than a
refactor of it: the comparison is only apples-to-apples if the fine-tuned system
is indexed exactly the way the baselines were —

* same catalog: data/processed/test.json (6,963 held-out products, never seen in
  training),
* indexed BY IMAGE, bypassing DenseRetriever.index() (frozen core, text-only),
* same InMemoryVectorStore (NOT Qdrant — the Sprint 2 numbers came from the
  in-memory store, and swapping the store would confound the comparison),
* same 155 queries, same top-20 depth.

The only difference from the ``siglip2`` row is ``adapter_path``.

Usage:
    python -m scripts.run_finetuned_eval --limit 200   # cheap smoke, no write
    python -m scripts.run_finetuned_eval               # full run, merges result
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapters.embedders.siglip2_adapter import Siglip2Embedder  # noqa: E402
from src.adapters.vectorstore.inmemory_adapter import InMemoryVectorStore  # noqa: E402
from src.core.dense_retriever import DenseRetriever  # noqa: E402
from src.core.models import Product, Query, SearchResult  # noqa: E402

RAW = ROOT / "data" / "raw" / "fashionpedia"
CATALOG_PATH = ROOT / "data" / "processed" / "test.json"
QUERIES_PATH = ROOT / "data" / "eval" / "queries.json"
POOL_PATH = ROOT / "data" / "eval" / "pool_top20.json"
ADAPTER_PATH = ROOT / "checkpoints" / "hung-run1" / "best"
SYSTEM_KEY = "siglip2_lora"
TOP_K = 20


def load_catalog(limit: int | None = None) -> list[Product]:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    if limit is not None:
        raw = raw[:limit]
    return [
        Product(
            product_id=r["product_id"],
            title=r["title"],
            category=r["category"],
            image_path=r["image_path"],
            attributes=r.get("attributes") or {},
        )
        for r in raw
    ]


def load_queries() -> list[dict[str, Any]]:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries: list[dict[str, Any]] = json.load(f)["queries"]
    return queries


def image_bytes_for(product: Product) -> bytes:
    assert product.image_path is not None
    return (RAW / product.image_path).read_bytes()


def index_by_image(embedder: Siglip2Embedder, products: list[Product]) -> DenseRetriever:
    store = InMemoryVectorStore(dim=embedder.dim)
    batch = embedder.batch_size
    for i in range(0, len(products), batch):
        chunk = products[i : i + batch]
        vectors = embedder.encode_image([image_bytes_for(p) for p in chunk])
        store.upsert(chunk, vectors)
        done = min(i + batch, len(products))
        if done % 500 < batch or done >= len(products):
            print(f"    ... {done}/{len(products)}", flush=True)
    return DenseRetriever(embedder=embedder, store=store, name=embedder.name)


def run_system(
    name: str, retriever: DenseRetriever, queries: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    print(f"[{name}] chạy {len(queries)} truy vấn...")
    t0 = time.time()
    results: dict[str, list[dict[str, Any]]] = {}
    for q in queries:
        query = Query(query_id=str(q["id"]), text=q["text"], top_k=TOP_K)
        hits: list[SearchResult] = retriever.search(query)
        results[str(q["id"])] = [
            {"product_id": h.product_id, "score": h.score, "rank": h.rank} for h in hits
        ]
    print(f"[{name}] xong trong {time.time() - t0:.1f}s")
    return results


def merge_into_pool(run: dict[str, list[dict[str, Any]]]) -> None:
    """Add ``siglip2_lora`` to pool_top20.json, leaving the 4 baselines untouched.

    Read-modify-write on the parsed dict: the existing systems' entries are the
    same objects that came out of json.load, so their content round-trips
    unchanged (same dump options as run_sprint2_baselines.py).
    """
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    existing = [k for k in pool if k != SYSTEM_KEY]
    pool[SYSTEM_KEY] = run
    with open(POOL_PATH, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False)
    print(f"Đã ghi '{SYSTEM_KEY}' vào {POOL_PATH} (giữ nguyên: {', '.join(existing)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chỉ index N sản phẩm đầu (smoke run); khi có cờ này KHÔNG ghi vào pool",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch ảnh khi encode (giảm xuống nếu CUDA OOM; Sprint 2 dùng 32)",
    )
    args = parser.parse_args()

    products = load_catalog(args.limit)
    queries = load_queries()
    print(
        f"Catalog: {len(products)} sản phẩm | Truy vấn: {len(queries)} "
        f"| adapter: {ADAPTER_PATH}"
    )

    embedder = Siglip2Embedder(adapter_path=str(ADAPTER_PATH), batch_size=args.batch_size)
    print(f"\n== {SYSTEM_KEY} == ({embedder.name}, {embedder.dim}-d, batch={embedder.batch_size})")

    t0 = time.time()
    retriever = index_by_image(embedder, products)
    encode_seconds = time.time() - t0
    print(f"  index xong trong {encode_seconds:.1f}s ({encode_seconds / 60:.1f} phút)")

    run = run_system(SYSTEM_KEY, retriever, queries)

    if args.limit is not None:
        print(
            f"\n--limit {args.limit} => smoke run, KHÔNG ghi vào {POOL_PATH.name}. "
            "Chạy lại không có --limit để lấy số thật."
        )
    else:
        merge_into_pool(run)

    catalog_by_id = {p.product_id: p for p in products}
    for q in queries[:5]:
        top3 = run[str(q["id"])][:3]
        titles = [catalog_by_id[h["product_id"]].title for h in top3]
        print(f"\nTruy vấn: \"{q['text']}\" (nhóm {q['group']})")
        print(f"  {SYSTEM_KEY:14s}: {titles}")


if __name__ == "__main__":
    main()
