"""Evaluate one SigLIP-family system on the same 155-query eval set as the Sprint 2
baselines and merge its top-20 run into data/eval/pool_top20.json.

Sprint 3 used it for the fine-tuned SigLIP2+DoRA checkpoint (key ``siglip2_lora``,
the defaults below); Sprint 4 axis 7 (SigLIP 1 vs SigLIP 2) reuses it unchanged
via ``--model-id`` / ``--system-name`` / ``--no-adapter``, because that axis is
answerable by a zero-shot inference pass and needs no training.

Deliberately a copy of scripts/run_sprint2_baselines.py's mechanics rather than a
refactor of it: the comparison is only apples-to-apples if the evaluated system
is indexed exactly the way the baselines were —

* same catalog: data/processed/test.json (6,963 held-out products, never seen in
  training),
* indexed BY IMAGE, bypassing DenseRetriever.index() (frozen core, text-only),
* same InMemoryVectorStore (NOT Qdrant — the Sprint 2 numbers came from the
  in-memory store, and swapping the store would confound the comparison),
* same 155 queries, same top-20 depth.

So the only difference from the ``siglip2`` row is the flags passed here.

!! Systems evaluated through this script are POST-POOL: the relevance pool was
built from the top-20 of bm25/clip/siglip2/visiglip_ot (plus a second round on
the fine-tuned system's top-10). Their top-K can therefore contain pairs nobody
graded, which every metric scores as 0, so their numbers are a LOWER BOUND until
the pool is extended. Always read the ``unlabelled_hits_at_k`` table printed by
scripts/compute_baseline_results.py before quoting a result from here.

Usage:
    python -m scripts.run_finetuned_eval --limit 200   # cheap smoke, no write
    python -m scripts.run_finetuned_eval               # full run, merges result
    # Sprint 4 axis 7 (SigLIP 1, zero-shot):
    python -m scripts.run_finetuned_eval --no-adapter --system-name siglip1
        --model-id google/siglip-base-patch16-256
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapters.embedders.siglip2_adapter import (  # noqa: E402
    DEFAULT_MODEL_ID,
    Siglip2Embedder,
)
from src.adapters.vectorstore.inmemory_adapter import InMemoryVectorStore  # noqa: E402
from src.core.dense_retriever import DenseRetriever  # noqa: E402
from src.core.models import Product, Query, SearchResult  # noqa: E402

RAW = ROOT / "data" / "raw" / "fashionpedia"
CATALOG_PATH = ROOT / "data" / "processed" / "test.json"
QUERIES_PATH = ROOT / "data" / "eval" / "queries.json"
POOL_PATH = ROOT / "data" / "eval" / "pool_top20.json"
ADAPTER_PATH = ROOT / "checkpoints" / "hung-run1" / "best"
DEFAULT_SYSTEM_KEY = "siglip2_lora"
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


def merge_into_pool(
    run: dict[str, list[dict[str, Any]]], system_key: str, pool_path: Path
) -> None:
    """Add ``system_key`` to pool_top20.json, leaving every other system untouched.

    Read-modify-write on the parsed dict: the existing systems' entries are the
    same objects that came out of json.load, so their content round-trips
    unchanged (same dump options as run_sprint2_baselines.py).
    """
    with open(pool_path, encoding="utf-8") as f:
        pool = json.load(f)
    existing = [k for k in pool if k != system_key]
    pool[system_key] = run
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False)
    print(f"Đã ghi '{system_key}' vào {pool_path} (giữ nguyên: {', '.join(existing)})")


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="backbone HuggingFace; trục 7 dùng google/siglip-base-patch16-256",
    )
    parser.add_argument(
        "--system-name",
        default=DEFAULT_SYSTEM_KEY,
        help=f"key ghi vào {POOL_PATH.name} (mặc định {DEFAULT_SYSTEM_KEY})",
    )
    adapter = parser.add_mutually_exclusive_group()
    adapter.add_argument(
        "--adapter-path",
        default=str(ADAPTER_PATH),
        help="checkpoint LoRA/DoRA chồng lên backbone",
    )
    adapter.add_argument(
        "--no-adapter",
        action="store_true",
        help="chạy zero-shot, không nạp adapter (trục 7)",
    )
    return parser


def resolve_adapter_path(args: argparse.Namespace) -> str | None:
    """``None`` means zero-shot; anything else is a LoRA/DoRA checkpoint to stack on.

    Getting this wrong is silent: SigLIP 1 loaded with the SigLIP 2 DoRA adapter
    would still produce a full run file, just a meaningless one.
    """
    return None if args.no_adapter else str(args.adapter_path)


def main() -> None:
    # reconfigure, not a fresh wrapper: see measure_latency.force_utf8_stdout.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()

    adapter_path = resolve_adapter_path(args)
    products = load_catalog(args.limit)
    queries = load_queries()
    print(
        f"Catalog: {len(products)} sản phẩm | Truy vấn: {len(queries)} "
        f"| adapter: {adapter_path or 'zero-shot'}"
    )

    embedder = Siglip2Embedder(
        model_id=args.model_id, adapter_path=adapter_path, batch_size=args.batch_size
    )
    print(
        f"\n== {args.system_name} == "
        f"({embedder.name}, {embedder.dim}-d, batch={embedder.batch_size})"
    )

    t0 = time.time()
    retriever = index_by_image(embedder, products)
    encode_seconds = time.time() - t0
    print(f"  index xong trong {encode_seconds:.1f}s ({encode_seconds / 60:.1f} phút)")

    run = run_system(args.system_name, retriever, queries)

    if args.limit is not None:
        print(
            f"\n--limit {args.limit} => smoke run, KHÔNG ghi vào {POOL_PATH.name}. "
            "Chạy lại không có --limit để lấy số thật."
        )
    else:
        merge_into_pool(run, args.system_name, POOL_PATH)

    catalog_by_id = {p.product_id: p for p in products}
    for q in queries[:5]:
        top3 = run[str(q["id"])][:3]
        titles = [catalog_by_id[h["product_id"]].title for h in top3]
        print(f"\nTruy vấn: \"{q['text']}\" (nhóm {q['group']})")
        print(f"  {args.system_name:14s}: {titles}")


if __name__ == "__main__":
    main()
