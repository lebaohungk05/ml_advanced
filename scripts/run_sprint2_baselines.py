"""Sprint 2 deliverable: index the catalog + run all queries through each
baseline, save top-20 per system for blind pooling (labeling happens Sprint 3).

Catalog = data/processed/test.json (held-out split, 6,963 products) — kept
separate from train/val so a Sprint 3 fine-tuned checkpoint is evaluated on
images it never saw during training.

DenseRetriever.index() (src/core/dense_retriever.py, frozen) embeds
Product.to_text() only — fine for the tiny text-only demo catalog, wrong for a
real image catalog. Rather than touch frozen core code, this script indexes
image-capable embedders by IMAGE embedding directly (bypassing .index()) so
retrieval is genuine text-query -> image-catalog cross-modal search, then
reuses DenseRetriever only for the query side (`.search()`), which already
picks the right encode method per query modality.

ResNet-50+KNN is excluded here: it's an image-only baseline and this query set
(data/eval/queries.json) is 100% Vietnamese text — no sample query images were
sourced. SigLIP2+machine-translation is excluded per the earlier scope cut
(no MT model set up, and it was already flagged as a system worth dropping).
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapters.embedders.clip_adapter import ClipEmbedder  # noqa: E402
from src.adapters.embedders.siglip2_adapter import Siglip2Embedder  # noqa: E402
from src.adapters.embedders.visiglip_adapter import ViSiglipEmbedder  # noqa: E402
from src.adapters.lexical.bm25_adapter import BM25Retriever  # noqa: E402
from src.adapters.vectorstore.inmemory_adapter import InMemoryVectorStore  # noqa: E402
from src.core.dense_retriever import DenseRetriever  # noqa: E402
from src.core.models import Product, Query, SearchResult  # noqa: E402

RAW = ROOT / "data" / "raw" / "fashionpedia"
CATALOG_PATH = ROOT / "data" / "processed" / "test.json"
QUERIES_PATH = ROOT / "data" / "eval" / "queries.json"
OUT_PATH = ROOT / "data" / "eval" / "pool_top20.json"
TOP_K = 20


def load_catalog() -> list[Product]:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        raw = json.load(f)
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


def load_queries() -> list[dict]:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        return json.load(f)["queries"]


def image_bytes_for(product: Product) -> bytes:
    assert product.image_path is not None
    return (RAW / product.image_path).read_bytes()


def index_by_image(embedder, products: list[Product]) -> DenseRetriever:  # type: ignore[no-untyped-def]
    store = InMemoryVectorStore(dim=embedder.dim)
    batch = 32
    for i in range(0, len(products), batch):
        chunk = products[i : i + batch]
        vectors = embedder.encode_image([image_bytes_for(p) for p in chunk])
        store.upsert(chunk, vectors)
        if (i + batch) % 500 == 0 or i + batch >= len(products):
            print(f"    ... {min(i + batch, len(products))}/{len(products)}")
    return DenseRetriever(embedder=embedder, store=store, name=embedder.name)


def run_system(name: str, retriever, queries: list[dict]) -> dict[str, list[dict]]:  # type: ignore[no-untyped-def]
    print(f"[{name}] chạy {len(queries)} truy vấn...")
    t0 = time.time()
    results: dict[str, list[dict]] = {}
    for q in queries:
        query = Query(query_id=str(q["id"]), text=q["text"], top_k=TOP_K)
        try:
            hits: list[SearchResult] = retriever.search(query)
        except NotImplementedError:
            hits = []
        results[str(q["id"])] = [
            {"product_id": h.product_id, "score": h.score, "rank": h.rank} for h in hits
        ]
    print(f"[{name}] xong trong {time.time() - t0:.1f}s")
    return results


def main() -> None:
    products = load_catalog()
    queries = load_queries()
    print(f"Catalog: {len(products)} sản phẩm | Truy vấn: {len(queries)}")

    pool: dict[str, dict[str, list[dict]]] = {}

    print("\n== BM25 ==")
    bm25 = BM25Retriever()
    bm25.index(products)
    pool["bm25"] = run_system("bm25", bm25, queries)

    for label, embedder in [
        ("clip", ClipEmbedder()),
        ("siglip2", Siglip2Embedder()),
        ("visiglip_ot", ViSiglipEmbedder()),
    ]:
        print(f"\n== {label} == (encoding catalog images, {embedder.dim}-d)")
        t0 = time.time()
        retriever = index_by_image(embedder, products)
        print(f"  index xong trong {time.time() - t0:.1f}s")
        pool[label] = run_system(label, retriever, queries)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False)
    print(f"\nĐã lưu kết quả pool top-{TOP_K} vào {OUT_PATH}")

    # Small human-readable preview for a handful of queries.
    catalog_by_id = {p.product_id: p for p in products}
    preview_lines = []
    for q in queries[:5]:
        qid = str(q["id"])
        preview_lines.append(f"\nTruy vấn: \"{q['text']}\" (nhóm {q['group']})")
        for system in pool:
            top3 = pool[system][qid][:3]
            titles = [catalog_by_id[h["product_id"]].title for h in top3]
            preview_lines.append(f"  {system:12s}: {titles}")
    preview_text = "\n".join(preview_lines)
    print(preview_text)
    with open(ROOT / "data" / "eval" / "preview_top3.txt", "w", encoding="utf-8") as f:
        f.write(preview_text)


if __name__ == "__main__":
    main()
