"""Sprint 4 deliverable: 3-component search latency (p50/p95) for every system.

Kế hoạch Sprint 4 (Hiệp) asks for "bảng latency 3 thành phần (p50/p95) cho tất cả
hệ thống". The three components measured per query are:

1. ``encode_query``  — turning the Vietnamese query text into a vector
                       (for BM25: tokenising it, there is no vector),
2. ``index_search``  — the vector store / lexical index lookup itself,
3. ``end_to_end``    — the whole ``retriever.search(query)`` call.

Measurement hygiene (the numbers are worthless without it):

* Every query is timed INDIVIDUALLY, one at a time, no batching — that is what a
  single real user request costs. Batched throughput would look much better and
  would not answer "how long does one search take?".
* Each system is warmed up (``--warmup`` queries, discarded) before timing. The
  first call pays model load, CUDA context init and lazy allocations; quoting it
  as steady-state latency inflates the table by orders of magnitude.
* ``time.perf_counter()`` around each region, with ``torch.cuda.synchronize()``
  before starting AND before stopping the clock on CUDA systems. Without the
  sync you time asynchronous kernel *launches*, not their execution — the single
  most common way GPU latency numbers come out fabricated-fast.
* The device each system actually ran on is recorded per system. They are NOT
  all the same: ``visiglip_ot`` deliberately runs on CPU on this machine because
  its image tower returns NaN on this GPU (see the module docstring of
  ``src/adapters/embedders/visiglip_adapter.py``), so its latency is not
  comparable to the CUDA rows. Reported honestly instead of hidden.
* Catalog, query set, top-k depth and vector store are the same ones that
  produced the accuracy table: data/processed/test.json (6,963 held-out
  products), data/eval/queries.json (155 Vietnamese queries), top-20,
  InMemoryVectorStore. Qdrant would be a different (approximate) search and the
  latency would no longer line up with the accuracy numbers.

BM25 caveat, stated because it shows up in the table: ``BM25Retriever.search``
has no public seam between tokenising and scoring, so ``index_search`` for BM25
times the built ``rank_bm25`` index's ``get_scores`` call directly. That covers
the scoring kernel but not the filter + top-k sort that ``search`` still does
afterwards (for the dense systems that sort happens inside ``store.search`` and
IS included). The leftover is printed as a ``residual`` column so the gap is
visible rather than silently swallowed.

Usage:
    python -m scripts.measure_latency --limit 200   # fast smoke run
    python -m scripts.measure_latency               # full pass, writes JSON
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapters.lexical.bm25_adapter import BM25Retriever  # noqa: E402
from src.adapters.lexical.bm25_adapter import _tokenize as bm25_tokenize  # noqa: E402
from src.core.models import Product, Query  # noqa: E402

OUT_PATH = ROOT / "data" / "eval" / "latency_results.json"
TOP_K = 20
STORE_NAME = "InMemoryVectorStore"
ALL_SYSTEMS = ("bm25", "clip", "siglip2", "visiglip_ot", "siglip2_lora")
COMPONENTS = ("encode_query", "index_search", "end_to_end")
ADAPTER_PATH = ROOT / "checkpoints" / "hung-run1" / "best"


@dataclass
class System:
    """One measurable system: its retriever plus how/where it actually runs."""

    key: str
    retriever: Any
    device: str
    backend: str


def force_utf8_stdout() -> None:
    """Make the Windows console print Vietnamese instead of raising.

    ``reconfigure``, NOT a fresh ``io.TextIOWrapper`` around ``sys.stdout.buffer``
    like the older scripts do: wrapping a stdout that a previously imported
    script already wrapped closes the inner wrapper as soon as it is collected,
    and every later print dies with "I/O operation on closed file".
    """
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")


def percentile(samples: Sequence[float], q: float) -> float:
    """``q``-th percentile with linear interpolation (numpy's default method).

    Hand-written instead of ``np.percentile`` so the latency table has no hidden
    dependency on which interpolation flavour a numpy version defaults to.
    """
    if not samples:
        raise ValueError("need at least one sample")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be within 0..100, got {q}")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * q / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * weight


def summarise(samples: Sequence[float]) -> dict[str, float]:
    """``n`` / p50 / p95 / mean of one component's per-query timings, in ms."""
    return {
        "n": len(samples),
        "p50_ms": percentile(samples, 50.0),
        "p95_ms": percentile(samples, 95.0),
        "mean_ms": sum(samples) / len(samples),
    }


def residual_samples(measured: dict[str, list[float]]) -> list[float]:
    """Per-query ``end_to_end - encode_query - index_search`` (ms).

    Everything the end-to-end call does outside the two measured components:
    for the dense systems that is only glue, for BM25 it is the filter + sort.
    """
    return [
        total - encode - search
        for total, encode, search in zip(
            measured["end_to_end"],
            measured["encode_query"],
            measured["index_search"],
            strict=True,
        )
    ]


def device_of(embedder: Any) -> str:
    """Which device an embedder settled on AFTER it has been loaded.

    The adapters resolve ``None`` -> "cuda"/"cpu" lazily inside their loader, and
    ViSigLIP-OT keeps its choice on a private ``_device`` instead (it overrides
    the auto-detect convention on purpose).
    """
    device = getattr(embedder, "device", None) or getattr(embedder, "_device", None)
    return str(device) if device else "unknown"


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _time_calls(
    device: str,
    queries: Sequence[Query],
    steps: dict[str, Any],
    warmup: int,
) -> dict[str, list[float]]:
    """Time each ``steps`` callable once per query, after ``warmup`` dry queries.

    ``steps`` maps a component name to a ``callable(query) -> object``; the
    return value is only kept alive long enough to be handed to the next step
    through the closures the caller builds.
    """
    for query in queries[:warmup]:
        for step in steps.values():
            step(query)

    samples: dict[str, list[float]] = {name: [] for name in steps}
    for query in queries:
        for name, step in steps.items():
            _sync(device)
            start = time.perf_counter()
            step(query)
            _sync(device)
            samples[name].append((time.perf_counter() - start) * 1000.0)
    return samples


def measure_bm25(system: System, queries: Sequence[Query], warmup: int) -> dict[str, list[float]]:
    retriever: BM25Retriever = system.retriever
    index = retriever._bm25  # no public seam between tokenising and scoring
    assert index is not None, "BM25Retriever was not indexed"

    tokens: list[str] = []

    def encode(query: Query) -> None:
        nonlocal tokens
        tokens = bm25_tokenize(query.normalized_text())

    steps = {
        "encode_query": encode,
        "index_search": lambda query: index.get_scores(tokens),
        "end_to_end": retriever.search,
    }
    return _time_calls(system.device, queries, steps, warmup)


def measure_dense(system: System, queries: Sequence[Query], warmup: int) -> dict[str, list[float]]:
    retriever = system.retriever
    embedder = retriever.embedder
    store = retriever.store

    vector: list[float] = []

    def encode(query: Query) -> None:
        nonlocal vector
        vector = embedder.encode_text([query.normalized_text()])[0]

    steps = {
        "encode_query": encode,
        "index_search": lambda query: store.search(vector, TOP_K, None),
        "end_to_end": retriever.search,
    }
    return _time_calls(system.device, queries, steps, warmup)


def build_system(key: str, products: Sequence[Product], batch_size: int) -> System:
    """Index ``products`` for one system, exactly as the accuracy runs did it."""
    from scripts.run_finetuned_eval import index_by_image

    if key == "bm25":
        bm25 = BM25Retriever()
        bm25.index(products)
        return System(key=key, retriever=bm25, device="cpu", backend="rank_bm25.BM25Okapi")

    embedder: Any
    if key == "clip":
        from src.adapters.embedders.clip_adapter import ClipEmbedder

        embedder = ClipEmbedder(batch_size=batch_size)
    elif key == "siglip2":
        from src.adapters.embedders.siglip2_adapter import Siglip2Embedder

        embedder = Siglip2Embedder(batch_size=batch_size)
    elif key == "siglip2_lora":
        from src.adapters.embedders.siglip2_adapter import Siglip2Embedder

        embedder = Siglip2Embedder(adapter_path=str(ADAPTER_PATH), batch_size=batch_size)
    elif key == "visiglip_ot":
        from src.adapters.embedders.visiglip_adapter import ViSiglipEmbedder

        embedder = ViSiglipEmbedder(batch_size=batch_size)
    else:
        raise ValueError(f"unknown system {key!r}; expected one of {ALL_SYSTEMS}")

    print(f"  index {len(products)} ảnh ({embedder.name}, {embedder.dim}-d)...", flush=True)
    started = time.perf_counter()
    retriever = index_by_image(embedder, list(products))
    print(f"  index xong trong {time.perf_counter() - started:.1f}s", flush=True)
    return System(key=key, retriever=retriever, device=device_of(embedder), backend=embedder.name)


def print_table(results: dict[str, Any]) -> None:
    print(
        f"\nLatency 3 thành phần, mỗi truy vấn đo riêng lẻ (ms) | "
        f"catalog {results['catalog']['n_products']} sản phẩm | "
        f"{results['queries']['n']} truy vấn | top-{results['top_k']} | "
        f"store: {results['vector_store']}"
    )
    header = (
        f"{'Hệ thống':<14}{'Thiết bị':<10}"
        f"{'encode p50':>12}{'encode p95':>12}"
        f"{'index p50':>12}{'index p95':>12}"
        f"{'tổng p50':>12}{'tổng p95':>12}"
        f"{'residual p50':>14}"
    )
    print(header)
    print("-" * len(header))
    for key, row in results["systems"].items():
        components = row["components"]
        print(
            f"{key:<14}{row['device']:<10}"
            f"{components['encode_query']['p50_ms']:>12.2f}"
            f"{components['encode_query']['p95_ms']:>12.2f}"
            f"{components['index_search']['p50_ms']:>12.2f}"
            f"{components['index_search']['p95_ms']:>12.2f}"
            f"{components['end_to_end']['p50_ms']:>12.2f}"
            f"{components['end_to_end']['p95_ms']:>12.2f}"
            f"{row['residual']['p50_ms']:>14.2f}"
        )
    print(
        "\nGhi chú: 'residual' = tổng - encode - index (glue của DenseRetriever; "
        "với bm25 là bước lọc + sắp xếp top-k nằm ngoài get_scores)."
    )
    devices = {row["device"] for row in results["systems"].values()}
    if len(devices) > 1:
        print(
            "CẢNH BÁO: các hệ thống KHÔNG chạy trên cùng thiết bị "
            f"({', '.join(sorted(devices))}) — không so sánh trực tiếp giữa các dòng "
            "khác thiết bị. visiglip_ot buộc phải chạy CPU do lỗi NaN của image tower "
            "trên GPU máy này (xem docstring visiglip_adapter.py)."
        )


def main() -> None:
    force_utf8_stdout()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chỉ index N sản phẩm đầu (smoke run); khi có cờ này KHÔNG ghi file kết quả",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(ALL_SYSTEMS),
        choices=list(ALL_SYSTEMS),
        help="chọn tập hệ thống cần đo (mặc định: tất cả)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch ảnh khi index (giảm nếu CUDA OOM; Sprint 2 dùng 32)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="số truy vấn chạy nháp trước khi bấm giờ (loại bỏ cold start)",
    )
    args = parser.parse_args()

    from scripts.run_finetuned_eval import load_catalog, load_queries

    products = load_catalog(args.limit)
    raw_queries = load_queries()
    queries = [Query(query_id=str(q["id"]), text=q["text"], top_k=TOP_K) for q in raw_queries]
    print(
        f"Catalog: {len(products)} sản phẩm | Truy vấn: {len(queries)} | top-{TOP_K} "
        f"| store: {STORE_NAME} | warmup: {args.warmup}"
    )

    results: dict[str, Any] = {
        "catalog": {"path": "data/processed/test.json", "n_products": len(products)},
        "queries": {"path": "data/eval/queries.json", "n": len(queries)},
        "top_k": TOP_K,
        "vector_store": STORE_NAME,
        "warmup_queries": args.warmup,
        "batched": False,
        "limit": args.limit,
        "systems": {},
    }

    for key in args.systems:
        print(f"\n== {key} ==", flush=True)
        system = build_system(key, products, args.batch_size)
        measure = measure_bm25 if key == "bm25" else measure_dense
        samples = measure(system, queries, args.warmup)
        results["systems"][key] = {
            "device": system.device,
            "backend": system.backend,
            "components": {name: summarise(samples[name]) for name in COMPONENTS},
            "residual": summarise(residual_samples(samples)),
        }
        print(
            f"  tổng p50 = {results['systems'][key]['components']['end_to_end']['p50_ms']:.2f} ms "
            f"| p95 = {results['systems'][key]['components']['end_to_end']['p95_ms']:.2f} ms "
            f"| device = {system.device}",
            flush=True,
        )

    print_table(results)

    if args.limit is not None:
        print(
            f"\n--limit {args.limit} => smoke run, KHÔNG ghi {OUT_PATH.name}. "
            "Chạy lại không có --limit để lấy số thật."
        )
    else:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu {OUT_PATH}")


if __name__ == "__main__":
    main()
