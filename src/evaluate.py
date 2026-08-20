"""CLI: run one retrieval system over the query set and score it.

    python -m src.evaluate --config configs/default.yaml \
        --queries data/eval/queries.json --relevance data/eval/relevance.json

Pipeline: load queries + pooled 0/1/2 labels -> build the retriever from the
config -> search every query -> compute Recall@1/5/10, MRR, nDCG@10 -> append one
row to experiments/runs.csv (columns per DeCuong Mục 4.4).

Status: query/label loading, retriever construction, the search loop, the top-20
pooling dump and the runs.csv writer are real. Metric computation stops at
``RankingMetrics``, which is still a stub (Sprint 2, Hiếu).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from src.adapters.metrics.ranking_metrics import RankingMetrics
from src.core.models import Query, RelevanceLabel, SearchResult
from src.core.ports import Retriever, RunResults
from src.registry import build_retriever_from_config, load_config, products_from_config

logger = logging.getLogger("evaluate")

POOLING_DEPTH = 20
"""Top-20 per system goes into the blind pooling (DeCuong Mục 3.4)."""

RUNS_CSV_COLUMNS = (
    "run_id",
    "date",
    "run_by",
    "commit",
    "config",
    "lora_rank",
    "learning_rate",
    "effective_batch",
    "epochs",
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
    "ndcg@10",
    "runtime_s",
    "notes",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Evaluate one retrieval system on the shared query set.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"), help="retriever config"
    )
    parser.add_argument(
        "--queries", type=Path, default=Path("data/eval/queries.json"), help="query set"
    )
    parser.add_argument(
        "--relevance",
        type=Path,
        default=Path("data/eval/relevance.json"),
        help="pooled 0/1/2 labels; omit with --dump-pool to produce a pool to label",
    )
    parser.add_argument("--top-k", type=int, default=POOLING_DEPTH, help="results per query")
    parser.add_argument(
        "--dump-pool",
        type=Path,
        default=None,
        help="write this run's top-k to a JSON file for blind pooling, then exit",
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("experiments/runs.csv"),
        help="experiment log to append to",
    )
    parser.add_argument("--run-id", default=None, help="defaults to <retriever>-<timestamp>")
    parser.add_argument("--run-by", default="", help="who ran it (for the log)")
    parser.add_argument("--notes", default="", help="free-text note for the log")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def load_queries(path: Path) -> list[Query]:
    """Read ``queries.json``: a list of objects with ``query_id`` and ``text``.

    Extra keys (tier, author, ...) are kept out of the Query object but are fine
    to have in the file — the stratification lives in the eval notebook, not here.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw["queries"] if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list of queries")
    queries: list[Query] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: expected query objects")
        queries.append(
            Query(
                query_id=str(entry["query_id"]),
                text=entry.get("text"),
                image_path=entry.get("image_path"),
                top_k=int(entry.get("top_k", 10)),
                filters=dict(entry.get("filters") or {}),
            )
        )
    if not queries:
        raise ValueError(f"{path}: no queries found")
    return queries


def load_relevance(path: Path) -> list[RelevanceLabel]:
    """Read ``relevance.json``: flat list of {query_id, product_id, grade, annotator}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw["labels"] if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list of labels")
    return [
        RelevanceLabel(
            query_id=str(entry["query_id"]),
            product_id=str(entry["product_id"]),
            grade=int(entry["grade"]),
            annotator=entry.get("annotator"),
        )
        for entry in entries
    ]


def run_search(
    retriever: Retriever, queries: Sequence[Query], top_k: int
) -> dict[str, list[SearchResult]]:
    """Search every query, keeping per-query wall time out of the returned run."""
    run: dict[str, list[SearchResult]] = {}
    for query in queries:
        effective = Query(
            query_id=query.query_id,
            text=query.text,
            image_bytes=query.image_bytes,
            image_path=query.image_path,
            top_k=top_k,
            filters=query.filters,
        )
        run[query.query_id] = retriever.search(effective)
    return run


def dump_pool(run: RunResults, path: Path, retriever_name: str) -> None:
    """Write top-k hits for blind pooling: no scores, no system name per hit.

    Scores and provenance are stripped on purpose — annotators must not be able to
    tell which system produced a hit, or the labels are biased (DeCuong Mục 3.4).
    """
    payload = {
        "pool_source": retriever_name,
        "depth": max((len(hits) for hits in run.values()), default=0),
        "pool": {
            query_id: sorted(hit.product_id for hit in hits) for query_id, hits in run.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def git_commit() -> str:
    """Short commit hash, so a row in runs.csv can be traced back to code."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def append_run_row(path: Path, row: dict[str, Any]) -> None:
    """Append one experiment to runs.csv, writing the header if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RUNS_CSV_COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in RUNS_CSV_COLUMNS})


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    retriever = build_retriever_from_config(config)

    products = products_from_config(config)
    if products:
        retriever.index(products)
        logger.info("indexed %d inline demo products", len(products))
    else:
        raise NotImplementedError(
            "TODO(Sprint 2, Hiệp): loading the cleaned dataset for evaluation needs "
            "src/index.py against a persistent store; inline configs only for now"
        )

    queries = load_queries(args.queries)
    logger.info("loaded %d queries", len(queries))

    started = time.perf_counter()
    run = run_search(retriever, queries, args.top_k)
    runtime = time.perf_counter() - started
    logger.info("searched %d queries in %.2fs", len(run), runtime)

    if args.dump_pool is not None:
        dump_pool(run, args.dump_pool, retriever.name)
        logger.info("wrote pooling file %s", args.dump_pool)
        return 0

    labels = load_relevance(args.relevance)
    logger.info("loaded %d relevance labels", len(labels))

    metrics = RankingMetrics()
    # Everything above this line works today; RankingMetrics is still a stub.
    scores = metrics.summary(run, labels)  # TODO(Sprint 2, Hiếu)

    run_id = args.run_id or f"{retriever.name}-{int(time.time())}"
    append_run_row(
        args.runs_csv,
        {
            "run_id": run_id,
            "date": date.today().isoformat(),
            "run_by": args.run_by,
            "commit": git_commit(),
            "config": str(args.config),
            "runtime_s": f"{runtime:.2f}",
            "notes": args.notes,
            **{key: f"{value:.4f}" for key, value in scores.items()},
        },
    )
    for key, value in scores.items():
        logger.info("%s = %.4f", key, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
