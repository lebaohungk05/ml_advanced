"""CLI: encode the whole catalog and push it into a vector store.

    python -m src.index --config configs/default.yaml
    python -m src.index --config configs/default.yaml --products data/processed/catalog.jsonl

With the default config this indexes the ~20 inline demo products into the
in-memory store, which is only useful as a smoke test (nothing persists after the
process exits). The real job — streaming the cleaned public dataset into Qdrant
in batches — needs ``qdrant_adapter.py`` and the processed dataset, both Sprint 2.

Status: CLI, config loading and JSONL catalog loading are real. Persisting to
Qdrant and reading the parquet dataset are not.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from src.core.models import Product
from src.core.ports import Retriever
from src.registry import build_retriever_from_config, load_config, products_from_config

logger = logging.getLogger("index")

PRODUCT_FIELDS = frozenset(
    {
        "product_id",
        "title",
        "category",
        "color",
        "material",
        "price_vnd",
        "image_path",
        "description",
        "attributes",
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.index",
        description="Encode the catalog and upsert it into the configured vector store.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="retriever config (default: %(default)s)",
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=None,
        help="JSONL catalog file; defaults to the inline 'products' block of the config",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="products per upsert batch"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="index at most N products (smoke tests)"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and recreate the collection before indexing",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def load_products_jsonl(path: Path) -> list[Product]:
    """Read a JSONL catalog, one product object per line.

    Unknown keys are ignored so that the cleaned dataset can carry extra columns
    (split, source id, ...) without breaking the indexer.
    """
    products: list[Product] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            products.append(Product(**{k: v for k, v in raw.items() if k in PRODUCT_FIELDS}))
    return products


def index_products(
    retriever: Retriever, products: Sequence[Product], batch_size: int
) -> int:
    """Index in batches so a large catalog does not have to fit in one call."""
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    for start in range(0, len(products), batch_size):
        batch = products[start : start + batch_size]
        retriever.index(batch)
        logger.info("indexed %d/%d products", start + len(batch), len(products))
    return len(products)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    retriever = build_retriever_from_config(config)

    if args.products is not None:
        if args.products.suffix != ".jsonl":
            raise NotImplementedError(
                "TODO(Sprint 2, Hiệp): only .jsonl is wired up; parquet loading of "
                "data/processed/ comes with the dataset cleaning script"
            )
        products = load_products_jsonl(args.products)
    else:
        products = products_from_config(config)
        logger.info("no --products given, using the inline demo catalog")

    if args.limit is not None:
        products = products[: args.limit]
    if not products:
        logger.error("nothing to index")
        return 1

    if args.recreate:
        raise NotImplementedError(
            "TODO(Sprint 2, Hiệp): --recreate needs QdrantVectorStore.ensure_collection; "
            "the in-memory store is empty at every start anyway"
        )

    started = time.perf_counter()
    count = index_products(retriever, products, args.batch_size)
    logger.info(
        "indexed %d products into %s in %.2fs", count, retriever.name, time.perf_counter() - started
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
