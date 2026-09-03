"""CLI: encode the whole catalog and push it into a vector store.

    python -m src.index --config configs/default.yaml
    python -m src.index --config configs/default.yaml --products data/processed/catalog.jsonl

    # the real demo index: 6.963 ảnh test split -> Qdrant, encoded by IMAGE
    python -m src.index --config configs/demo_siglip2_qdrant.yaml \
        --products data/processed/test.json --modality image --batch-size 16 --recreate

``--modality image`` bypasses ``DenseRetriever.index()`` on purpose: that method
(``src/core/dense_retriever.py``, frozen) embeds ``Product.to_text()`` only,
which for this dataset is just the detected category names. A cross-modal search
demo needs the catalog embedded from the actual images, exactly like
``scripts/run_sprint2_baselines.py`` does it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from src.core.dense_retriever import DenseRetriever
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
        help=".jsonl or .json catalog file; defaults to the inline 'products' block",
    )
    parser.add_argument(
        "--modality",
        choices=("text", "image"),
        default="text",
        help="embed each product from its caption or from its image file (default: %(default)s)",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("data/raw/fashionpedia"),
        help="root that Product.image_path is relative to (default: %(default)s)",
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


def load_products_json(path: Path) -> list[Product]:
    """Read a catalog stored as one big JSON array (``data/processed/*.json``).

    Unknown keys are ignored, same as the JSONL loader, so the prepared dataset
    can carry extra columns without breaking the indexer.
    """
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of product objects")
    products: list[Product] = []
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}[{position}]: expected a JSON object")
        products.append(Product(**{k: v for k, v in entry.items() if k in PRODUCT_FIELDS}))
    return products


def load_products(path: Path) -> list[Product]:
    """Dispatch on the file suffix: ``.jsonl`` line-delimited, ``.json`` array."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_products_jsonl(path)
    if suffix == ".json":
        return load_products_json(path)
    raise ValueError(f"unsupported catalog format {path.suffix!r}: expected .json or .jsonl")


def read_image_bytes(images_root: Path, product: Product) -> bytes:
    """Read the product's image, resolving its path under ``images_root``.

    ``image_path`` is stored with Windows separators in the prepared dataset, so
    it is normalised before joining to keep the same catalog file usable on Linux.
    """
    if product.image_path is None:
        raise FileNotFoundError(f"product {product.product_id} has no image_path")
    full = images_root / product.image_path.replace("\\", "/")
    if not full.is_file():
        raise FileNotFoundError(f"image for product {product.product_id} not found: {full}")
    return full.read_bytes()


def index_products_by_image(
    retriever: DenseRetriever,
    products: Sequence[Product],
    images_root: Path,
    batch_size: int,
) -> int:
    """Embed each product's IMAGE and upsert it, bypassing ``retriever.index()``.

    ``DenseRetriever.index()`` is frozen core code that encodes text only; this
    mirrors ``scripts/run_sprint2_baselines.py`` so the catalog side of the demo
    is genuinely image-based.
    """
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    total = len(products)
    for start in range(0, total, batch_size):
        chunk = products[start : start + batch_size]
        vectors = retriever.embedder.encode_image(
            [read_image_bytes(images_root, product) for product in chunk]
        )
        retriever.store.upsert(chunk, vectors)
        done = start + len(chunk)
        if done % 300 < batch_size or done == total:
            logger.info("encoded %d/%d product images", done, total)
    return total


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


def _prepare_store(retriever: Retriever, recreate: bool) -> None:
    """Make the target collection exist, dropping it first when ``--recreate``.

    Duck-typed: ``InMemoryVectorStore`` has neither method and needs neither, so
    both lookups miss and this is a no-op for the zero-infrastructure config.
    """
    store = getattr(retriever, "store", None)
    if recreate:
        drop_and_create = getattr(store, "recreate_collection", None)
        if drop_and_create is not None:
            logger.warning("dropping the existing collection before indexing")
            drop_and_create()
            return
        logger.warning(
            "--recreate ignored: store %s cannot drop a collection; nothing was deleted",
            type(store).__name__ if store is not None else "<none>",
        )
    ensure = getattr(store, "ensure_collection", None)
    if ensure is not None:
        ensure()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    retriever = build_retriever_from_config(config)

    if args.products is not None:
        products = load_products(args.products)
    else:
        products = products_from_config(config)
        logger.info("no --products given, using the inline demo catalog")

    if args.limit is not None:
        products = products[: args.limit]
    if not products:
        logger.error("nothing to index")
        return 1

    # Validated before _prepare_store: --recreate drops the collection, and a run
    # that is going to be refused must not destroy anything on its way out.
    dense: DenseRetriever | None = None
    if args.modality == "image":
        if not isinstance(retriever, DenseRetriever):
            logger.error(
                "--modality image needs a dense retriever, got %s",
                type(retriever).__name__,
            )
            return 1
        dense = retriever

    _prepare_store(retriever, args.recreate)

    started = time.perf_counter()
    if dense is not None:
        count = index_products_by_image(
            dense, products, args.images_root, args.batch_size
        )
    else:
        count = index_products(retriever, products, args.batch_size)
    logger.info(
        "indexed %d products into %s in %.2fs", count, retriever.name, time.perf_counter() - started
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
