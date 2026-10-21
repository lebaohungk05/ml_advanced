"""Catalog loading + image indexing of ``python -m src.index``.

Runs entirely on FakeEmbedder + InMemoryVectorStore: no GPU, no Qdrant, no
model download, so it stays in the default fast suite.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from src import registry
from src.adapters.embedders.fake_embedder import FakeEmbedder
from src.adapters.vectorstore.inmemory_adapter import InMemoryVectorStore
from src.core.dense_retriever import DenseRetriever
from src.core.models import Matrix, Product, Query, SearchResult, Vector
from src.index import (
    index_products_by_image,
    load_products,
    load_products_json,
    main,
    read_image_bytes,
)

CATALOG = [
    {
        "product_id": "fp-1",
        "title": "Áo khoác jean nam",
        "category": "Áo khoác (jacket)",
        "image_path": "train\\images\\000001.jpg",
        "attributes": {"chi_tiet": "Túi áo/quần"},
        "split": "test",  # unknown key: must be ignored, not crash
    },
    {
        "product_id": "fp-2",
        "title": "Váy đầm dự tiệc",
        "category": "Váy",
        "image_path": "train\\images\\000002.jpg",
        "attributes": {},
    },
]


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_products_json_ignores_unknown_keys(tmp_path: Path) -> None:
    products = load_products_json(_write_json(tmp_path / "test.json", CATALOG))

    assert [p.product_id for p in products] == ["fp-1", "fp-2"]
    assert products[0].title == "Áo khoác jean nam"
    assert products[0].attributes == {"chi_tiet": "Túi áo/quần"}


def test_load_products_json_rejects_a_non_array(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON array"):
        load_products_json(_write_json(tmp_path / "bad.json", {"products": []}))


def test_load_products_dispatches_on_suffix(tmp_path: Path) -> None:
    json_path = _write_json(tmp_path / "catalog.json", CATALOG)
    jsonl_path = tmp_path / "catalog.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in CATALOG), encoding="utf-8"
    )

    assert [p.product_id for p in load_products(json_path)] == ["fp-1", "fp-2"]
    assert [p.product_id for p in load_products(jsonl_path)] == ["fp-1", "fp-2"]


def test_load_products_rejects_an_unsupported_suffix(tmp_path: Path) -> None:
    parquet = tmp_path / "catalog.parquet"
    parquet.write_bytes(b"")

    with pytest.raises(ValueError, match=r"\.parquet"):
        load_products(parquet)


def test_read_image_bytes_normalizes_windows_separators(tmp_path: Path) -> None:
    image = tmp_path / "train" / "images" / "000001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg-bytes")
    product = Product(
        product_id="fp-1",
        title="Áo",
        category="Áo",
        image_path="train\\images\\000001.jpg",
    )

    assert read_image_bytes(tmp_path, product) == b"jpeg-bytes"


def test_read_image_bytes_names_the_missing_file(tmp_path: Path) -> None:
    product = Product(
        product_id="fp-9", title="Áo", category="Áo", image_path="train/images/nope.jpg"
    )

    with pytest.raises(FileNotFoundError, match=r"nope\.jpg"):
        read_image_bytes(tmp_path, product)


def test_index_products_by_image_embeds_images_not_captions(tmp_path: Path) -> None:
    products = []
    for index in range(5):
        relative = f"train/images/{index}.jpg"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image-{index}".encode())
        products.append(
            Product(
                product_id=f"fp-{index}",
                title=f"Sản phẩm {index}",
                category="Váy",
                image_path=relative,
            )
        )
    embedder = FakeEmbedder(dim=16)
    store = InMemoryVectorStore(dim=embedder.dim)
    retriever = DenseRetriever(embedder=embedder, store=store, name="fake-image")

    count = index_products_by_image(retriever, products, tmp_path, batch_size=2)

    assert count == 5
    assert store.count() == 5
    # Stored vectors are the image vectors: querying by the exact bytes of one
    # catalog image must return that product first, which text indexing cannot do.
    query_vector = embedder.encode_image([b"image-3"])[0]
    assert store.search(query_vector, top_k=1)[0].product_id == "fp-3"


def test_load_products_dispatch_ignores_suffix_case(tmp_path: Path) -> None:
    products = load_products(_write_json(tmp_path / "catalog.JSON", CATALOG))

    assert [p.product_id for p in products] == ["fp-1", "fp-2"]


def test_main_refuses_image_modality_on_a_non_dense_retriever(tmp_path: Path) -> None:
    pytest.importorskip("rank_bm25")
    config = tmp_path / "bm25.yaml"
    config.write_text("retriever:\n  type: bm25\n", encoding="utf-8")
    catalog = _write_json(tmp_path / "catalog.json", CATALOG)

    assert main(["--config", str(config), "--products", str(catalog), "--modality", "image"]) == 1


# --- --recreate is destructive: what it touches, and in what order ------------


class _SpyStore:
    """VectorStore double that records the calls ``_prepare_store`` makes on it.

    Duck-typed rather than a subclass on purpose: ``_prepare_store`` looks the
    collection-lifecycle methods up with ``getattr``, so the only thing that
    matters is that they exist. Writes land in the same ``calls`` list as the
    lifecycle hooks, which is what lets a test pin down their relative order.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.calls: list[str] = []

    def ensure_collection(self) -> None:
        self.calls.append("ensure_collection")

    def recreate_collection(self) -> None:
        self.calls.append("recreate_collection")

    def upsert(self, products: Sequence[Product], vectors: Matrix) -> None:
        self.calls.append(f"upsert:{len(products)}")

    def search(
        self, vector: Vector, top_k: int, filters: Mapping[str, str] | None = None
    ) -> list[SearchResult]:
        return []

    def count(self) -> int:
        return 0


class _SpyLexicalRetriever:
    """A Retriever that is deliberately NOT a DenseRetriever but owns a droppable store.

    BM25Retriever has no ``store`` at all, so it cannot show whether a refused
    run would have dropped a collection on its way out. This one can.
    """

    def __init__(self, store: _SpyStore, name: str = "spy-lexical") -> None:
        self.store = store
        self.name = name
        self.indexed: list[str] = []

    def index(self, products: Sequence[Product]) -> None:
        self.indexed.extend(product.product_id for product in products)

    def search(self, query: Query) -> list[SearchResult]:
        return []


def _write_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_main_does_not_drop_the_collection_when_it_refuses_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --modality image on a non-dense retriever is refused. --recreate is asked
    # for in the same breath, so the refusal has to happen BEFORE the collection
    # is dropped — otherwise the run wipes the catalog and then exits 1.
    store = _SpyStore()
    monkeypatch.setitem(
        registry._RETRIEVERS, "spy_lexical", lambda **params: _SpyLexicalRetriever(store)
    )
    config = _write_config(tmp_path / "spy.yaml", "retriever:\n  type: spy_lexical\n")
    catalog = _write_json(tmp_path / "catalog.json", CATALOG)

    exit_code = main(
        [
            "--config", str(config),
            "--products", str(catalog),
            "--modality", "image",
            "--recreate",
        ]
    )

    assert exit_code == 1
    assert store.calls == []  # nothing dropped, nothing created, nothing written


def test_main_drops_the_collection_before_writing_anything_when_recreate_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The positive control for the test above: on a run that is NOT refused,
    # --recreate really does drop first and only then upsert.
    store = _SpyStore(dim=16)
    monkeypatch.setitem(registry._VECTOR_STORES, "spy_store", lambda **params: store)
    config = _write_config(
        tmp_path / "dense.yaml",
        "retriever:\n"
        "  type: dense\n"
        "  name: fake-spy\n"
        "  embedder:\n"
        "    type: fake\n"
        "    params:\n"
        "      dim: 16\n"
        "  store:\n"
        "    type: spy_store\n",
    )
    catalog = _write_json(tmp_path / "catalog.json", CATALOG)

    exit_code = main(["--config", str(config), "--products", str(catalog), "--recreate"])

    assert exit_code == 0
    assert store.calls == ["recreate_collection", "upsert:2"]


def test_main_warns_that_recreate_was_ignored_by_a_store_that_cannot_drop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # InMemoryVectorStore has no recreate_collection: --recreate is a no-op there.
    # Silently succeeding would let an operator believe stale vectors were purged.
    config = _write_config(
        tmp_path / "inmemory.yaml",
        "retriever:\n"
        "  type: dense\n"
        "  name: fake-inmemory\n"
        "  embedder:\n"
        "    type: fake\n"
        "    params:\n"
        "      dim: 16\n"
        "  store:\n"
        "    type: inmemory\n",
    )
    catalog = _write_json(tmp_path / "catalog.json", CATALOG)

    with caplog.at_level(logging.WARNING, logger="index"):
        exit_code = main(
            ["--config", str(config), "--products", str(catalog), "--recreate"]
        )

    assert exit_code == 0  # a no-op, not a failure
    assert "--recreate ignored" in caplog.text
    assert "InMemoryVectorStore" in caplog.text
    assert "nothing was deleted" in caplog.text


def test_main_names_a_missing_store_in_the_recreate_ignored_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # BM25Retriever has no `store` attribute at all — the warning must still be
    # readable rather than crashing on type(None).__name__.
    pytest.importorskip("rank_bm25")
    config = _write_config(tmp_path / "bm25.yaml", "retriever:\n  type: bm25\n")
    catalog = _write_json(tmp_path / "catalog.json", CATALOG)

    with caplog.at_level(logging.WARNING, logger="index"):
        exit_code = main(
            ["--config", str(config), "--products", str(catalog), "--recreate"]
        )

    assert exit_code == 0
    assert "--recreate ignored: store <none> cannot drop a collection" in caplog.text


def _dense_spy_config(path: Path, store_type: str) -> Path:
    return _write_config(
        path,
        "retriever:\n"
        "  type: dense\n"
        "  name: fake-spy\n"
        "  embedder:\n"
        "    type: fake\n"
        "    params:\n"
        "      dim: 16\n"
        "  store:\n"
        f"    type: {store_type}\n",
    )


def test_main_indexes_by_image_without_ever_dropping_the_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Sprint 4 demo command, minus --recreate: the collection is *ensured*
    # (created if missing) and the catalog is upserted batch by batch. A run
    # without --recreate must never touch the destructive path.
    store = _SpyStore(dim=16)
    monkeypatch.setitem(registry._VECTOR_STORES, "spy_store", lambda **params: store)
    images_root = tmp_path / "images"
    for entry in CATALOG:
        image = images_root / str(entry["image_path"]).replace("\\", "/")
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"jpeg-{entry['product_id']}".encode())
    config = _dense_spy_config(tmp_path / "dense.yaml", "spy_store")
    catalog = _write_json(tmp_path / "catalog.json", CATALOG)

    exit_code = main(
        [
            "--config", str(config),
            "--products", str(catalog),
            "--modality", "image",
            "--images-root", str(images_root),
            "--batch-size", "1",
        ]
    )

    assert exit_code == 0
    assert store.calls == ["ensure_collection", "upsert:1", "upsert:1"]


def test_main_limit_caps_how_many_products_are_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _SpyStore(dim=16)
    monkeypatch.setitem(registry._VECTOR_STORES, "spy_store", lambda **params: store)
    config = _dense_spy_config(tmp_path / "dense.yaml", "spy_store")
    catalog = _write_json(tmp_path / "catalog.json", CATALOG)

    exit_code = main(
        ["--config", str(config), "--products", str(catalog), "--limit", "1"]
    )

    assert exit_code == 0
    assert store.calls == ["ensure_collection", "upsert:1"]  # 2 in the file, 1 indexed


def test_main_exits_nonzero_on_an_empty_catalog(tmp_path: Path) -> None:
    config = _dense_spy_config(tmp_path / "inmemory.yaml", "inmemory")
    catalog = _write_json(tmp_path / "empty.json", [])

    assert main(["--config", str(config), "--products", str(catalog)]) == 1


def test_index_products_by_image_rejects_a_zero_batch_size(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dim=16)
    retriever = DenseRetriever(
        embedder=embedder, store=InMemoryVectorStore(dim=embedder.dim), name="fake-image"
    )

    with pytest.raises(ValueError, match="batch-size"):
        index_products_by_image(retriever, [], tmp_path, batch_size=0)
