"""String -> factory registry, so configs pick implementations, not imports.

``configs/*.yaml`` names an embedder, a store and a retriever kind; nothing in
``app/`` or ``src/evaluate.py`` imports a concrete adapter. Adding an adapter is
therefore a one-line registration here plus the adapter file itself.

Heavy adapters are registered through a thunk that imports inside the factory,
so the registry stays importable with none of the ML extras installed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import yaml

from src.core.dense_retriever import DenseRetriever
from src.core.models import Product
from src.core.ports import Embedder, Retriever, VectorStore

_T = TypeVar("_T")

EmbedderFactory = Callable[..., Embedder]
VectorStoreFactory = Callable[..., VectorStore]
RetrieverFactory = Callable[..., Retriever]

_EMBEDDERS: dict[str, EmbedderFactory] = {}
_VECTOR_STORES: dict[str, VectorStoreFactory] = {}
_RETRIEVERS: dict[str, RetrieverFactory] = {}


class RegistryError(KeyError):
    """Raised when a config names something that is not registered."""


def register_embedder(name: str, factory: EmbedderFactory) -> None:
    _EMBEDDERS[name] = factory


def register_vector_store(name: str, factory: VectorStoreFactory) -> None:
    _VECTOR_STORES[name] = factory


def register_retriever(name: str, factory: RetrieverFactory) -> None:
    _RETRIEVERS[name] = factory


def available() -> dict[str, list[str]]:
    """Everything registered — handy for CLI ``--help`` output and error messages."""
    return {
        "embedders": sorted(_EMBEDDERS),
        "vector_stores": sorted(_VECTOR_STORES),
        "retrievers": sorted(_RETRIEVERS),
    }


def _lookup(
    table: Mapping[str, Callable[..., _T]], kind: str, name: str
) -> Callable[..., _T]:
    try:
        return table[name]
    except KeyError:
        raise RegistryError(
            f"unknown {kind} {name!r}; registered: {sorted(table)}"
        ) from None


# The registry key is positional-only: adapters legitimately take their own
# ``name`` keyword, which would otherwise collide with the lookup argument.
def build_embedder(kind: str, /, **params: Any) -> Embedder:
    return _lookup(_EMBEDDERS, "embedder", kind)(**params)


def build_vector_store(kind: str, /, **params: Any) -> VectorStore:
    return _lookup(_VECTOR_STORES, "vector store", kind)(**params)


def build_retriever(kind: str, /, **params: Any) -> Retriever:
    return _lookup(_RETRIEVERS, "retriever", kind)(**params)


# --- built-in registrations -------------------------------------------------


def _make_fake_embedder(**params: Any) -> Embedder:
    from src.adapters.embedders.fake_embedder import FakeEmbedder

    return FakeEmbedder(**params)


def _make_inmemory_store(**params: Any) -> VectorStore:
    from src.adapters.vectorstore.inmemory_adapter import InMemoryVectorStore

    return InMemoryVectorStore(**params)


def _make_clip_embedder(**params: Any) -> Embedder:
    from src.adapters.embedders.clip_adapter import ClipEmbedder

    return ClipEmbedder(**params)


def _make_siglip2_embedder(**params: Any) -> Embedder:
    from src.adapters.embedders.siglip2_adapter import Siglip2Embedder

    return Siglip2Embedder(**params)


def _make_visiglip_embedder(**params: Any) -> Embedder:
    from src.adapters.embedders.visiglip_adapter import ViSiglipEmbedder

    return ViSiglipEmbedder(**params)


def _make_resnet_embedder(**params: Any) -> Embedder:
    from src.adapters.embedders.resnet_adapter import ResnetImageEmbedder

    return ResnetImageEmbedder(**params)


def _make_qdrant_store(**params: Any) -> VectorStore:
    from src.adapters.vectorstore.qdrant_adapter import QdrantVectorStore

    return QdrantVectorStore(**params)


def _make_bm25_retriever(**params: Any) -> Retriever:
    from src.adapters.lexical.bm25_adapter import BM25Retriever

    return BM25Retriever(**params)


register_embedder("fake", _make_fake_embedder)
register_embedder("clip", _make_clip_embedder)
register_embedder("siglip2", _make_siglip2_embedder)
register_embedder("visiglip", _make_visiglip_embedder)
register_embedder("resnet", _make_resnet_embedder)

register_vector_store("inmemory", _make_inmemory_store)
register_vector_store("qdrant", _make_qdrant_store)

register_retriever("bm25", _make_bm25_retriever)


def _make_dense_retriever(
    name: str = "dense",
    embedder: Mapping[str, Any] | None = None,
    store: Mapping[str, Any] | None = None,
) -> Retriever:
    """Build a DenseRetriever from the nested ``embedder`` / ``store`` config blocks.

    The store inherits the embedder's dimensionality unless the config sets it,
    so a config cannot silently pair a 768-dim backbone with a 64-dim store.
    """
    embedder_cfg = dict(embedder or {})
    store_cfg = dict(store or {})
    built_embedder = build_embedder(
        str(embedder_cfg.pop("type", "fake")), **embedder_cfg.pop("params", {})
    )
    store_params = dict(store_cfg.pop("params", {}))
    store_params.setdefault("dim", built_embedder.dim)
    built_store = build_vector_store(str(store_cfg.pop("type", "inmemory")), **store_params)
    return DenseRetriever(embedder=built_embedder, store=built_store, name=name)


register_retriever("dense", _make_dense_retriever)


# --- config helpers ---------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a plain dict."""
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"config {path} must be a YAML mapping")
    return dict(loaded)


def build_retriever_from_config(config: Mapping[str, Any]) -> Retriever:
    """Instantiate the retriever described by the ``retriever`` block of a config."""
    retriever_cfg = dict(config.get("retriever") or {})
    if not retriever_cfg:
        raise ValueError("config has no 'retriever' block")
    kind = str(retriever_cfg.pop("type", "dense"))
    return build_retriever(kind, **retriever_cfg)


def products_from_config(config: Mapping[str, Any]) -> list[Product]:
    """Read the inline demo catalog (``products:``) out of a config."""
    raw_products = config.get("products") or []
    if not isinstance(raw_products, Sequence):
        raise ValueError("'products' must be a list")
    return [Product(**dict(entry)) for entry in raw_products]
