"""Every adapter must satisfy the port it claims — stubs included.

This is the test that keeps four people's adapters interchangeable: if someone
renames a method, this fails here rather than in the middle of an eval run.
"""

from __future__ import annotations

import pytest

from src.adapters.embedders.clip_adapter import ClipEmbedder
from src.adapters.embedders.fake_embedder import FakeEmbedder
from src.adapters.embedders.resnet_adapter import ResnetImageEmbedder
from src.adapters.embedders.siglip2_adapter import Siglip2Embedder
from src.adapters.embedders.visiglip_adapter import ViSiglipEmbedder
from src.adapters.lexical.bm25_adapter import BM25Retriever
from src.adapters.metrics.ranking_metrics import RankingMetrics
from src.adapters.vectorstore.inmemory_adapter import InMemoryVectorStore
from src.adapters.vectorstore.qdrant_adapter import QdrantVectorStore
from src.core.dense_retriever import DenseRetriever
from src.core.ports import Embedder, MetricsPort, Retriever, VectorStore

EMBEDDERS = [
    FakeEmbedder(dim=8),
    ClipEmbedder(),
    Siglip2Embedder(),
    ViSiglipEmbedder(),
    ResnetImageEmbedder(),
]


@pytest.mark.parametrize("embedder", EMBEDDERS, ids=lambda e: type(e).__name__)
def test_embedders_satisfy_embedder_port(embedder: object) -> None:
    assert isinstance(embedder, Embedder)


@pytest.mark.parametrize(
    "store",
    [InMemoryVectorStore(dim=8), QdrantVectorStore(dim=8)],
    ids=lambda s: type(s).__name__,
)
def test_stores_satisfy_vector_store_port(store: object) -> None:
    assert isinstance(store, VectorStore)


def test_retrievers_satisfy_retriever_port() -> None:
    dense = DenseRetriever(
        embedder=FakeEmbedder(dim=8), store=InMemoryVectorStore(dim=8), name="dense"
    )
    assert isinstance(dense, Retriever)
    assert isinstance(BM25Retriever(), Retriever)


def test_ranking_metrics_satisfies_metrics_port() -> None:
    assert isinstance(RankingMetrics(), MetricsPort)


def test_dense_retriever_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match store dim"):
        DenseRetriever(
            embedder=FakeEmbedder(dim=8), store=InMemoryVectorStore(dim=16), name="mismatch"
        )
