"""Registry wiring, plus a smoke test per stub module.

A stub has nothing to verify beyond "importing and constructing it works, and
calling it raises NotImplementedError rather than ImportError" — that distinction
is what keeps ``pytest -q`` green on a laptop with no ML extras installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.embedders.clip_adapter import ClipEmbedder
from src.adapters.embedders.resnet_adapter import ResnetImageEmbedder
from src.adapters.embedders.siglip2_adapter import Siglip2Embedder
from src.adapters.embedders.visiglip_adapter import ViSiglipEmbedder
from src.adapters.lexical.bm25_adapter import BM25Retriever
from src.adapters.metrics.ranking_metrics import RankingMetrics
from src.adapters.training.lora_dora_trainer import TrainConfig, lora_target_modules
from src.adapters.vectorstore.qdrant_adapter import QdrantVectorStore
from src.core.dense_retriever import DenseRetriever
from src.core.models import Product, Query
from src.registry import (
    RegistryError,
    available,
    build_embedder,
    build_retriever_from_config,
    load_config,
    products_from_config,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def test_registry_lists_all_planned_adapters() -> None:
    registered = available()
    assert set(registered["embedders"]) == {"clip", "fake", "resnet", "siglip2", "visiglip"}
    assert set(registered["vector_stores"]) == {"inmemory", "qdrant"}
    assert set(registered["retrievers"]) == {"bm25", "dense"}


def test_unknown_name_reports_what_is_registered() -> None:
    with pytest.raises(RegistryError, match="unknown embedder 'nope'"):
        build_embedder("nope")


def test_default_config_builds_an_indexable_retriever() -> None:
    config = load_config(DEFAULT_CONFIG)
    retriever = build_retriever_from_config(config)
    products = products_from_config(config)

    assert isinstance(retriever, DenseRetriever)
    assert len(products) >= 20
    assert len({p.product_id for p in products}) == len(products)

    retriever.index(products)
    hits = retriever.search(Query(query_id="q1", text="áo hoodie nỉ bông oversize", top_k=3))
    assert hits[0].product_id == "P012"


def test_store_dim_follows_embedder_dim() -> None:
    retriever = build_retriever_from_config(
        {"retriever": {"type": "dense", "embedder": {"type": "fake", "params": {"dim": 24}}}}
    )
    assert isinstance(retriever, DenseRetriever)
    assert retriever.store.dim == 24


def test_multimodal_query_is_rejected_until_fusion_is_designed() -> None:
    retriever = build_retriever_from_config({"retriever": {"type": "dense"}})
    retriever.index([Product(product_id="P1", title="Áo thun", category="áo")])

    with pytest.raises(NotImplementedError, match="Sprint 4"):
        retriever.search(Query(query_id="q1", text="áo", image_bytes=b"x"))


@pytest.mark.parametrize(
    "embedder",
    [ClipEmbedder(), Siglip2Embedder(), ResnetImageEmbedder(), ViSiglipEmbedder()],
    ids=lambda e: type(e).__name__,
)
def test_real_embedders_defer_model_loading_until_first_use(embedder: object) -> None:
    # Construction must stay free of torch/downloads, otherwise plain `pytest -q`
    # would pull hundreds of MB of weights on a laptop with no ML extras.
    assert isinstance(embedder.name, str)  # type: ignore[attr-defined]
    assert embedder._model is None  # type: ignore[attr-defined]


def test_resnet_documents_that_it_has_no_text_tower() -> None:
    with pytest.raises(NotImplementedError, match="no text tower"):
        ResnetImageEmbedder().encode_text(["áo thun"])


def test_qdrant_reads_url_from_env_and_defaults_to_localhost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    assert QdrantVectorStore(dim=8).url == "http://localhost:6333"

    monkeypatch.setenv("QDRANT_URL", "http://qdrant.internal:6333")
    assert QdrantVectorStore(dim=8).url == "http://qdrant.internal:6333"


def test_bm25_is_implemented_but_stays_text_only() -> None:
    retriever = BM25Retriever()
    retriever.index([])

    with pytest.raises(NotImplementedError, match="lexical text index"):
        retriever.search(Query(query_id="q1", image_path="anh.jpg"))


def test_metrics_is_implemented_and_handles_an_empty_run() -> None:
    assert RankingMetrics().recall_at_k({}, [], k=5) == 0.0


def test_train_config_defaults_match_the_agreed_hyperparameters() -> None:
    config = TrainConfig()

    assert config.learning_rate == 1e-4
    assert config.warmup_steps == 500
    assert config.weight_decay == 0.01
    assert config.epochs == 10
    assert config.physical_batch_size == 32
    assert config.effective_batch_size == 256
    assert config.grad_accumulation_steps == 8
    assert config.seed == 42
    assert config.bf16 is True
    assert config.lora_rank == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.1
    assert config.use_dora is True
    assert config.target_modules == ("q_proj", "k_proj", "v_proj", "out_proj")
    assert config.freeze_backbone is True


def test_train_config_rejects_batch_sizes_that_do_not_divide() -> None:
    with pytest.raises(ValueError, match="must be a multiple"):
        TrainConfig(physical_batch_size=32, effective_batch_size=100)


class _FakeBackbone:
    """SigLIP's module tree, trimmed to what target-module selection looks at.

    Lets the tower filtering be tested without torch or a weight download; the
    real names come from ``AutoModel.from_pretrained(...).named_modules()``.
    """

    NAMES = (
        "text_model",
        "text_model.encoder.layers.0.self_attn.q_proj",
        "text_model.encoder.layers.0.self_attn.out_proj",
        "text_model.encoder.layers.0.mlp.fc1",
        "text_model.head",
        "vision_model",
        "vision_model.encoder.layers.0.self_attn.q_proj",
        "vision_model.encoder.layers.0.self_attn.v_proj",
        "vision_model.encoder.layers.0.mlp.fc1",
        # The pooling head owns an out_proj too, but it belongs to
        # nn.MultiheadAttention and must NOT be adapted.
        "vision_model.head.attention.out_proj",
    )

    def named_modules(self) -> list[tuple[str, None]]:
        return [(name, None) for name in self.NAMES]


def test_lora_adapts_only_the_towers_the_config_asks_for() -> None:
    model = _FakeBackbone()

    both = lora_target_modules(model, TrainConfig())
    assert both == [
        "text_model.encoder.layers.0.self_attn.q_proj",
        "text_model.encoder.layers.0.self_attn.out_proj",
        "vision_model.encoder.layers.0.self_attn.q_proj",
        "vision_model.encoder.layers.0.self_attn.v_proj",
    ]

    image_only = lora_target_modules(model, TrainConfig(adapt_text_tower=False))
    assert all(name.startswith("vision_model.encoder") for name in image_only)

    text_only = lora_target_modules(model, TrainConfig(adapt_image_tower=False))
    assert all(name.startswith("text_model.encoder") for name in text_only)


def test_lora_refuses_a_backbone_whose_module_names_it_does_not_recognize() -> None:
    class _Renamed:
        def named_modules(self) -> list[tuple[str, None]]:
            return [("encoder.blocks.0.attn.qkv", None)]

    with pytest.raises(ValueError, match="module names changed"):
        lora_target_modules(_Renamed(), TrainConfig())
