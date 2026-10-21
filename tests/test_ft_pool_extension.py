"""Pool extension for the fine-tuned system: the invariants that protect the metrics.

The dangerous failure mode here is silent, not loud: if a (query, product) pair
in the fine-tuned system's top-10 ends up in NEITHER the auto-zero file nor the
human template, it keeps scoring as grade 0 forever and the headline comparison
table is quietly wrong. So the coverage/disjointness property is asserted
directly, on top of the narrower behaviours.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str) -> ModuleType:
    """Import a file under scripts/ (``scripts`` is not an installed package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ext = _load("build_ft_pool_extension_under_test", "scripts/build_ft_pool_extension.py")
catfilter = _load("category_filter_under_test", "scripts/category_filter.py")

# Mirrors data/category_mapping.json: the filter only fires on words longer than
# 2 characters, so "áo sơ mi" (all words <= 2 chars) would never match anything.
GARMENT_NAMES = ["Áo len", "Váy đầm", "Quần jean"]


def _hit(pid: str, rank: int) -> dict[str, object]:
    return {"product_id": pid, "score": 1.0 / rank, "rank": rank}


@pytest.fixture
def pool() -> dict[str, dict[str, list[dict]]]:
    return {
        "siglip2_lora": {
            # 3 knitwear candidates + 1 dress (category mismatch -> auto-zero)
            "1": [_hit("p1", 1), _hit("p2", 2), _hit("p3", 3), _hit("p4", 4)],
            # group D query: never auto-zeroed, no garment keyword
            "2": [_hit("p4", 1), _hit("p1", 2)],
        },
        "siglip2": {"1": [_hit("p9", 1)]},
    }


@pytest.fixture
def queries() -> dict[str, dict]:
    return {
        "1": {"id": "1", "text": "áo len dệt kim xám", "group": "A"},
        "2": {"id": "2", "text": "đi tiệc cưới buổi tối", "group": "D"},
    }


@pytest.fixture
def catalog() -> dict[str, dict]:
    return {
        "p1": {"product_id": "p1", "image_path": "a/1.jpg", "category": "Áo len / áo dệt kim"},
        "p2": {"product_id": "p2", "image_path": "a/2.jpg", "category": "Áo len / áo dệt kim"},
        "p3": {"product_id": "p3", "image_path": "a/3.jpg", "category": "Áo len / áo dệt kim"},
        "p4": {"product_id": "p4", "image_path": "a/4.jpg", "category": "Váy đầm / đầm"},
    }


def _pairs(rows: list[dict]) -> set[tuple[str, str]]:
    return {(str(r["query_id"]), str(r["product_id"])) for r in rows}


def test_already_labelled_top10_pair_is_excluded(pool, queries, catalog) -> None:
    labelled = {("1", "p2")}
    needs_human, auto_zero, counts = ext.build_extension(
        pool, queries, catalog, labelled, GARMENT_NAMES
    )

    assert ("1", "p2") not in _pairs(needs_human) | _pairs(auto_zero)
    assert counts.already_labelled == 1
    assert counts.total_pairs == 6


def test_outputs_are_disjoint_and_cover_every_unlabelled_pair(pool, queries, catalog) -> None:
    labelled = {("1", "p1")}
    needs_human, auto_zero, counts = ext.build_extension(
        pool, queries, catalog, labelled, GARMENT_NAMES
    )

    expected = {
        (qid, str(hit["product_id"]))
        for qid, hits in pool["siglip2_lora"].items()
        for hit in hits[:10]
    } - labelled

    human_pairs, auto_pairs = _pairs(needs_human), _pairs(auto_zero)
    assert not human_pairs & auto_pairs
    assert human_pairs | auto_pairs == expected
    assert counts.needs_human + counts.auto_zeroed == len(expected)
    assert counts.missing_from_catalog == 0


def test_category_mismatch_is_auto_zeroed_but_group_d_is_not(pool, queries, catalog) -> None:
    needs_human, auto_zero, _ = ext.build_extension(
        pool, queries, catalog, set(), GARMENT_NAMES
    )

    # query 1 names "áo len"; p4 is a dress -> auto-zero.
    assert _pairs(auto_zero) == {("1", "p4")}
    assert all(row["grade"] == 0 for row in auto_zero)
    assert all(row["graded_by"] == "auto-filter:category-mismatch" for row in auto_zero)
    # the same product under the group-D query still goes to a human.
    assert ("2", "p4") in _pairs(needs_human)


def test_human_rows_carry_no_system_identifying_field(pool, queries, catalog) -> None:
    needs_human, _, _ = ext.build_extension(pool, queries, catalog, set(), GARMENT_NAMES)

    assert needs_human
    for row in needs_human:
        assert set(row) == {
            "query_id",
            "query_text",
            "query_group",
            "product_id",
            "image_path",
            "category",
            "grade",
            "graded_by",
        }
        assert row["grade"] is None and row["graded_by"] is None
    blob = json.dumps(needs_human, ensure_ascii=False)
    assert "siglip2_lora" not in blob
    assert "rank" not in blob and "score" not in blob


def test_shuffle_is_deterministic_but_not_rank_order(pool, queries, catalog) -> None:
    first, _, _ = ext.build_extension(pool, queries, catalog, set(), GARMENT_NAMES)
    second, _, _ = ext.build_extension(pool, queries, catalog, set(), GARMENT_NAMES)

    assert [r["product_id"] for r in first] == [r["product_id"] for r in second]
    q1 = [r["product_id"] for r in first if r["query_id"] == "1"]
    assert sorted(q1) == ["p1", "p2", "p3"]
    assert q1 != ["p1", "p2", "p3"]


def test_only_the_fine_tuned_systems_hits_are_pooled(pool, queries, catalog) -> None:
    needs_human, auto_zero, _ = ext.build_extension(
        pool, queries, catalog, set(), GARMENT_NAMES
    )

    assert "p9" not in {r["product_id"] for r in needs_human + auto_zero}


def test_load_labelled_pairs_reads_auto_zero_globs_and_grades(tmp_path: Path) -> None:
    (tmp_path / "auto_zero_labels.json").write_text(
        json.dumps([{"query_id": 1, "product_id": "p1", "grade": 0, "graded_by": "auto"}]),
        encoding="utf-8",
    )
    (tmp_path / "auto_zero_labels_ft.json").write_text(
        json.dumps([{"query_id": "2", "product_id": "p2", "grade": 0, "graded_by": "auto"}]),
        encoding="utf-8",
    )
    (tmp_path / "grades_Hưng.jsonl").write_text(
        '{"query_id": "3", "product_id": "p3", "annotator": "Hưng", "grade": 2}\n\n',
        encoding="utf-8",
    )

    # query_id 1 arrives as an int in the file and must still match the pool's str key.
    assert ext.load_labelled_pairs(tmp_path) == {("1", "p1"), ("2", "p2"), ("3", "p3")}


def test_is_category_mismatch_matches_build_final_pool_rule() -> None:
    assert catfilter.is_category_mismatch(None, "Váy đầm / đầm") is False
    assert catfilter.is_category_mismatch("Áo len", "Váy đầm / đầm") is True
    assert catfilter.is_category_mismatch("Áo len", "Áo len / áo dệt kim") is False
