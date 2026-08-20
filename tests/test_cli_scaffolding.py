"""The parts of the CLI scripts that are real today: arg parsing and file IO."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluate import RUNS_CSV_COLUMNS, append_run_row, load_queries, load_relevance
from src.evaluate import parse_args as parse_evaluate_args
from src.index import load_products_jsonl
from src.index import parse_args as parse_index_args


def test_clis_default_to_the_demo_config() -> None:
    assert parse_index_args([]).config == Path("configs/default.yaml")
    assert parse_index_args([]).products is None

    evaluate_args = parse_evaluate_args([])
    assert evaluate_args.config == Path("configs/default.yaml")
    assert evaluate_args.top_k == 20
    assert evaluate_args.runs_csv == Path("experiments/runs.csv")


def test_load_products_jsonl_ignores_unknown_columns(tmp_path: Path) -> None:
    path = tmp_path / "catalog.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "product_id": "P1",
                        "title": "Áo khoác bomber",
                        "category": "áo khoác",
                        "split": "train",
                    },
                    ensure_ascii=False,
                ),
                "",
                json.dumps(
                    {"product_id": "P2", "title": "Váy hai dây", "category": "váy"},
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    products = load_products_jsonl(path)

    assert [p.product_id for p in products] == ["P1", "P2"]
    assert products[0].title == "Áo khoác bomber"


def test_load_products_jsonl_reports_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "catalog.jsonl"
    path.write_text('{"product_id": "P1"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"catalog\.jsonl:1"):
        load_products_jsonl(path)


def test_load_queries_accepts_both_layouts(tmp_path: Path) -> None:
    flat = tmp_path / "flat.json"
    flat.write_text(
        json.dumps([{"query_id": "q1", "text": "váy hai dây", "tier": "multi-attribute"}]),
        encoding="utf-8",
    )
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(
        json.dumps({"queries": [{"query_id": "q1", "text": "váy hai dây"}]}), encoding="utf-8"
    )

    assert load_queries(flat)[0].query_id == "q1"
    assert load_queries(wrapped)[0].text == "váy hai dây"


def test_load_relevance_builds_labels(tmp_path: Path) -> None:
    path = tmp_path / "relevance.json"
    path.write_text(
        json.dumps(
            [
                {"query_id": "q1", "product_id": "P1", "grade": 2, "annotator": "A"},
                {"query_id": "q1", "product_id": "P2", "grade": 0},
            ]
        ),
        encoding="utf-8",
    )

    labels = load_relevance(path)

    assert [label.grade for label in labels] == [2, 0]
    assert labels[0].annotator == "A"
    assert labels[1].is_relevant is False


def test_append_run_row_writes_header_once(tmp_path: Path) -> None:
    path = tmp_path / "runs.csv"
    append_run_row(path, {"run_id": "r1", "recall@5": "0.5000"})
    append_run_row(path, {"run_id": "r2"})

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == ",".join(RUNS_CSV_COLUMNS)
    assert len(lines) == 3
    assert lines[1].startswith("r1,")
