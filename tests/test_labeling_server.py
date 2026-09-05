"""Labeling tool: undo must not corrupt grades, and rows must arrive grouped by query.

Every test runs against a throwaway template + grades dir in ``tmp_path``. The
real ``data/eval/grades_*.jsonl`` files are finished ground truth for 4,011
pairs plus a live round, so a test that wrote there would be destroying data.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

TEMPLATE = [
    # Two queries, deliberately interleaved and NOT sorted by product_id, so a
    # test can tell "grouped by query" apart from "sorted".
    {"query_id": "q1", "query_text": "áo thun trắng", "product_id": "p9",
     "image_path": "train\\images\\9.jpg", "category": "áo", "grade": None},
    {"query_id": "q2", "query_text": "váy đen dài", "product_id": "p3",
     "image_path": "train\\images\\3.jpg", "category": "váy", "grade": None},
    {"query_id": "q1", "query_text": "áo thun trắng", "product_id": "p1",
     "image_path": "train\\images\\1.jpg", "category": "áo", "grade": None},
    {"query_id": "q2", "query_text": "váy đen dài", "product_id": "p7",
     "image_path": "train\\images\\7.jpg", "category": "váy", "grade": None},
    {"query_id": "q1", "query_text": "áo thun trắng", "product_id": "p5",
     "image_path": "train\\images\\5.jpg", "category": "áo", "grade": None},
]


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The module re-imported against a temp template + temp grades dir."""
    template = tmp_path / "template.json"
    template.write_text(json.dumps(TEMPLATE, ensure_ascii=False), encoding="utf-8")
    grades = tmp_path / "grades"
    grades.mkdir()

    monkeypatch.setenv("LABELING_TEMPLATE", str(template))
    import scripts.labeling_server as mod

    mod = importlib.reload(mod)
    monkeypatch.setattr(mod, "GRADES_DIR", grades)
    return mod


def _grade(client: TestClient, annotator: str, query_id: str, product_id: str, g: int) -> None:
    res = client.post(
        "/api/grade",
        json={"query_id": query_id, "product_id": product_id, "annotator": annotator, "grade": g},
    )
    assert res.status_code == 200


def test_next_stays_on_one_query_until_that_group_is_finished(server: Any) -> None:
    client = TestClient(server.app)

    seen: list[tuple[str, str]] = []
    for _ in range(5):
        params = {"annotator": "Hùng", "current_query": seen[-1][0] if seen else None}
        data = client.get("/api/next", params=params).json()
        assert not data["done"]
        seen.append((data["query_id"], data["product_id"]))
        _grade(client, "Hùng", data["query_id"], data["product_id"], 1)

    # All of one query's rows come before the other query's rows.
    assert [q for q, _ in seen] == ["q1", "q1", "q1", "q2", "q2"] or \
           [q for q, _ in seen] == ["q2", "q2", "q1", "q1", "q1"]


def test_within_a_group_the_template_shuffle_is_preserved_not_sorted(server: Any) -> None:
    client = TestClient(server.app)

    order: list[str] = []
    current = None
    for _ in range(3):
        data = client.get(
            "/api/next", params={"annotator": "Hùng", "current_query": current}
        ).json()
        if data["query_id"] != "q1":
            _grade(client, "Hùng", data["query_id"], data["product_id"], 0)
            continue
        current = data["query_id"]
        order.append(data["product_id"])
        _grade(client, "Hùng", data["query_id"], data["product_id"], 1)

    # Template order inside q1 is p9, p1, p5 — sorting would give p1, p5, p9.
    assert order == ["p9", "p1", "p5"]
    assert order != sorted(order)


def test_group_progress_counts_match_the_fixture(server: Any) -> None:
    client = TestClient(server.app)
    data = client.get("/api/next", params={"annotator": "Hùng"}).json()

    assert data["group_total_queries"] == 2
    assert data["group_total"] == (3 if data["query_id"] == "q1" else 2)
    assert data["group_position"] == 1


def test_a_query_with_half_done_rows_is_served_before_untouched_queries(server: Any) -> None:
    client = TestClient(server.app)
    # Hiệp grades one q2 row, so q2 now has a half-finished pair.
    _grade(client, "Hiệp", "q2", "p3", 2)

    data = client.get("/api/next", params={"annotator": "Hùng"}).json()
    assert data["query_id"] == "q2"
    assert data["product_id"] == "p3"


def test_rows_already_graded_by_the_caller_or_fully_graded_are_skipped(server: Any) -> None:
    client = TestClient(server.app)
    _grade(client, "Hùng", "q1", "p9", 1)  # caller's own -> skip
    _grade(client, "Hiệp", "q1", "p1", 1)
    _grade(client, "Hiếu", "q1", "p1", 1)  # two graders -> skip

    served = []
    for _ in range(3):
        data = client.get("/api/next", params={"annotator": "Hùng"}).json()
        if data["done"]:
            break
        served.append((data["query_id"], data["product_id"]))
        _grade(client, "Hùng", data["query_id"], data["product_id"], 0)

    assert ("q1", "p9") not in served
    assert ("q1", "p1") not in served


def test_undo_removes_only_the_callers_last_line_and_keeps_the_rest_byte_identical(
    server: Any,
) -> None:
    client = TestClient(server.app)
    path = server.GRADES_DIR / "grades_Hùng.jsonl"
    # Interleave two annotators in ONE file to prove the filter is on the
    # annotator field, not merely on the filename.
    lines = [
        {"query_id": "q1", "product_id": "p9", "annotator": "Hùng", "grade": 1},
        {"query_id": "q1", "product_id": "p1", "annotator": "Hiệp", "grade": 2},
        {"query_id": "q1", "product_id": "p5", "annotator": "Hùng", "grade": 0},
        {"query_id": "q2", "product_id": "p3", "annotator": "Hiệp", "grade": 1},
    ]
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8"
    )

    res = client.post("/api/undo", json={"annotator": "Hùng"})
    body = res.json()

    assert body["undone"] is True
    assert body["previous_grade"] == 0
    assert body["row"]["product_id"] == "p5"
    # Exactly the p5/Hùng line is gone; the other three survive in order.
    expected = "".join(
        json.dumps(x, ensure_ascii=False) + "\n" for x in (lines[0], lines[1], lines[3])
    )
    assert path.read_text(encoding="utf-8") == expected


def test_undo_never_deletes_another_annotators_grade_for_the_same_pair(server: Any) -> None:
    client = TestClient(server.app)
    _grade(client, "Hiệp", "q1", "p9", 2)
    _grade(client, "Hùng", "q1", "p9", 0)

    client.post("/api/undo", json={"annotator": "Hùng"})

    hiep = (server.GRADES_DIR / "grades_Hiệp.jsonl").read_text(encoding="utf-8")
    assert json.loads(hiep.strip())["grade"] == 2
    assert (server.GRADES_DIR / "grades_Hùng.jsonl").read_text(encoding="utf-8") == ""


def test_undo_with_nothing_to_undo_creates_and_truncates_nothing(server: Any) -> None:
    client = TestClient(server.app)

    body = client.post("/api/undo", json={"annotator": "Hùng"}).json()

    assert body["undone"] is False
    assert "hoàn tác" in body["reason"]
    assert list(server.GRADES_DIR.iterdir()) == []


def test_after_undo_the_row_is_served_again(server: Any) -> None:
    client = TestClient(server.app)
    first = client.get("/api/next", params={"annotator": "Hùng"}).json()
    _grade(client, "Hùng", first["query_id"], first["product_id"], 1)

    after_grade = client.get(
        "/api/next", params={"annotator": "Hùng", "current_query": first["query_id"]}
    ).json()
    assert after_grade["product_id"] != first["product_id"]

    client.post("/api/undo", json={"annotator": "Hùng"})
    again = client.get(
        "/api/next", params={"annotator": "Hùng", "current_query": first["query_id"]}
    ).json()
    assert (again["query_id"], again["product_id"]) == (first["query_id"], first["product_id"])


def test_two_consecutive_undos_walk_back_two_items_in_order(server: Any) -> None:
    client = TestClient(server.app)
    _grade(client, "Hùng", "q1", "p9", 1)
    _grade(client, "Hùng", "q1", "p1", 2)

    first = client.post("/api/undo", json={"annotator": "Hùng"}).json()
    second = client.post("/api/undo", json={"annotator": "Hùng"}).json()

    assert first["row"]["product_id"] == "p1"
    assert first["previous_grade"] == 2
    assert second["row"]["product_id"] == "p9"
    assert second["previous_grade"] == 1
    assert (server.GRADES_DIR / "grades_Hùng.jsonl").read_text(encoding="utf-8") == ""


def test_undo_leaves_no_temp_file_behind(server: Any) -> None:
    client = TestClient(server.app)
    _grade(client, "Hùng", "q1", "p9", 1)

    client.post("/api/undo", json={"annotator": "Hùng"})

    assert [p.name for p in server.GRADES_DIR.iterdir()] == ["grades_Hùng.jsonl"]


def test_undo_requires_a_name(server: Any) -> None:
    client = TestClient(server.app)
    assert client.post("/api/undo", json={"annotator": "   "}).status_code == 400
