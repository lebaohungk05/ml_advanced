"""Fast blind-labeling tool for data/eval/relevance_template.json.

Run: python scripts/labeling_server.py
Then open the printed URL in a browser — on the same machine, or on any
device on the same Wi-Fi/LAN (the server binds 0.0.0.0), since it serves the
catalog images straight from this machine's data/raw/fashionpedia/.

Each (query, product) row needs 2 DIFFERENT annotators (DeCuong Mục 3.4 điểm
4). Every submitted grade is appended to data/eval/grades_<annotator>.jsonl
(one file per person, append-only — safe if the server restarts or two people
grade at once).

/api/next serves rows GROUPED BY QUERY: it finishes every row this annotator
still owes for one query before moving to the next one, so the query text has
to be read once per group instead of once per image. Group choice, in order:
  1. the query the annotator is already working through
  2. otherwise the query with the most rows that already have exactly 1 grade
     from someone else (finishing pairs is what closes the pool)
  3. otherwise the first query with untouched rows
Never serves a row this same annotator already graded, and never serves a
row that already has 2 grades from two different people.

Grouping does NOT weaken the blind pooling of Mục 3.4. The shuffle baked into
the template exists so an annotator cannot tell which retrieval system
produced a hit from where it appears; the grouping below is *stable* over that
shuffled order, so within one query the candidates stay in random order. Do
not "tidy" this into a sort by product_id/rank — that would leak provenance.

/api/undo removes the caller's most recent grade and hands the row back, for
the very common case of hitting the wrong key. It rewrites that one annotator's
file atomically (temp file + os.replace) and touches nobody else's grades.

Product category is deliberately NOT shown in the UI — the point is a blind
judgement of the image against the query text alone.
"""

from __future__ import annotations

import io
import json
import os
import socket
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TEMPLATE = ROOT / "data" / "eval" / "relevance_template.json"
TEMPLATE_PATH = Path(os.environ.get("LABELING_TEMPLATE", _DEFAULT_TEMPLATE))
IMAGES_ROOT = ROOT / "data" / "raw" / "fashionpedia"
GRADES_DIR = ROOT / "data" / "eval"

app = FastAPI(title="Fashion Search — Blind Labeling Tool")

with open(TEMPLATE_PATH, encoding="utf-8") as f:
    ROWS: list[dict[str, Any]] = json.load(f)


def _grade_files() -> list[Path]:
    return sorted(GRADES_DIR.glob("grades_*.jsonl"))


def _load_existing_grades() -> dict[tuple[str, str], list[str]]:
    """(query_id, product_id) -> list of annotator names who already graded it."""
    graders: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in _grade_files():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["query_id"], rec["product_id"])
                graders[key].append(rec["annotator"])
    return graders


def _row_payload(row: dict[str, Any], pending: int, total: int, group_index: int) -> dict[str, Any]:
    """One item as the UI wants it, plus where it sits in its query's group."""
    return {
        "done": False,
        "query_id": row["query_id"],
        "query_text": row["query_text"],
        "product_id": row["product_id"],
        "image_url": f"/image/{row['image_path'].replace(chr(92), '/')}",
        # group_* let the UI show "câu 12/155 · ảnh 3/8" and flag a new query.
        "group_index": group_index,
        "group_total_queries": len({r["query_id"] for r in ROWS}),
        "group_position": total - pending + 1,
        "group_total": total,
    }


def _pending_by_query(annotator: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Rows this annotator still owes, grouped by query, plus each group's half-done count.

    Insertion order of both the groups and the rows inside them follows ROWS,
    i.e. the template's shuffled order — see the module docstring on why that
    must not be re-sorted.
    """
    graders = _load_existing_grades()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    half_done: dict[str, int] = defaultdict(int)
    for row in ROWS:
        existing = graders.get((row["query_id"], row["product_id"]), [])
        if annotator in existing or len(existing) >= 2:
            continue
        groups[row["query_id"]].append(row)
        if len(existing) == 1:
            half_done[row["query_id"]] += 1
    return groups, half_done


class GradeIn(BaseModel):
    query_id: str
    product_id: str
    annotator: str
    grade: int


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()) or "anon"


@app.get("/api/progress")
def progress() -> dict[str, Any]:
    graders = _load_existing_grades()
    total = len(ROWS)
    counts = [len(graders.get((row["query_id"], row["product_id"]), [])) for row in ROWS]
    done = sum(1 for c in counts if c >= 2)
    partial = sum(1 for c in counts if c == 1)
    per_annotator: dict[str, int] = defaultdict(int)
    for path in _grade_files():
        name = path.stem.removeprefix("grades_")
        with open(path, encoding="utf-8") as f:
            per_annotator[name] = sum(1 for line in f if line.strip())
    return {
        "total_rows": total,
        "fully_labeled": done,
        "half_labeled": partial,
        "not_started": total - done - partial,
        "per_annotator": per_annotator,
    }


@app.get("/api/next")
def next_row(annotator: str, current_query: str | None = None) -> dict[str, Any]:
    """Next row for this annotator, grouped by query (see module docstring).

    ``current_query`` is the query the browser is already working through; it is
    honoured while that group still has rows so the annotator is not bounced
    between queries mid-group. It is a hint only — a group that is finished (or
    was never theirs) falls through to normal group selection.
    """
    if not annotator.strip():
        raise HTTPException(400, "annotator name required")

    groups, half_done = _pending_by_query(annotator)
    if not groups:
        return {"done": True}

    query_order = list(groups)
    if current_query in groups:
        chosen = current_query
    else:
        # Most half-done rows first (closing pairs closes the pool); ties and
        # the all-fresh case fall back to template order via the index key.
        chosen = max(query_order, key=lambda q: (half_done.get(q, 0), -query_order.index(q)))
    assert chosen is not None  # narrowed by the branches above

    pending = groups[chosen]
    # Group size = rows this annotator owes now + the ones they already did,
    # so "ảnh 3/8" counts the whole query, not just what is left.
    total_for_query = sum(1 for row in ROWS if row["query_id"] == chosen)
    already_done = total_for_query - len(pending)
    return _row_payload(
        pending[0],
        pending=len(pending),
        total=len(pending) + already_done,
        group_index=sorted({r["query_id"] for r in ROWS}).index(chosen) + 1,
    )


@app.post("/api/grade")
def submit_grade(payload: GradeIn) -> dict[str, str]:
    if payload.grade not in (0, 1, 2):
        raise HTTPException(400, "grade must be 0, 1 or 2")
    GRADES_DIR.mkdir(parents=True, exist_ok=True)
    path = GRADES_DIR / f"grades_{_safe_filename(payload.annotator)}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload.model_dump(), ensure_ascii=False) + "\n")
    return {"status": "ok"}


class UndoIn(BaseModel):
    annotator: str


@app.post("/api/undo")
def undo_last(payload: UndoIn) -> dict[str, Any]:
    """Drop this annotator's most recent grade and hand the row back for re-grading.

    Only ever rewrites ``grades_<this annotator>.jsonl``, only ever drops ONE
    line, and only a line whose ``annotator`` field is the caller — a shared
    file is never touched and another person's judgement can never be deleted
    from here. The rewrite goes through a temp file in the same directory plus
    ``os.replace`` (atomic on Windows and POSIX), so an interrupted undo cannot
    leave a truncated grade file behind; the surviving lines keep their exact
    bytes and order.
    """
    annotator = payload.annotator.strip()
    if not annotator:
        raise HTTPException(400, "annotator name required")

    path = GRADES_DIR / f"grades_{_safe_filename(annotator)}.jsonl"
    if not path.exists():
        return {"undone": False, "reason": "chưa chấm dòng nào để hoàn tác"}

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    target = None
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if json.loads(stripped).get("annotator") == annotator:
            target = i
            break
    if target is None:
        return {"undone": False, "reason": "chưa chấm dòng nào để hoàn tác"}

    record = json.loads(lines[target].strip())
    remaining = lines[:target] + lines[target + 1 :]

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.writelines(remaining)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)  # never leave a half-written temp behind
        raise

    row = next(
        (
            r
            for r in ROWS
            if r["query_id"] == record["query_id"] and r["product_id"] == record["product_id"]
        ),
        None,
    )
    if row is None:  # graded against a different template than the one loaded now
        return {"undone": True, "row": None, "previous_grade": record.get("grade")}

    groups, _ = _pending_by_query(annotator)
    pending = groups.get(row["query_id"], [])
    total_for_query = sum(1 for r in ROWS if r["query_id"] == row["query_id"])
    payload_row = _row_payload(
        row,
        pending=len(pending),
        total=total_for_query,
        group_index=sorted({r["query_id"] for r in ROWS}).index(row["query_id"]) + 1,
    )
    return {"undone": True, "row": payload_row, "previous_grade": record.get("grade")}


@app.get("/image/{path:path}")
def get_image(path: str) -> FileResponse:
    full = (IMAGES_ROOT / path).resolve()
    if not str(full).startswith(str(IMAGES_ROOT.resolve())) or not full.exists():
        raise HTTPException(404, "image not found")
    return FileResponse(full)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


_PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Chấm nhãn — Fashion Search</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 24px auto;
         padding: 0 16px; background:#111; color:#eee; }
  #name-box { display:flex; gap:8px; margin-bottom:16px; }
  #name-box input { flex:1; padding:8px; font-size:16px; }
  #name-box button { padding:8px 16px; font-size:16px; }
  #query { font-size:22px; margin: 12px 0; text-align:center; }
  #img-wrap { text-align:center; margin: 12px 0; }
  #img-wrap img { max-width:100%; max-height:60vh; border-radius:8px; }
  #buttons { display:flex; gap:12px; justify-content:center; margin: 16px 0; }
  #buttons button { flex:1; padding:20px; font-size:20px; border-radius:8px;
                    border:none; cursor:pointer; }
  .b0 { background:#7a2020; color:#fff; }
  .b1 { background:#7a6a20; color:#fff; }
  .b2 { background:#207a2e; color:#fff; }
  #progress { text-align:center; color:#999; font-size:14px; margin-top:16px; }
  #done { text-align:center; font-size:24px; margin-top:60px; }
  .hint { text-align:center; color:#777; font-size:13px; }
  #group { text-align:center; color:#8ab; font-size:14px; margin-top:4px; }
  #undo-row { text-align:center; margin-top:10px; }
  #undo-btn { padding:8px 18px; font-size:15px; border-radius:6px; border:1px solid #555;
              background:#222; color:#ddd; cursor:pointer; }
  #undo-btn:disabled { opacity:.35; cursor:default; }
  #undo-note { color:#c9a227; font-size:14px; min-height:20px; text-align:center; }
  /* A new query must be impossible to miss: grouping means the annotator stops
     re-reading the text, and silently changing it is how mislabels happen. */
  #query.new-query { animation: flash 1.1s ease-out; border-radius:6px; }
  @keyframes flash {
    0%   { background:#1d4e6b; box-shadow:0 0 0 6px #1d4e6b; }
    100% { background:transparent; box-shadow:none; }
  }
  #new-query-tag { display:none; text-align:center; color:#6cf; font-size:13px;
                   letter-spacing:.5px; }
  #new-query-tag.show { display:block; }
</style>
</head>
<body>
  <div id="name-box">
    <input id="name-input" placeholder="Tên của bạn (vd: Hùng)">
    <button onclick="startSession()">Bắt đầu</button>
  </div>
  <div id="work" style="display:none">
    <div id="new-query-tag">CÂU HỎI MỚI — đọc lại đề</div>
    <div id="query"></div>
    <div id="group"></div>
    <div id="img-wrap"><img id="img"></div>
    <div id="buttons">
      <button class="b0" onclick="grade(0)">0 — Không liên quan</button>
      <button class="b1" onclick="grade(1)">1 — Một phần</button>
      <button class="b2" onclick="grade(2)">2 — Đúng ý</button>
    </div>
    <div class="hint">Phím tắt: 0 / 1 / 2 · quay lại sửa: Backspace</div>
    <div id="undo-row">
      <button id="undo-btn" onclick="undoLast()">← Quay lại (sửa)</button>
    </div>
    <div id="undo-note"></div>
    <div id="progress"></div>
  </div>
  <div id="done" style="display:none">🎉 Hết việc — không còn dòng nào cần bạn chấm nữa!</div>

<script>
let annotator = localStorage.getItem("annotator") || "";
let current = null;
// Guards double-submits: holding a grade key or spamming Backspace would
// otherwise fire overlapping requests and skip/undo more rows than intended.
let busy = false;

function startSession() {
  const v = document.getElementById("name-input").value.trim();
  if (!v) return;
  annotator = v;
  localStorage.setItem("annotator", v);
  document.getElementById("work").style.display = "block";
  loadNext();
}

if (annotator) {
  document.getElementById("name-input").value = annotator;
  startSession();
}

async function loadNext() {
  let url = `/api/next?annotator=${encodeURIComponent(annotator)}`;
  if (current) url += `&current_query=${encodeURIComponent(current.query_id)}`;
  const res = await fetch(url);
  const data = await res.json();
  if (data.done) {
    document.getElementById("work").style.display = "none";
    document.getElementById("done").style.display = "block";
    return;
  }
  showRow(data, null);
}

// Render one item. previousGrade != null means we just walked back to it.
function showRow(data, previousGrade) {
  const changedQuery = !current || current.query_id !== data.query_id;
  current = data;

  const queryEl = document.getElementById("query");
  queryEl.textContent = `"${data.query_text}"`;
  const tag = document.getElementById("new-query-tag");
  // Re-trigger the flash: removing and re-adding in one frame is a no-op.
  queryEl.classList.remove("new-query");
  tag.classList.remove("show");
  if (changedQuery) {
    void queryEl.offsetWidth;
    queryEl.classList.add("new-query");
    tag.classList.add("show");
  }

  document.getElementById("group").textContent =
    `câu ${data.group_index}/${data.group_total_queries}` +
    ` · ảnh ${data.group_position}/${data.group_total} của câu này`;
  document.getElementById("img").src = data.image_url;
  document.getElementById("undo-note").textContent =
    previousGrade === null || previousGrade === undefined
      ? ""
      : `đã lùi lại — trước đó bạn chấm ${previousGrade}, bấm lại để sửa`;
  loadProgress();
}

async function loadProgress() {
  const res = await fetch("/api/progress");
  const p = await res.json();
  const mine = p.per_annotator[annotator] || 0;
  document.getElementById("progress").textContent =
    `Toàn nhóm: ${p.fully_labeled}/${p.total_rows} xong · Bạn đã chấm: ${mine} lượt`;
}

async function grade(g) {
  if (!current || busy) return;
  busy = true;
  try {
    await fetch("/api/grade", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        query_id: current.query_id,
        product_id: current.product_id,
        annotator: annotator,
        grade: g,
      }),
    });
    await loadNext();
  } finally {
    busy = false;
  }
}

async function undoLast() {
  if (busy || !annotator) return;
  busy = true;
  const note = document.getElementById("undo-note");
  try {
    const res = await fetch("/api/undo", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({annotator: annotator}),
    });
    const data = await res.json();
    if (!data.undone) {
      note.textContent = data.reason || "không có gì để hoàn tác";
      return;
    }
    // The row is gone from the loaded template (undone against another
    // template) — fall back to the normal queue rather than showing nothing.
    if (!data.row) {
      note.textContent = "đã hoàn tác";
      await loadNext();
      return;
    }
    document.getElementById("work").style.display = "block";
    document.getElementById("done").style.display = "none";
    showRow(data.row, data.previous_grade);
  } finally {
    busy = false;
  }
}

document.addEventListener("keydown", (e) => {
  if (e.key === "0") grade(0);
  if (e.key === "1") grade(1);
  if (e.key === "2") grade(2);
  if (e.key === "Backspace") { e.preventDefault(); undoLast(); }
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # no packet actually sent; just picks the outbound interface
            local_ip = s.getsockname()[0]
    except OSError:
        local_ip = "127.0.0.1"

    print(f"{len(ROWS)} dòng cần chấm.")
    print("Mở 1 trong các địa chỉ sau (máy này hoặc máy khác cùng wifi/LAN):")
    print("  http://localhost:8001")
    print(f"  http://{local_ip}:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
