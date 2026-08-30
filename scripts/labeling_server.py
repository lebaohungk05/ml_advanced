"""Fast blind-labeling tool for data/eval/relevance_template.json.

Run: python scripts/labeling_server.py
Then open the printed URL in a browser — on the same machine, or on any
device on the same Wi-Fi/LAN (the server binds 0.0.0.0), since it serves the
catalog images straight from this machine's data/raw/fashionpedia/.

Each (query, product) row needs 2 DIFFERENT annotators (DeCuong Mục 3.4 điểm
4). Every submitted grade is appended to data/eval/grades_<annotator>.jsonl
(one file per person, append-only — safe if the server restarts or two people
grade at once). /api/next picks, in order of priority:
  1. a row this annotator hasn't graded yet AND already has exactly 1 grade
     from someone else (finish the pair first)
  2. a row with 0 grades yet
Never serves a row this same annotator already graded, and never serves a
row that already has 2 grades from two different people.

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
def next_row(annotator: str) -> dict[str, Any]:
    if not annotator.strip():
        raise HTTPException(400, "annotator name required")
    graders = _load_existing_grades()

    half_done_by_others = []
    fresh = []
    for row in ROWS:
        key = (row["query_id"], row["product_id"])
        existing = graders.get(key, [])
        if annotator in existing:
            continue
        if len(existing) >= 2:
            continue
        if len(existing) == 1:
            half_done_by_others.append(row)
        else:
            fresh.append(row)

    pick = half_done_by_others[0] if half_done_by_others else (fresh[0] if fresh else None)
    if pick is None:
        return {"done": True}

    return {
        "done": False,
        "query_id": pick["query_id"],
        "query_text": pick["query_text"],
        "product_id": pick["product_id"],
        "image_url": f"/image/{pick['image_path'].replace(chr(92), '/')}",
    }


@app.post("/api/grade")
def submit_grade(payload: GradeIn) -> dict[str, str]:
    if payload.grade not in (0, 1, 2):
        raise HTTPException(400, "grade must be 0, 1 or 2")
    GRADES_DIR.mkdir(parents=True, exist_ok=True)
    path = GRADES_DIR / f"grades_{_safe_filename(payload.annotator)}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload.model_dump(), ensure_ascii=False) + "\n")
    return {"status": "ok"}


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
</style>
</head>
<body>
  <div id="name-box">
    <input id="name-input" placeholder="Tên của bạn (vd: Hùng)">
    <button onclick="startSession()">Bắt đầu</button>
  </div>
  <div id="work" style="display:none">
    <div id="query"></div>
    <div id="img-wrap"><img id="img"></div>
    <div id="buttons">
      <button class="b0" onclick="grade(0)">0 — Không liên quan</button>
      <button class="b1" onclick="grade(1)">1 — Một phần</button>
      <button class="b2" onclick="grade(2)">2 — Đúng ý</button>
    </div>
    <div class="hint">Phím tắt: 0 / 1 / 2 trên bàn phím</div>
    <div id="progress"></div>
  </div>
  <div id="done" style="display:none">🎉 Hết việc — không còn dòng nào cần bạn chấm nữa!</div>

<script>
let annotator = localStorage.getItem("annotator") || "";
let current = null;

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
  const res = await fetch(`/api/next?annotator=${encodeURIComponent(annotator)}`);
  const data = await res.json();
  if (data.done) {
    document.getElementById("work").style.display = "none";
    document.getElementById("done").style.display = "block";
    return;
  }
  current = data;
  document.getElementById("query").textContent = `"${data.query_text}"`;
  document.getElementById("img").src = data.image_url;
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
  if (!current) return;
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
  loadNext();
}

document.addEventListener("keydown", (e) => {
  if (e.key === "0") grade(0);
  if (e.key === "1") grade(1);
  if (e.key === "2") grade(2);
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
