"""Split the remaining labeling work evenly across N named teammates.

Reads data/eval/relevance_template.json (4011 rows) + every existing
data/eval/grades_*.jsonl (already-submitted grades, e.g. Hưng's own), figures
out how many MORE graders each row still needs (every row needs exactly 2
total), and assigns those remaining slots as evenly as possible across the
names given on the command line — each row's slots go to DISTINCT people, and
running totals are kept balanced with a simple greedy (always give the next
slot to whoever currently has the fewest assigned so far).

Writes one filtered template per person: data/eval/assignment_<name>.json —
feed each into scripts/package_labeling_kit.py (via --template) to build a
personal kit for that person, sized to only their share.

Usage:
    python scripts/split_assignments.py Hùng Hiếu Hiệp
"""

from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "data" / "eval" / "relevance_template.json"
GRADES_DIR = ROOT / "data" / "eval"


def load_existing_graders() -> dict[tuple[str, str], list[str]]:
    graders: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in GRADES_DIR.glob("grades_*.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                graders[(rec["query_id"], rec["product_id"])].append(rec["annotator"])
    return graders


def main() -> None:
    names = sys.argv[1:]
    if len(names) < 2:
        raise SystemExit(
            "cần ít nhất 2 tên, ví dụ: python scripts/split_assignments.py Hùng Hiếu Hiệp"
        )

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    existing = load_existing_graders()

    assigned: dict[str, list[dict]] = {name: [] for name in names}
    load_count: dict[str, int] = {name: 0 for name in names}
    fully_covered = 0
    already_needed_total = 0

    for row in rows:
        key = (row["query_id"], row["product_id"])
        current_graders = set(existing.get(key, []))
        remaining = 2 - len(current_graders)
        if remaining <= 0:
            fully_covered += 1
            continue
        eligible = [n for n in names if n not in current_graders]
        remaining = min(remaining, len(eligible))
        already_needed_total += remaining
        eligible.sort(key=lambda n: load_count[n])
        for name in eligible[:remaining]:
            assigned[name].append(row)
            load_count[name] += 1

    for name in names:
        out_path = GRADES_DIR / f"assignment_{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(assigned[name], f, ensure_ascii=False, indent=1)
        print(f"{name}: {len(assigned[name])} dòng -> {out_path}")

    print(f"\nTổng: {len(rows)} dòng | đã đủ 2 người chấm: {fully_covered} | "
          f"còn cần chấm: {already_needed_total} lượt, chia cho {len(names)} người")


if __name__ == "__main__":
    main()
