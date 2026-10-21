"""Hiếu left the team before grading any of their share. Split their originally
assigned rows (recovered from the kit snapshot, since Hiếu's own template never
got touched) across the 3 remaining people (Hùng, Hiệp, Hưng), reading REAL
grades_*.jsonl files to see who has already graded which row -- so nobody gets
handed a row they (or someone else, twice) already covered.

Output: data/eval/extra_assignment_<name>.json -- ADDITIONAL rows on top of
what each person already graded, not a replacement of their existing kit.
"""

from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
OUT_DIR = ROOT / "data" / "eval"

REMAINING_NAMES = ["Hùng", "Hiệp", "Hưng"]


def load_actual_graders() -> dict[tuple[str, str], set[str]]:
    graders: dict[tuple[str, str], set[str]] = defaultdict(set)
    for name in REMAINING_NAMES:
        path = OUT_DIR / f"grades_{name}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                graders[(rec["query_id"], rec["product_id"])].add(rec["annotator"])
    return graders


def main() -> None:
    hieu_path = DIST / "labeling_kit_Hiếu" / "data" / "eval" / "relevance_template.json"
    with open(hieu_path, encoding="utf-8") as f:
        hieu_rows = json.load(f)

    actual_graders = load_actual_graders()

    load_count = {name: 0 for name in REMAINING_NAMES}
    extra: dict[str, list[dict]] = {name: [] for name in REMAINING_NAMES}
    already_done = 0
    skipped_no_eligible = 0

    for row in hieu_rows:
        key = (row["query_id"], row["product_id"])
        current = actual_graders.get(key, set())
        if len(current) >= 2:
            already_done += 1
            continue
        eligible = [n for n in REMAINING_NAMES if n not in current]
        if not eligible:
            skipped_no_eligible += 1
            continue
        eligible.sort(key=lambda n: load_count[n])
        chosen = eligible[0]
        extra[chosen].append(row)
        load_count[chosen] += 1

    for name in REMAINING_NAMES:
        out_path = OUT_DIR / f"extra_assignment_{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(extra[name], f, ensure_ascii=False, indent=1)
        print(f"{name}: +{len(extra[name])} dòng (thêm) -> {out_path}")

    print(f"\nTổng phần của Hiếu: {len(hieu_rows)} dòng | đã đủ 2 người chấm rồi: "
          f"{already_done} | không ai đủ điều kiện: {skipped_no_eligible}")


if __name__ == "__main__":
    main()
