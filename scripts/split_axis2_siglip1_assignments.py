"""Split the axis2 + siglip1 pool extension across exactly 2 graders.

Same greedy 2-graders-per-row logic as scripts/split_assignments.py (imported,
not re-implemented), pointed at relevance_template_axis2_siglip1.json and
writing data/eval/assignment_axis2siglip1_<name>.json — mirrors what
scripts/split_ft_assignments.py did for the fine-tuned round. With exactly 2
names and no prior grades on these brand-new pairs, both graders get every row
(the full independent overlap DeCuong Mục 3.4 asks for).

Usage:
    python scripts/split_axis2_siglip1_assignments.py Hùng Hiệp
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.split_assignments import (  # noqa: E402
    load_existing_graders,
    split_rows,
    write_assignments,
)

TEMPLATE_PATH = ROOT / "data" / "eval" / "relevance_template_axis2_siglip1.json"
OUT_PREFIX = "assignment_axis2siglip1"


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    names = sys.argv[1:]
    if len(names) < 2:
        raise SystemExit(
            "cần ít nhất 2 tên, ví dụ: "
            "python scripts/split_axis2_siglip1_assignments.py Hùng Hiệp"
        )
    if not TEMPLATE_PATH.exists():
        raise SystemExit(
            f"{TEMPLATE_PATH} chưa có — chạy "
            "scripts/build_axis2_siglip1_pool_extension.py trước"
        )

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    assigned, fully_covered, already_needed_total = split_rows(
        rows, names, load_existing_graders()
    )
    write_assignments(assigned, names, prefix=OUT_PREFIX)

    print(f"\nTổng: {len(rows)} dòng | đã đủ 2 người chấm: {fully_covered} | "
          f"còn cần chấm: {already_needed_total} lượt, chia cho {len(names)} người")


if __name__ == "__main__":
    main()
