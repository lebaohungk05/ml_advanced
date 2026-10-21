"""Chạy server chấm nhãn CHỈ với 143 dòng còn thiếu cuối cùng."""
import os
from pathlib import Path

os.environ["LABELING_TEMPLATE"] = str(
    Path(__file__).resolve().parent.parent / "data" / "eval" / "relevance_template_conlai_143.json"
)

import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "labeling_server.py"), run_name="__main__")
