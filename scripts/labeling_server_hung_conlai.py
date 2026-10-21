"""Chạy server chấm nhãn CHỈ với 572 dòng còn thiếu của Hưng — không cần set
biến môi trường, không sợ quên. Dùng thay cho labeling_server.py khi chấm nốt
phần riêng của mình.

Chạy:
    python scripts/labeling_server_hung_conlai.py
"""

import os
from pathlib import Path

os.environ["LABELING_TEMPLATE"] = str(
    Path(__file__).resolve().parent.parent / "data" / "eval" / "relevance_template_Hưng_conlai.json"
)

import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "labeling_server.py"), run_name="__main__")
