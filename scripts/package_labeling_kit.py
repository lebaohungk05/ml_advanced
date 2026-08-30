"""Package a portable labeling kit for teammates who are NOT on the same
LAN as this machine — only the images actually referenced in
data/eval/relevance_template.json (not the full 46.781-image dataset), plus
the labeling server itself and a short instruction file.

Output: dist/labeling_kit.zip — share this via Drive/Zalo/whatever. Each
teammate unzips it anywhere, then from inside that folder:

    pip install fastapi uvicorn pydantic
    python labeling_server.py

...and opens http://localhost:8001 in their own browser. No GPU, no torch,
no cloning the repo needed for this part.

After everyone finishes, collect each person's data/eval/grades_<name>.jsonl
(created next to labeling_server.py in the unzipped kit) back onto this
machine and drop them into this repo's data/eval/ folder — scripts/labeling_server.py's
own /api/progress and future merge/kappa scripts read every grades_*.jsonl
it finds there, regardless of which machine produced it.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_PATH = ROOT / "data" / "eval" / "relevance_template.json"
IMAGES_ROOT = ROOT / "data" / "raw" / "fashionpedia"

README = """# Bộ chấm nhãn — Fashion Search

## Cài 1 lần (máy chưa có Python packages này)

    pip install fastapi uvicorn pydantic

## Chạy

    python labeling_server.py

Mở trình duyệt vào http://localhost:8001, gõ tên bạn, bấm "Bắt đầu".
Chấm bằng phím 0 / 1 / 2 trên bàn phím.

## Tiêu chí chấm (3 mức)

- 2 = đúng ý — sản phẩm khớp đúng những gì câu hỏi mô tả
- 1 = liên quan một phần — đúng loại trang phục, nhưng sai 1 thuộc tính
  (màu, chất liệu, kiểu dáng...)
- 0 = không liên quan — sai loại trang phục hẳn

Nếu thấy mơ hồ (ví dụ màu be có tính là "trắng" không) — cứ chấm theo cảm
nhận riêng, không cần đúng tuyệt đối. Có nhiều người chấm độc lập nên chỗ
lệch nhau sẽ được xem lại sau.

## Xong việc — gửi lại gì

Sau khi chấm xong (hoặc chấm được bao nhiêu cũng gửi), gửi lại đúng 1 file:

    data/eval/grades_<tên bạn>.jsonl

(nằm cùng thư mục với labeling_server.py, tự sinh ra sau khi bạn chấm ít
nhất 1 dòng) — gửi qua nhóm chat, không cần gửi gì khác.
"""


def build_kit(template_path: Path, kit_name: str) -> None:
    if not template_path.exists():
        raise SystemExit(f"{template_path} chưa có")

    build_dir = ROOT / "dist" / kit_name
    zip_path = ROOT / "dist" / f"{kit_name}.zip"

    with open(template_path, encoding="utf-8") as f:
        rows = json.load(f)
    unique_images = {row["image_path"] for row in rows}

    if build_dir.exists():
        shutil.rmtree(build_dir)
    (build_dir / "data" / "eval").mkdir(parents=True)
    (build_dir / "data" / "raw" / "fashionpedia").mkdir(parents=True)

    print(f"[{kit_name}] Đang copy {len(unique_images)} ảnh...")
    missing = 0
    for i, rel_path in enumerate(sorted(unique_images)):
        src = IMAGES_ROOT / rel_path
        dst = build_dir / "data" / "raw" / "fashionpedia" / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            missing += 1
            continue
        shutil.copy2(src, dst)
        if (i + 1) % 300 == 0:
            print(f"  ... {i + 1}/{len(unique_images)}")
    if missing:
        print(f"CẢNH BÁO: {missing} ảnh không tìm thấy trên đĩa, đã bỏ qua")

    shutil.copy2(template_path, build_dir / "data" / "eval" / "relevance_template.json")

    server_src = ROOT / "scripts" / "labeling_server.py"
    server_text = server_src.read_text(encoding="utf-8")
    # The kit is self-contained: relative paths from the kit's own root instead
    # of this repo's layout (ROOT.parent.parent -> just ROOT of the kit).
    server_text = server_text.replace(
        "ROOT = Path(__file__).resolve().parent.parent\n",
        "ROOT = Path(__file__).resolve().parent\n",
    )
    (build_dir / "labeling_server.py").write_text(server_text, encoding="utf-8")
    (build_dir / "README.md").write_text(README, encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in build_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(build_dir))

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[{kit_name}] Đã tạo: {zip_path} ({size_mb:.1f} MB), {len(rows)} dòng cần chấm\n")


def main() -> None:
    names = sys.argv[1:]
    if not names:
        build_kit(DEFAULT_TEMPLATE_PATH, "labeling_kit")
        return

    for name in names:
        assignment_path = ROOT / "data" / "eval" / f"assignment_{name}.json"
        build_kit(assignment_path, f"labeling_kit_{name}")


if __name__ == "__main__":
    main()
