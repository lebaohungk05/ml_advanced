"""Sprint 4 ablation runner: one command per ablation cell, no code editing.

Every cell inherits the EXACT reference hyperparameters of the completed run
``hung-run1`` (which is plain ``TrainConfig()`` plus a ``run_id``) and overrides
one single field, so a difference in the resulting metrics can be attributed to
that field alone. ``--list`` prints the diff of every cell against the reference
so the one-variable property is visible without reading the code.

The sprint plan (docs/KeHoach_Sprint_MultimodalFashionSearch.md, Sprint 4) lists
7 axes; only 5 of them need extra training runs, and ``hung-run1`` already is
the "DoRA, rank 8, effective batch 256, both towers, hard negatives on" cell of
each of them, so 8 runs cover the whole table:

* axis 1 — DoRA vs plain LoRA: 1 run (``use_dora=False``)
* axis 2 — hard-negative mining on/off: 1 run (``use_hard_negatives=False``)
* axis 3 — SigLIP2 vs ViSigLIP-OT: NOT here. Both are zero-shot backbones and
  the answer is already in the Sprint 2 zero-shot baseline table; fine-tuning a
  second backbone would change two things at once (backbone AND adapter) and
  answer a different question. Offering a training cell for it would invite
  exactly that confusion, so the registry deliberately has none.
* axis 4 — LoRA rank {4, 8, 16}: 2 runs (rank 8 is the reference)
* axis 5 — effective batch {32, 128, 256}: 2 runs (256 is the reference).
  ``physical_batch_size`` stays 32 in both: 32 and 128 are already multiples of
  32, so ``TrainConfig.__post_init__`` is satisfied without touching a second
  field. Only the GradCache accumulation depth changes.
* axis 6 — adapter image-only / text-only / both: 2 runs (both is the reference)
* axis 7 — SigLIP1 vs SigLIP2: NOT here either. It is an inference-only
  comparison — a different ``model_id`` through the existing embedder on the
  eval side — and needs no adapter training at all.

Usage:
    python scripts/run_ablation.py --list
    python scripts/run_ablation.py --axis axis1_lora_no_dora --dry-run
    python scripts/run_ablation.py --axis axis1_lora_no_dora
    python scripts/run_ablation.py --all
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapters.training.lora_dora_trainer import TrainConfig  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
IMAGE_ROOT = ROOT / "data" / "raw" / "fashionpedia"

REFERENCE_RUN_ID = "hung-run1"
REFERENCE_RUNTIME_S = 11084.7
"""Measured wall clock of ``hung-run1`` (experiments/runs.csv) — the runtime estimate.

Every cell processes the same 32.5k products for the same 10 epochs, so the
per-cell estimate is the same number rather than a made-up scaling factor.
"""


@dataclass(frozen=True, slots=True)
class AblationCell:
    """One ablation cell: an axis key, a Vietnamese label, and its overrides."""

    key: str
    axis: str
    description: str
    overrides: Mapping[str, Any]


CELLS: tuple[AblationCell, ...] = (
    AblationCell(
        key="axis1_lora_no_dora",
        axis="1. DoRA vs LoRA",
        description="LoRA thường (tắt DoRA), giữ nguyên rank 8",
        overrides={"use_dora": False},
    ),
    AblationCell(
        key="axis2_no_hard_negatives",
        axis="2. Hard negative",
        description="Tắt hard-negative mining, batch xếp ngẫu nhiên",
        overrides={"use_hard_negatives": False},
    ),
    AblationCell(
        key="axis4_rank4",
        axis="4. LoRA rank",
        description="LoRA rank 4 (ít tham số hơn reference)",
        overrides={"lora_rank": 4},
    ),
    AblationCell(
        key="axis4_rank16",
        axis="4. LoRA rank",
        description="LoRA rank 16 (nhiều tham số hơn reference)",
        overrides={"lora_rank": 16},
    ),
    AblationCell(
        key="axis5_batch32",
        axis="5. Batch hiệu dụng",
        description="Batch hiệu dụng 32 (ít negative in-batch nhất)",
        overrides={"effective_batch_size": 32},
    ),
    AblationCell(
        key="axis5_batch128",
        axis="5. Batch hiệu dụng",
        description="Batch hiệu dụng 128 (một nửa reference)",
        overrides={"effective_batch_size": 128},
    ),
    AblationCell(
        key="axis6_image_only",
        axis="6. Tower được adapt",
        description="Chỉ adapt tower ảnh, tower text đóng băng",
        overrides={"adapt_text_tower": False},
    ),
    AblationCell(
        key="axis6_text_only",
        axis="6. Tower được adapt",
        description="Chỉ adapt tower text, tower ảnh đóng băng",
        overrides={"adapt_image_tower": False},
    ),
)


def reference_config(
    output_dir: Path = ROOT / "checkpoints",
    wandb_project: str | None = "fashion-multimodal-search",
) -> TrainConfig:
    """The hyperparameters of ``hung-run1``: ``TrainConfig`` defaults, nothing tuned."""
    return TrainConfig(
        run_id=REFERENCE_RUN_ID,
        output_dir=output_dir,
        wandb_project=wandb_project,
        extra={"image_root": str(IMAGE_ROOT)},
    )


def cell_config(
    cell: AblationCell,
    output_dir: Path = ROOT / "checkpoints",
    wandb_project: str | None = "fashion-multimodal-search",
) -> TrainConfig:
    """Reference config with ``cell.overrides`` applied and ``run_id`` = the axis key."""
    return dataclasses.replace(
        reference_config(output_dir, wandb_project),
        run_id=cell.key,
        **dict(cell.overrides),
    )


def config_differences(config: TrainConfig, reference: TrainConfig) -> dict[str, tuple[Any, Any]]:
    """``{field: (reference, cell)}``, ignoring the intentionally-different ``run_id``."""
    left = dataclasses.asdict(reference)
    right = dataclasses.asdict(config)
    return {
        name: (left[name], right[name])
        for name in left
        if name != "run_id" and left[name] != right[name]
    }


def find_cell(key: str) -> AblationCell:
    for cell in CELLS:
        if cell.key == key:
            return cell
    raise SystemExit(
        f"không có trục '{key}'. Các key hợp lệ:\n  "
        + "\n  ".join(cell.key for cell in CELLS)
    )


def checkpoint_dir(cell: AblationCell, output_dir: Path) -> Path:
    return output_dir / cell.key / "best"


def format_registry(output_dir: Path) -> str:
    """The ``--list`` table: key, description, diff vs reference, runtime estimate."""
    reference = reference_config(output_dir)
    hours = REFERENCE_RUNTIME_S / 3600.0
    lines = [
        f"Reference run: {REFERENCE_RUN_ID} (DoRA, rank 8, lr 1e-4, batch 256, "
        f"10 epoch, {REFERENCE_RUNTIME_S:.0f}s)",
        f"Mỗi cell dưới đây chỉ khác reference ĐÚNG 1 field. Ước tính ~{hours:.1f}h/run.",
        "",
    ]
    for cell in CELLS:
        diff = config_differences(cell_config(cell, output_dir), reference)
        changed = ", ".join(f"{name}: {old} -> {new}" for name, (old, new) in diff.items())
        lines.append(f"{cell.key}  [{cell.axis}]")
        lines.append(f"    {cell.description}")
        lines.append(f"    khác reference: {changed}")
        lines.append(f"    ước tính: ~{hours:.1f}h  ->  {checkpoint_dir(cell, output_dir)}")
    lines.append("")
    lines.append(
        "Trục 3 (SigLIP2 vs ViSigLIP-OT) và trục 7 (SigLIP1 vs SigLIP2) không cần "
        "training — xem docstring đầu file."
    )
    return "\n".join(lines)


def format_dry_run(cell: AblationCell, config: TrainConfig, output_dir: Path) -> str:
    diff = config_differences(config, reference_config(output_dir))
    lines = [
        f"[dry-run] {cell.key} — {cell.description}",
        f"[dry-run] khác reference {REFERENCE_RUN_ID}: "
        + ", ".join(f"{name}: {old} -> {new}" for name, (old, new) in diff.items()),
        f"[dry-run] checkpoint sẽ ghi vào: {checkpoint_dir(cell, output_dir)}",
        "[dry-run] TrainConfig:",
    ]
    for name, value in sorted(dataclasses.asdict(config).items()):
        lines.append(f"    {name} = {value!r}")
    lines.append(
        f"[dry-run] grad_accumulation_steps = {config.grad_accumulation_steps} "
        f"({config.effective_batch_size} / {config.physical_batch_size})"
    )
    return "\n".join(lines)


def selected_cells(args: argparse.Namespace) -> list[AblationCell]:
    if args.all:
        return list(CELLS)
    return [find_cell(args.axis)]


def guard_existing_checkpoints(cells: Sequence[AblationCell], output_dir: Path) -> None:
    """Fail before any GPU time is spent if a run would overwrite a finished one."""
    clashes = [cell for cell in cells if checkpoint_dir(cell, output_dir).exists()]
    if clashes:
        listed = "\n  ".join(str(checkpoint_dir(cell, output_dir)) for cell in clashes)
        raise SystemExit(
            "checkpoint đã tồn tại, dừng để không ghi đè kết quả cũ:\n  "
            f"{listed}\n"
            "Thêm --force nếu thực sự muốn chạy lại và ghi đè."
        )


def run_cell(cell: AblationCell, config: TrainConfig) -> Path:
    from src.adapters.training.lora_dora_trainer import train
    from src.adapters.training.product_loading import load_training_products

    if not (PROCESSED / "train.json").exists():
        raise SystemExit(
            "data/processed/train.json chưa có -- chạy trước:\n"
            "  python scripts/download_fashionpedia.py\n"
            "  python scripts/prepare_catalog.py"
        )

    train_products, train_skipped = load_training_products(PROCESSED / "train.json", IMAGE_ROOT)
    val_products, val_skipped = load_training_products(PROCESSED / "val.json", IMAGE_ROOT)
    if train_skipped or val_skipped:
        print(f"bỏ qua {train_skipped} train / {val_skipped} val sản phẩm thiếu ảnh trên đĩa")
    print(f"\n=== {cell.key} — {cell.description} ===")
    print(f"train: {len(train_products)} sản phẩm | val: {len(val_products)} sản phẩm")

    checkpoint = train(config, train_products, val_products)
    print(f"\n[{cell.key}] checkpoint: {checkpoint}")
    print(
        f"[{cell.key}] LƯU Ý: cột recall@5 mà run này ghi vào experiments/runs.csv là "
        "self-query proxy (caption của chính sản phẩm), KHÔNG phải số headline. "
        "Phải chạy bước eval trên tập truy vấn đã chấm nhãn mới có "
        "Recall/MRR/nDCG so sánh được giữa các trục."
    )
    return checkpoint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy các cell ablation Sprint 4, mỗi lệnh 1 trục.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="in bảng registry rồi thoát")
    mode.add_argument("--axis", default=None, help="key của cell cần chạy")
    mode.add_argument("--all", action="store_true", help="chạy tuần tự toàn bộ cell")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="in TrainConfig đã resolve, không load torch, không train",
    )
    parser.add_argument(
        "--force", action="store_true", help="cho phép ghi đè checkpoint đã có"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--wandb-project", default="fashion-multimodal-search")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args(argv)


def _force_utf8_output() -> None:
    """Windows console defaults to cp1252 and mangles/raises on the Vietnamese output.

    Guarded on the current encoding so that running under pytest's capture (already
    UTF-8) leaves the captured streams alone. stderr too: the error messages here
    are Vietnamese as well.
    """
    if not _is_utf8(sys.stdout):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if not _is_utf8(sys.stderr):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def _is_utf8(stream: Any) -> bool:
    encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
    return encoding == "utf8"


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_output()
    args = parse_args(argv)

    if args.list:
        print(format_registry(args.output_dir))
        return 0

    cells = selected_cells(args)
    wandb_project = None if args.no_wandb else args.wandb_project
    configs = [cell_config(cell, args.output_dir, wandb_project) for cell in cells]

    if args.dry_run:
        for cell, config in zip(cells, configs, strict=True):
            print(format_dry_run(cell, config, args.output_dir))
            print()
        return 0

    if not args.force:
        guard_existing_checkpoints(cells, args.output_dir)

    for cell, config in zip(cells, configs, strict=True):
        run_cell(cell, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
