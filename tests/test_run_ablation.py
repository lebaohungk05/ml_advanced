"""Sprint 4 ablation registry: the one-variable-per-cell property and the CLI.

Nothing here touches a GPU or downloads weights: ``--list`` and ``--dry-run``
must resolve every ``TrainConfig`` without importing torch, and that is asserted
rather than assumed.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_runner() -> ModuleType:
    """Import ``scripts/run_ablation.py`` (``scripts`` is not an installed package)."""
    spec = importlib.util.spec_from_file_location(
        "run_ablation_under_test", ROOT / "scripts" / "run_ablation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_registry_covers_only_the_axes_that_need_training() -> None:
    keys = [cell.key for cell in runner.CELLS]

    assert len(keys) == len(set(keys)) == 8
    # Axes 3 and 7 are answered without fine-tuning (see the module docstring).
    assert not any(key.startswith(("axis3", "axis7")) for key in keys)
    assert sorted(keys) == sorted(
        [
            "axis1_lora_no_dora",
            "axis2_no_hard_negatives",
            "axis4_rank4",
            "axis4_rank16",
            "axis5_batch32",
            "axis5_batch128",
            "axis6_image_only",
            "axis6_text_only",
        ]
    )


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("axis1_lora_no_dora", {"use_dora": False}),
        ("axis2_no_hard_negatives", {"use_hard_negatives": False}),
        ("axis4_rank4", {"lora_rank": 4}),
        ("axis4_rank16", {"lora_rank": 16}),
        ("axis5_batch32", {"effective_batch_size": 32}),
        ("axis5_batch128", {"effective_batch_size": 128}),
        ("axis6_image_only", {"adapt_text_tower": False}),
        ("axis6_text_only", {"adapt_image_tower": False}),
    ],
)
def test_each_cell_changes_exactly_the_intended_field(
    key: str, expected: dict[str, object]
) -> None:
    """The ablation is only valid if ONE variable moves per cell."""
    cell = runner.find_cell(key)
    reference = runner.reference_config()

    diff = runner.config_differences(runner.cell_config(cell), reference)

    assert {name: new for name, (_old, new) in diff.items()} == expected


def test_cells_inherit_every_other_reference_hyperparameter() -> None:
    reference = dataclasses.asdict(runner.reference_config())
    for cell in runner.CELLS:
        config = dataclasses.asdict(runner.cell_config(cell))
        assert config["run_id"] == cell.key
        untouched = set(reference) - set(cell.overrides) - {"run_id"}
        for name in untouched:
            assert config[name] == reference[name], f"{cell.key} drifted on {name}"


def test_every_cell_passes_trainconfig_validation() -> None:
    for cell in runner.CELLS:
        config = runner.cell_config(cell)  # __post_init__ runs here
        assert config.effective_batch_size % config.physical_batch_size == 0
        assert config.grad_accumulation_steps >= 1
        assert config.adapt_image_tower or config.adapt_text_tower


def test_list_and_dry_run_exit_zero_without_importing_torch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sys.modules.pop("torch", None)

    assert runner.main(["--list"]) == 0
    for cell in runner.CELLS:
        assert runner.main(["--axis", cell.key, "--dry-run"]) == 0

    assert "torch" not in sys.modules
    output = capsys.readouterr().out
    assert "axis1_lora_no_dora" in output
    assert "use_dora" in output


def test_dry_run_all_prints_one_config_per_cell(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.main(["--all", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert output.count("[dry-run] TrainConfig:") == len(runner.CELLS)


def test_unknown_axis_key_is_rejected() -> None:
    with pytest.raises(SystemExit, match="axis1_lora_no_dora"):
        runner.main(["--axis", "axis9_nope", "--dry-run"])


def test_existing_checkpoint_is_not_overwritten_without_force(tmp_path: Path) -> None:
    cell = runner.CELLS[0]
    (tmp_path / cell.key / "best").mkdir(parents=True)

    with pytest.raises(SystemExit, match="đã tồn tại"):
        runner.main(["--axis", cell.key, "--output-dir", str(tmp_path)])

    # --force gets past the guard; --dry-run keeps it from actually training.
    forced = ["--axis", cell.key, "--output-dir", str(tmp_path), "--force", "--dry-run"]
    assert runner.main(forced) == 0


def test_a_free_run_id_passes_the_guard(tmp_path: Path) -> None:
    runner.guard_existing_checkpoints(runner.CELLS, tmp_path)
