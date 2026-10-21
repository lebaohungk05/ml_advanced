"""Pool merge + flag wiring for the eval runner.

Two invariants worth locking down, both of which fail silently rather than
loudly:

* adding a system to data/eval/pool_top20.json must not perturb the systems
  already there — otherwise the already-published Sprint 2/3 numbers move under
  our feet when Sprint 4 axis 7 writes its ``siglip1`` key;
* ``--no-adapter`` must actually mean zero-shot — SigLIP 1 loaded with the
  SigLIP 2 DoRA adapter would still produce a complete-looking run file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_script() -> ModuleType:
    """Import ``scripts/run_finetuned_eval.py`` (``scripts`` is not an installed package)."""
    spec = importlib.util.spec_from_file_location(
        "run_finetuned_eval_under_test", ROOT / "scripts" / "run_finetuned_eval.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evalrun = _load_script()

EXISTING = {
    "bm25": {"1": [{"product_id": "fp-1", "score": 7.692062659116517, "rank": 1}]},
    "siglip2": {"1": [{"product_id": "fp-2", "score": 0.1234567890123456, "rank": 1}]},
}
NEW_RUN = {"1": [{"product_id": "fp-9", "score": 0.5, "rank": 1}]}


def _pool_file(tmp_path: Path, pool: dict[str, object] | None = None) -> Path:
    path = tmp_path / "pool_top20.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(EXISTING if pool is None else pool, f, ensure_ascii=False)
    return path


def test_merge_appends_new_system_and_leaves_the_others_byte_identical(tmp_path: Path) -> None:
    path = _pool_file(tmp_path)
    before = path.read_bytes()

    evalrun.merge_into_pool(NEW_RUN, "siglip1", path)

    after = path.read_bytes()
    # Appended at the end, so the old text minus its closing brace is an exact prefix.
    assert after.startswith(before[:-1])
    written = json.loads(after)
    assert list(written) == ["bm25", "siglip2", "siglip1"]
    assert written["bm25"] == EXISTING["bm25"]
    assert written["siglip2"]["1"][0]["score"] == 0.1234567890123456
    assert written["siglip1"] == NEW_RUN


def test_merge_replaces_an_existing_key_in_place_without_reordering(tmp_path: Path) -> None:
    path = _pool_file(tmp_path)

    evalrun.merge_into_pool({"1": []}, "bm25", path)

    written = json.loads(path.read_bytes())
    assert list(written) == ["bm25", "siglip2"]
    assert written["bm25"] == {"1": []}
    assert written["siglip2"] == EXISTING["siglip2"]


def test_merge_keeps_vietnamese_diacritics_unescaped(tmp_path: Path) -> None:
    path = _pool_file(tmp_path, {"bm25": {"1": [{"product_id": "áo-sơ-mi"}]}})

    evalrun.merge_into_pool({"1": []}, "siglip1", path)

    assert "áo-sơ-mi" in path.read_text(encoding="utf-8")


def test_defaults_reproduce_the_sprint3_finetuned_run() -> None:
    args = evalrun.build_parser().parse_args([])

    assert args.system_name == "siglip2_lora"
    assert args.model_id == "google/siglip2-base-patch16-224"
    assert args.batch_size == 32
    assert args.limit is None
    assert evalrun.resolve_adapter_path(args).endswith("best")


def test_no_adapter_gives_a_zero_shot_run() -> None:
    args = evalrun.build_parser().parse_args(
        ["--no-adapter", "--system-name", "siglip1", "--model-id", "google/siglip-base-patch16-256"]
    )

    assert evalrun.resolve_adapter_path(args) is None
    assert args.system_name == "siglip1"
    assert args.model_id == "google/siglip-base-patch16-256"


def test_no_adapter_and_adapter_path_cannot_both_be_given() -> None:
    with pytest.raises(SystemExit):
        evalrun.build_parser().parse_args(["--no-adapter", "--adapter-path", "checkpoints/x"])
