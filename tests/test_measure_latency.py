"""Latency reporting maths: hand-computed percentiles, not shape checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_script() -> ModuleType:
    """Import ``scripts/measure_latency.py`` (``scripts`` is not an installed package)."""
    spec = importlib.util.spec_from_file_location(
        "measure_latency_under_test", ROOT / "scripts" / "measure_latency.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


latency = _load_script()


def test_percentile_interpolates_between_the_two_neighbouring_samples() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # position = (n-1)*q/100 = 9*0.5 = 4.5 -> halfway between samples[4] and samples[5].
    assert latency.percentile(samples, 50.0) == pytest.approx(5.5)
    # 9*0.95 = 8.55 -> samples[8] + 0.55*(samples[9]-samples[8]) = 9 + 0.55.
    assert latency.percentile(samples, 95.0) == pytest.approx(9.55)


def test_percentile_hits_the_exact_sample_on_whole_positions() -> None:
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]

    assert latency.percentile(samples, 0.0) == pytest.approx(10.0)
    assert latency.percentile(samples, 50.0) == pytest.approx(30.0)
    assert latency.percentile(samples, 100.0) == pytest.approx(50.0)


def test_percentile_sorts_unordered_samples() -> None:
    assert latency.percentile([9.0, 1.0, 5.0], 50.0) == pytest.approx(5.0)


def test_percentile_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError):
        latency.percentile([], 50.0)
    with pytest.raises(ValueError):
        latency.percentile([1.0], 101.0)


def test_summarise_reports_count_p50_p95_and_mean() -> None:
    samples = [4.0, 1.0, 2.0, 3.0]

    summary = latency.summarise(samples)

    assert summary["n"] == 4
    # position = 3*0.5 = 1.5 -> between 2.0 and 3.0.
    assert summary["p50_ms"] == pytest.approx(2.5)
    # position = 3*0.95 = 2.85 -> 3.0 + 0.85*(4.0-3.0).
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["mean_ms"] == pytest.approx(2.5)


def test_residual_is_the_end_to_end_time_left_over_per_query() -> None:
    measured = {
        "encode_query": [10.0, 20.0],
        "index_search": [5.0, 4.0],
        "end_to_end": [16.0, 25.0],
    }

    assert latency.residual_samples(measured) == pytest.approx([1.0, 1.0])


def test_residual_requires_the_same_number_of_samples_per_component() -> None:
    measured = {"encode_query": [1.0], "index_search": [1.0], "end_to_end": [3.0, 3.0]}

    with pytest.raises(ValueError):
        latency.residual_samples(measured)


class _Loaded:
    device = "cuda:0"


class _LazyCpu:
    """ViSigLIP-OT's shape: the public attribute stays None, ``_device`` is set."""

    device = None
    _device = "cpu"


class _Unloaded:
    device = None


def test_device_of_reads_the_resolved_device_including_the_cpu_override() -> None:
    assert latency.device_of(_Loaded()) == "cuda:0"
    assert latency.device_of(_LazyCpu()) == "cpu"
    assert latency.device_of(_Unloaded()) == "unknown"
