"""config/baseline.yaml and its loader (SF-06-build §2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from models.baseline.config import DEFAULT_CONFIG_PATH, BaselineConfig, load_baseline_config


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_defaults_match_sf06_spec() -> None:
    config = load_baseline_config()

    assert config.verdict.book_now.max_percentile == 25
    assert config.verdict.book_now.min_curve_rise_pct == 5.0
    assert config.verdict.wait.min_percentile == 60
    assert config.verdict.wait.min_curve_drop_pct == 7.0
    assert config.verdict.wait.search_horizon_days == 60
    assert config.confidence.low.max_observations == 100
    assert config.confidence.low.max_cov == 0.35
    assert config.confidence.high.min_observations == 500
    assert config.confidence.high.max_cov == 0.18
    assert config.curve.horizon_days == 90
    assert [b.name for b in config.advance_purchase_buckets] == [
        "0-3", "4-7", "8-14", "15-21", "22-30", "31-45", "46-60", "61-90", "90+",
    ]  # fmt: skip


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["verdict"]["wait"]["min_percentle"] = 60  # the typo that must not pass silently

    with pytest.raises(ValidationError, match="min_percentle"):
        load_baseline_config(_write(tmp_path, raw))


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    del raw["confidence"]["high"]["max_cov"]

    with pytest.raises(ValidationError, match="max_cov"):
        load_baseline_config(_write(tmp_path, raw))


def test_buckets_must_tile_without_gaps(tmp_path: Path) -> None:
    raw = _raw()
    raw["advance_purchase_buckets"][1]["min_days"] = 5  # 4 would have no bucket

    with pytest.raises(ValidationError, match="expected 4"):
        load_baseline_config(_write(tmp_path, raw))


def test_last_bucket_must_be_unbounded(tmp_path: Path) -> None:
    raw = _raw()
    raw["advance_purchase_buckets"][-1]["max_days"] = 400

    with pytest.raises(ValidationError, match="unbounded"):
        load_baseline_config(_write(tmp_path, raw))


def test_loader_is_cached_per_path() -> None:
    assert load_baseline_config() is load_baseline_config(DEFAULT_CONFIG_PATH)


def test_config_is_frozen() -> None:
    config = load_baseline_config()

    with pytest.raises(ValidationError):
        config.verdict.book_now.max_percentile = 90  # type: ignore[misc]
    assert isinstance(config, BaselineConfig)
