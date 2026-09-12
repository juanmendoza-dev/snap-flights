"""The baseline model's config — every threshold SF-06 uses (SF-06-build §2).

``config/baseline.yaml`` is the only place a number lives. ``extra="forbid"`` everywhere: a
typo'd key is an error, not a silently ignored setting.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "config" / "baseline.yaml"

_STRICT = ConfigDict(frozen=True, extra="forbid")


class BucketSpec(BaseModel):
    model_config = _STRICT
    name: str
    min_days: int = Field(ge=0)
    max_days: int | None = None


class HistoryConfig(BaseModel):
    model_config = _STRICT
    window_days: int = Field(ge=1)
    min_cell_observations: int = Field(ge=1)


class CurveConfig(BaseModel):
    model_config = _STRICT
    horizon_days: int = Field(ge=0)
    smoothing_window_days: int = Field(ge=1)
    min_cell_observations: int = Field(ge=1)


class BookNowConfig(BaseModel):
    model_config = _STRICT
    max_percentile: int = Field(ge=0, le=100)
    min_curve_rise_pct: float = Field(ge=0)


class WaitConfig(BaseModel):
    model_config = _STRICT
    min_percentile: int = Field(ge=0, le=100)
    min_curve_drop_pct: float = Field(ge=0, lt=100)
    search_horizon_days: int = Field(ge=1)


class VerdictConfig(BaseModel):
    model_config = _STRICT
    book_now: BookNowConfig
    wait: WaitConfig


class LowConfidenceBand(BaseModel):
    model_config = _STRICT
    max_observations: int = Field(ge=0)
    max_cov: float = Field(ge=0)


class HighConfidenceBand(BaseModel):
    model_config = _STRICT
    min_observations: int = Field(ge=0)
    max_cov: float = Field(ge=0)


class ConfidenceConfig(BaseModel):
    model_config = _STRICT
    low: LowConfidenceBand
    high: HighConfidenceBand


class BacktestConfig(BaseModel):
    model_config = _STRICT
    horizon_days: int = Field(ge=1)
    stride_days: int = Field(ge=1)
    report_path: str


class BaselineConfig(BaseModel):
    model_config = _STRICT
    schema_version: int
    history: HistoryConfig
    advance_purchase_buckets: tuple[BucketSpec, ...]
    curve: CurveConfig
    verdict: VerdictConfig
    confidence: ConfidenceConfig
    backtest: BacktestConfig

    @model_validator(mode="after")
    def _buckets_tile_the_day_line(self) -> BaselineConfig:
        """Buckets must start at 0, be contiguous, and end unbounded — otherwise some
        days-to-departure value has no bucket, or two."""
        buckets = self.advance_purchase_buckets
        if not buckets:
            raise ValueError("advance_purchase_buckets must not be empty")
        expected_min = 0
        for index, bucket in enumerate(buckets):
            if bucket.min_days != expected_min:
                raise ValueError(
                    f"bucket {bucket.name!r} starts at {bucket.min_days}, expected {expected_min}"
                )
            last = index == len(buckets) - 1
            if bucket.max_days is None:
                if not last:
                    raise ValueError(f"only the last bucket may be unbounded, not {bucket.name!r}")
                break
            if bucket.max_days < bucket.min_days:
                raise ValueError(f"bucket {bucket.name!r} has max_days < min_days")
            if last:
                raise ValueError("the last bucket must be unbounded (max_days: null)")
            expected_min = bucket.max_days + 1
        names = [bucket.name for bucket in buckets]
        if len(set(names)) != len(names):
            raise ValueError("bucket names must be unique")
        return self


@lru_cache(maxsize=8)
def _load(resolved: Path) -> BaselineConfig:
    with resolved.open() as handle:
        raw = yaml.safe_load(handle)
    return BaselineConfig.model_validate(raw)


def load_baseline_config(path: Path | None = None) -> BaselineConfig:
    """Parse and validate config/baseline.yaml. Cached per resolved path.
    extra='forbid' everywhere: a typo'd key is an error, not a silently ignored setting."""
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return _load(target.resolve())
