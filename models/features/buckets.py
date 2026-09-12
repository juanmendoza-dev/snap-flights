"""Advance-purchase buckets (SF-06-build §3.1).

The bucket bounds come from ``config/baseline.yaml``. There is a scalar path and a polars
path, built from the same config, so the frame path and the scalar path cannot drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from models.baseline.config import BaselineConfig

AP_BUCKETS: tuple[str, ...] = (
    "0-3", "4-7", "8-14", "15-21", "22-30", "31-45", "46-60", "61-90", "90+",
)  # fmt: skip


def resolve_config(config: BaselineConfig | None) -> BaselineConfig:
    """The given config, or the committed default. Imported lazily: models.baseline imports
    the feature builders, so a module-level import here would be circular."""
    if config is not None:
        return config
    from models.baseline.config import load_baseline_config

    return load_baseline_config()


def ap_bucket(days_to_departure: int, config: BaselineConfig | None = None) -> str:
    """Bucket name for a days-to-departure value. Bounds are inclusive on both ends.
    Negative days_to_departure raises ValueError — a departed flight has no AP bucket."""
    if days_to_departure < 0:
        raise ValueError(f"days_to_departure must be >= 0, got {days_to_departure}")
    for bucket in resolve_config(config).advance_purchase_buckets:
        if bucket.max_days is None or days_to_departure <= bucket.max_days:
            return bucket.name
    raise AssertionError("unreachable: the config validator guarantees an unbounded last bucket")


def ap_bucket_expr(config: BaselineConfig | None = None) -> pl.Expr:
    """The same mapping as a polars expression over a `days_to_departure` column, so the
    frame path and the scalar path cannot drift. Tested for equivalence over 0..400.
    Negative values map to null."""
    dtd = pl.col("days_to_departure")
    buckets = resolve_config(config).advance_purchase_buckets
    expr: pl.Expr = pl.lit(None, dtype=pl.String)
    # Built from the last bucket backwards so the first matching bucket ends up outermost.
    for bucket in reversed(buckets):
        condition = dtd >= bucket.min_days
        if bucket.max_days is not None:
            condition = condition & (dtd <= bucket.max_days)
        expr = pl.when(condition).then(pl.lit(bucket.name)).otherwise(expr)
    return expr.alias("ap_bucket")
