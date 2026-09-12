"""Trailing price distributions over the observation frame (SF-06-build §3.3).

- Percentiles use linear interpolation between order statistics, so p10/p90 are stable on
  small cells.
- ``cov = std / mean`` with population std (ddof=0); null when ``count < 2`` or
  ``mean == 0``. A null CoV can neither earn ``high`` confidence nor force ``low``.
- Every statistic stays a float in minor units. Rounding to int happens only where a value
  reaches a response.
"""

from __future__ import annotations

import polars as pl

DISTRIBUTION_COLUMNS: tuple[str, ...] = (
    "route_key", "ap_bucket", "travel_month", "travel_dow",
    "count", "median", "p10", "p25", "p75", "p90", "mean", "std", "cov",
)  # fmt: skip

BUCKET_DISTRIBUTION_COLUMNS: tuple[str, ...] = (
    "route_key", "ap_bucket", "count", "median", "p10", "p25", "p75", "p90",
    "mean", "std", "cov",
)  # fmt: skip

_KEY_DTYPES: dict[str, pl.DataType] = {
    "route_key": pl.String(),
    "ap_bucket": pl.String(),
    "travel_month": pl.Int64(),
    "travel_dow": pl.Int64(),
}
_STAT_DTYPES: dict[str, pl.DataType] = {
    "count": pl.Int64(),
    "median": pl.Float64(),
    "p10": pl.Float64(),
    "p25": pl.Float64(),
    "p75": pl.Float64(),
    "p90": pl.Float64(),
    "mean": pl.Float64(),
    "std": pl.Float64(),
    "cov": pl.Float64(),
}


def _stats(observations: pl.DataFrame, keys: tuple[str, ...]) -> pl.DataFrame:
    schema = {key: _KEY_DTYPES[key] for key in keys} | _STAT_DTYPES
    if observations.is_empty():
        return pl.DataFrame(schema=schema)

    amount = pl.col("amount_minor").cast(pl.Float64)
    frame = (
        observations.group_by(list(keys))
        .agg(
            pl.len().alias("count"),
            amount.median().alias("median"),
            amount.quantile(0.10, interpolation="linear").alias("p10"),
            amount.quantile(0.25, interpolation="linear").alias("p25"),
            amount.quantile(0.75, interpolation="linear").alias("p75"),
            amount.quantile(0.90, interpolation="linear").alias("p90"),
            amount.mean().alias("mean"),
            amount.std(ddof=0).alias("std"),
        )
        .with_columns(
            pl.when((pl.col("count") >= 2) & (pl.col("mean") != 0))
            .then(pl.col("std") / pl.col("mean"))
            .otherwise(None)
            .alias("cov")
        )
    )
    return frame.select([pl.col(name).cast(dtype) for name, dtype in schema.items()]).sort(
        list(keys)
    )


def build_distribution(observations: pl.DataFrame) -> pl.DataFrame:
    """Per (route_key, ap_bucket, travel_month, travel_dow). Backs expected_curve."""
    return _stats(observations, ("route_key", "ap_bucket", "travel_month", "travel_dow"))


def build_bucket_distribution(observations: pl.DataFrame) -> pl.DataFrame:
    """Per (route_key, ap_bucket). Backs price_percentile and confidence — this is the
    'cell' SF-06's confidence rules refer to."""
    return _stats(observations, ("route_key", "ap_bucket"))
