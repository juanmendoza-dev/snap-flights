"""Route history: one collapsed observation per (route, depart_date, fetched_date)
(SF-06-build §3.2).

``store.read_frame()`` has already excluded ``data_quality = "rejected"`` and deduplicated on
``observation_id`` (SF-03-build §9). Nothing here repeats either.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from models.features.buckets import ap_bucket_expr, resolve_config
from pipeline.store import ReadFilters, SnapshotStore, default_store

if TYPE_CHECKING:
    from models.baseline.config import BaselineConfig

OBSERVATION_FRAME_COLUMNS: tuple[str, ...] = (
    "route_key", "source", "price_kind", "fetched_date", "depart_date",
    "days_to_departure", "ap_bucket", "travel_month", "travel_dow", "amount_minor",
    "currency", "fetched_at",
)  # fmt: skip

OBSERVATION_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "route_key": pl.String(),
    "source": pl.String(),
    "price_kind": pl.String(),
    "fetched_date": pl.Date(),
    "depart_date": pl.Date(),
    "days_to_departure": pl.Int64(),
    "ap_bucket": pl.String(),
    "travel_month": pl.Int64(),
    "travel_dow": pl.Int64(),
    "amount_minor": pl.Int64(),
    "currency": pl.String(),
    "fetched_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}

_GROUP: tuple[str, ...] = ("route_key", "depart_date", "fetched_date")


def empty_observation_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=OBSERVATION_FRAME_SCHEMA)


def collapse_observations(
    raw: pl.DataFrame, config: BaselineConfig | None = None
) -> tuple[pl.DataFrame, int]:
    """Turn a ``read_frame()``-shaped frame into the observation frame.

    Returns ``(frame, dropped_currency_mismatch)``. Steps, in order:

    1. ``fetched_date`` = UTC date of ``fetched_at``; ``days_to_departure`` =
       ``(depart_date - fetched_date).days``. Rows with ``days_to_departure < 0`` are dropped.
    2. Per route, rows whose ``currency`` differs from the modal currency are dropped and
       counted (ties on the mode break alphabetically). This runs before the collapse so a
       price is never compared against one in another unit.
    3. Prefer-itinerary collapse: per ``(route_key, depart_date, fetched_date)`` keep the
       cheapest ``itinerary`` row if there is one, else the cheapest ``calendar_cheapest`` row;
       ties break on ``observation_id`` ascending.
    4. ``ap_bucket``, ``travel_month`` (1-12) and ``travel_dow`` (Mon=0) — month and weekday
       OF TRAVEL, i.e. from ``depart_date``, never ``fetched_date``.
    """
    config = resolve_config(config)
    if raw.is_empty():
        return empty_observation_frame(), 0

    frame = raw.with_columns(
        pl.col("fetched_at").dt.convert_time_zone("UTC").dt.date().alias("fetched_date")
    ).with_columns(
        (pl.col("depart_date") - pl.col("fetched_date")).dt.total_days().alias("days_to_departure")
    )
    frame = frame.filter(pl.col("days_to_departure") >= 0)

    modal = (
        frame.group_by("route_key", "currency")
        .len()
        .sort(["route_key", "len", "currency"], descending=[False, True, False])
        .unique(subset="route_key", keep="first", maintain_order=True)
        .select("route_key", pl.col("currency").alias("_modal_currency"))
    )
    before = frame.height
    frame = (
        frame.join(modal, on="route_key", how="left")
        .filter(pl.col("currency") == pl.col("_modal_currency"))
        .drop("_modal_currency")
    )
    dropped = before - frame.height

    frame = (
        frame.with_columns((pl.col("price_kind") == "itinerary").alias("_is_itinerary"))
        .sort(
            [*_GROUP, "_is_itinerary", "amount_minor", "observation_id"],
            descending=[False, False, False, True, False, False],
        )
        .unique(subset=list(_GROUP), keep="first", maintain_order=True)
    )

    frame = frame.with_columns(
        ap_bucket_expr(config),
        pl.col("depart_date").dt.month().cast(pl.Int64).alias("travel_month"),
        (pl.col("depart_date").dt.weekday().cast(pl.Int64) - 1).alias("travel_dow"),
    )
    frame = frame.select(
        [pl.col(name).cast(dtype) for name, dtype in OBSERVATION_FRAME_SCHEMA.items()]
    ).sort(["depart_date", "fetched_date", "route_key"])
    return frame, dropped


def load_route_history_with_stats(
    route_key: str,
    *,
    as_of: date,
    config: BaselineConfig | None = None,
    store: SnapshotStore | None = None,
) -> tuple[pl.DataFrame, int]:
    """``load_route_history`` plus the number of rows dropped for a currency mismatch, which
    ``basis`` reports and a bare frame has nowhere to carry."""
    config = resolve_config(config)
    store = store if store is not None else default_store()
    raw = store.read_frame(
        ReadFilters(
            route_key=route_key,
            fetched_date_from=as_of - timedelta(days=config.history.window_days),
            fetched_date_to=as_of,
        )
    )
    return collapse_observations(raw, config)


def load_route_history(
    route_key: str,
    *,
    as_of: date,
    config: BaselineConfig | None = None,
    store: SnapshotStore | None = None,
) -> pl.DataFrame:
    """All observations for one route with fetched_date in
    [as_of - config.history.window_days, as_of], via store.read_frame().

    Adds the derived columns:
      days_to_departure = (depart_date - fetched_date).days
      ap_bucket         = ap_bucket_expr()
      travel_month      = depart_date.month        (1-12, month OF TRAVEL)
      travel_dow        = depart_date.weekday()    (0=Mon .. 6=Sun, DOW OF TRAVEL)

    Rows with days_to_departure < 0 are dropped (the flight departed before we saw it).
    Then applies the prefer-itinerary collapse (see collapse_observations).

    Returns a frame with exactly OBSERVATION_FRAME_COLUMNS, sorted by
    (depart_date, fetched_date). Empty history returns an empty frame with that schema.
    """
    frame, _ = load_route_history_with_stats(route_key, as_of=as_of, config=config, store=store)
    return frame
