"""Trailing price distributions (SF-06-build §3.3)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from models.features.distributions import (
    BUCKET_DISTRIBUTION_COLUMNS,
    DISTRIBUTION_COLUMNS,
    build_bucket_distribution,
    build_distribution,
)
from models.features.observations import OBSERVATION_FRAME_SCHEMA, empty_observation_frame
from tests.models.conftest import fixture_route_keys

#: SF-03-build §8 — rows per (route, AP bucket) over the 90 fixture fetch days, after the
#: prefer-itinerary collapse (tier-1 routes are not doubled).
SF03_CELL_COUNTS: dict[str, int] = {
    "0-3": 270,
    "4-7": 360,
    "8-14": 630,
    "15-21": 630,
    "22-30": 810,
    "31-45": 1350,
    "46-60": 1350,
    "61-90": 2700,
    "90+": 2700,
}


def _frame(amounts: list[int], *, ap_bucket: str = "8-14") -> pl.DataFrame:
    n = len(amounts)
    return pl.DataFrame(
        {
            "route_key": ["JFK-LHR"] * n,
            "source": ["travelpayouts"] * n,
            "price_kind": ["calendar_cheapest"] * n,
            "fetched_date": [date(2026, 9, 1)] * n,
            "depart_date": [date(2026, 9, 11)] * n,
            "days_to_departure": [10] * n,
            "ap_bucket": [ap_bucket] * n,
            "travel_month": [9] * n,
            "travel_dow": [4] * n,
            "amount_minor": amounts,
            "currency": ["USD"] * n,
            "fetched_at": [datetime(2026, 9, 1, 6, tzinfo=UTC)] * n,
        },
        schema=OBSERVATION_FRAME_SCHEMA,
    )


@pytest.mark.parametrize("route_key", fixture_route_keys())
def test_cell_counts_match_sf03_build_table(route_key: str, fixture_history) -> None:
    buckets = build_bucket_distribution(fixture_history(route_key))

    assert dict(zip(buckets["ap_bucket"], buckets["count"], strict=True)) == SF03_CELL_COUNTS


def test_cov_null_on_single_row() -> None:
    row = build_bucket_distribution(_frame([42000])).row(0, named=True)

    assert row["count"] == 1
    assert row["cov"] is None
    assert row["median"] == 42000.0


def test_cov_uses_population_std() -> None:
    row = build_bucket_distribution(_frame([10000, 30000])).row(0, named=True)

    assert row["std"] == pytest.approx(10000.0)
    assert row["cov"] == pytest.approx(0.5)


def test_percentiles_interpolate_linearly() -> None:
    row = build_bucket_distribution(_frame([100, 200, 300, 400])).row(0, named=True)

    assert row["p10"] == pytest.approx(130.0)
    assert row["p25"] == pytest.approx(175.0)
    assert row["median"] == pytest.approx(250.0)
    assert row["p75"] == pytest.approx(325.0)
    assert row["p90"] == pytest.approx(370.0)


@pytest.mark.parametrize("route_key", ["JFK-LHR", "MAD-LIS", "LAX-NRT"])
def test_percentiles_ordered(route_key: str, fixture_history) -> None:
    history = fixture_history(route_key)

    for cells in (build_distribution(history), build_bucket_distribution(history)):
        bad = cells.filter(
            (pl.col("p10") > pl.col("p25"))
            | (pl.col("p25") > pl.col("median"))
            | (pl.col("median") > pl.col("p75"))
            | (pl.col("p75") > pl.col("p90"))
        )
        assert bad.is_empty()


def test_columns_are_frozen_including_empty(fixture_history) -> None:
    history = fixture_history("JFK-LHR")

    assert tuple(build_distribution(history).columns) == DISTRIBUTION_COLUMNS
    assert tuple(build_bucket_distribution(history).columns) == BUCKET_DISTRIBUTION_COLUMNS
    assert tuple(build_distribution(empty_observation_frame()).columns) == DISTRIBUTION_COLUMNS
    assert (
        tuple(build_bucket_distribution(empty_observation_frame()).columns)
        == BUCKET_DISTRIBUTION_COLUMNS
    )


def test_distribution_keys_on_month_and_dow_of_travel(fixture_history) -> None:
    cells = build_distribution(fixture_history("JFK-LHR"))

    assert cells["travel_month"].is_between(1, 12).all()
    assert cells["travel_dow"].is_between(0, 6).all()
    assert cells.group_by("ap_bucket", "travel_month", "travel_dow").len()["len"].max() == 1
