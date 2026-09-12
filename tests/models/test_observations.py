"""Route history loading and the prefer-itinerary collapse (SF-06-build §3.2)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from models.baseline.config import load_baseline_config
from models.features.observations import (
    OBSERVATION_FRAME_COLUMNS,
    load_route_history,
    load_route_history_with_stats,
)
from pipeline.schema import PriceKind
from pipeline.store import ReadFilters
from tests.models.conftest import FIXTURE_AS_OF, make_row

AS_OF = date(2026, 9, 9)


def test_prefer_itinerary_collapse(fixture_store, fixture_history) -> None:
    history = fixture_history("JFK-LHR")
    near = history.filter(pl.col("days_to_departure") <= 60)
    far = history.filter(pl.col("days_to_departure") > 60)

    # Tier-1, within 60 days: every group had an itinerary row, and it won.
    assert set(near["price_kind"].unique()) == {"itinerary"}
    assert set(near["source"].unique()) == {"fastflights"}
    assert set(far["price_kind"].unique()) == {"calendar_cheapest"}

    # The surviving amount is the itinerary row's, even where the calendar row is cheaper
    # (the calendar row is the route-level cheapest, so it always is).
    raw = fixture_store.read_frame(
        ReadFilters(route_key="JFK-LHR", fetched_date_from=AS_OF, fetched_date_to=AS_OF)
    )
    depart = AS_OF + timedelta(days=30)
    itinerary = raw.filter(
        (pl.col("depart_date") == depart) & (pl.col("price_kind") == "itinerary")
    )["amount_minor"]
    kept = near.filter((pl.col("depart_date") == depart) & (pl.col("fetched_date") == AS_OF))
    assert kept["amount_minor"].to_list() == [itinerary.min()]


def test_itinerary_wins_even_when_dearer(synthetic_store) -> None:
    depart = date(2026, 10, 1)
    store = synthetic_store(
        [
            make_row(fetched_date=AS_OF, depart_date=depart, amount_minor=30000),
            make_row(
                fetched_date=AS_OF,
                depart_date=depart,
                amount_minor=45000,
                price_kind=PriceKind.ITINERARY,
                carrier_primary="VS",
            ),
            make_row(
                fetched_date=AS_OF,
                depart_date=depart,
                amount_minor=41000,
                price_kind=PriceKind.ITINERARY,
                carrier_primary="BA",
            ),
        ]
    )

    history = load_route_history("JFK-LHR", as_of=AS_OF, store=store)

    assert history.height == 1
    assert history.row(0, named=True)["price_kind"] == "itinerary"
    assert history.row(0, named=True)["amount_minor"] == 41000


def test_collapse_is_one_row_per_group(fixture_history) -> None:
    for route_key in ("JFK-LHR", "ATL-MIA"):
        counts = fixture_history(route_key).group_by("depart_date", "fetched_date").len()
        assert counts["len"].max() == 1


def test_derived_columns(synthetic_store) -> None:
    # 2026-12-20 is a Sunday; 2026-09-09 is a Wednesday.
    store = synthetic_store(
        [make_row(fetched_date=AS_OF, depart_date=date(2026, 12, 20), amount_minor=42000)]
    )

    row = load_route_history("JFK-LHR", as_of=AS_OF, store=store).row(0, named=True)

    assert row["fetched_date"] == AS_OF
    assert row["days_to_departure"] == 102
    assert row["ap_bucket"] == "90+"
    assert row["travel_month"] == 12
    assert row["travel_dow"] == 6


def test_travel_month_dow_come_from_depart_not_fetched(synthetic_store) -> None:
    # Fetched on a Monday in August, flying on a Friday in October.
    fetched, depart = date(2026, 8, 31), date(2026, 10, 2)
    store = synthetic_store([make_row(fetched_date=fetched, depart_date=depart, amount_minor=1)])

    row = load_route_history("JFK-LHR", as_of=AS_OF, store=store).row(0, named=True)

    assert (row["travel_month"], row["travel_dow"]) == (10, 4)
    assert (fetched.month, fetched.weekday()) != (10, 4)


def test_history_window_respected(synthetic_store) -> None:
    window = load_baseline_config().history.window_days
    depart = AS_OF + timedelta(days=5)
    store = synthetic_store(
        [
            make_row(
                fetched_date=AS_OF - timedelta(days=window), depart_date=depart, amount_minor=1
            ),
            make_row(
                fetched_date=AS_OF - timedelta(days=window + 1), depart_date=depart, amount_minor=2
            ),
            make_row(fetched_date=AS_OF, depart_date=depart, amount_minor=3),
            make_row(fetched_date=AS_OF + timedelta(days=1), depart_date=depart, amount_minor=4),
        ]
    )

    history = load_route_history("JFK-LHR", as_of=AS_OF, store=store)

    assert sorted(history["amount_minor"].to_list()) == [1, 3]


def test_departed_rows_are_dropped(synthetic_store) -> None:
    store = synthetic_store(
        [make_row(fetched_date=AS_OF, depart_date=AS_OF - timedelta(days=1), amount_minor=1)]
    )

    assert load_route_history("JFK-LHR", as_of=AS_OF, store=store).is_empty()


def test_currency_mismatch_is_dropped_and_counted(synthetic_store) -> None:
    rows = [
        make_row(
            fetched_date=AS_OF - timedelta(days=i), depart_date=date(2026, 11, 1), amount_minor=1
        )
        for i in range(3)
    ]
    rows.append(
        make_row(fetched_date=AS_OF, depart_date=date(2026, 11, 2), amount_minor=1, currency="EUR")
    )
    store = synthetic_store(rows)

    history, dropped = load_route_history_with_stats("JFK-LHR", as_of=AS_OF, store=store)

    assert dropped == 1
    assert set(history["currency"].unique()) == {"USD"}


def test_frame_schema_is_frozen(fixture_history, synthetic_store) -> None:
    assert tuple(fixture_history("JFK-LHR").columns) == OBSERVATION_FRAME_COLUMNS

    empty = load_route_history("JFK-LHR", as_of=AS_OF, store=synthetic_store([]))
    assert empty.is_empty()
    assert tuple(empty.columns) == OBSERVATION_FRAME_COLUMNS


def test_frame_is_sorted_by_depart_then_fetched(fixture_history) -> None:
    history = fixture_history("ATL-MIA")

    assert history.equals(history.sort(["depart_date", "fetched_date"]))
    assert history["fetched_date"].max() == FIXTURE_AS_OF
