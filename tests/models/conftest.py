"""Stores for the model tests.

Every fixture-backed store is built the way SF-06's Done-when asks: fixture mode on, and a
snapshots root that does not exist, so nothing under ``data/snapshots/`` can leak in.
"""

from __future__ import annotations

import csv
import dataclasses
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, time
from pathlib import Path

import polars as pl
import pytest

from models.features.observations import load_route_history
from pipeline.schema import (
    Cabin,
    FareObservation,
    PriceKind,
    Source,
    TripType,
    build_observation,
)
from pipeline.store import SnapshotStore
from shared.settings import load_data_settings

#: The fixture's newest fetched_date, and CI's pinned SNAP_TODAY.
FIXTURE_AS_OF: date = date(2026, 9, 9)
RUN_ID: str = "33333333-3333-5333-8333-333333333333"


def fixture_route_keys() -> list[str]:
    settings = load_data_settings()
    with settings.routes_csv.open() as handle:
        return [row["route_key"] for row in csv.DictReader(handle)]


def make_row(
    *,
    fetched_date: date,
    depart_date: date,
    amount_minor: int,
    route_key: str = "JFK-LHR",
    price_kind: PriceKind = PriceKind.CALENDAR_CHEAPEST,
    currency: str = "USD",
    carrier_primary: str | None = None,
) -> FareObservation:
    """A schema-valid one-way economy row. Itinerary rows come from fastflights, calendar
    rows from travelpayouts, matching the fixture."""
    origin, destination = route_key.split("-")
    itinerary = price_kind == PriceKind.ITINERARY
    return build_observation(
        source=Source.FASTFLIGHTS if itinerary else Source.TRAVELPAYOUTS,
        fetched_at=datetime.combine(fetched_date, time(6, 0), tzinfo=UTC),
        origin=origin,
        destination=destination,
        depart_date=depart_date,
        trip_type=TripType.ONE_WAY,
        cabin=Cabin.ECONOMY,
        passengers=1,
        amount_minor=amount_minor,
        currency=currency,
        price_kind=price_kind,
        ingest_run_id=RUN_ID,
        stops_outbound=0 if itinerary else None,
        carrier_primary=carrier_primary if carrier_primary else ("BA" if itinerary else None),
    )


@pytest.fixture(scope="session")
def fixture_store(tmp_path_factory: pytest.TempPathFactory) -> SnapshotStore:
    """Fixture mode, with a snapshots root that is guaranteed absent."""
    settings = load_data_settings(use_fixtures=True)
    absent = tmp_path_factory.mktemp("no-snapshots") / "data" / "snapshots" / "fare_observations"
    return SnapshotStore(dataclasses.replace(settings, snapshots_root=absent))


@pytest.fixture(scope="session")
def fixture_history(fixture_store: SnapshotStore) -> Callable[[str], pl.DataFrame]:
    """Route history as of FIXTURE_AS_OF, read once per route for the whole session."""
    cache: dict[str, pl.DataFrame] = {}

    def get(route_key: str) -> pl.DataFrame:
        if route_key not in cache:
            cache[route_key] = load_route_history(
                route_key, as_of=FIXTURE_AS_OF, store=fixture_store
            )
        return cache[route_key]

    return get


@pytest.fixture
def synthetic_store(tmp_path: Path) -> Callable[[Iterable[FareObservation]], SnapshotStore]:
    """A small store with exactly the rows a test writes, fixtures off."""

    def build(rows: Iterable[FareObservation]) -> SnapshotStore:
        store = SnapshotStore(load_data_settings(repo_root=tmp_path, use_fixtures=False))
        batch = list(rows)
        if batch:
            store.write(batch)
        return store

    return build
