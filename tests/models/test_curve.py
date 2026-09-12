"""The expected curve and expected low (SF-06-build §4.3 as amended by 0002 D4 / §6a)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from models.baseline.config import BaselineConfig, load_baseline_config
from models.baseline.curve import expected_curve, find_expected_low, wait_candidates
from models.baseline.types import CurvePoint, CurveResult, TripShape
from models.features.distributions import build_bucket_distribution, build_distribution
from tests.models.conftest import FIXTURE_AS_OF

CONFIG = load_baseline_config()

#: SF-03-build §4.4 — (route_key, fetched_date, dtd) of the 12 injected error fares.
ERROR_FARE_CELLS: tuple[tuple[str, date, int], ...] = (
    ("JFK-LHR", date(2026, 6, 20), 17),
    ("JFK-LHR", date(2026, 8, 2), 73),
    ("LHR-JFK", date(2026, 7, 11), 41),
    ("LAX-NRT", date(2026, 6, 28), 96),
    ("LAX-NRT", date(2026, 8, 19), 12),
    ("SFO-LHR", date(2026, 7, 3), 55),
    ("BOS-DUB", date(2026, 9, 1), 29),
    ("JFK-LAX", date(2026, 6, 15), 6),
    ("JFK-LAX", date(2026, 8, 25), 84),
    ("ORD-DEN", date(2026, 7, 22), 33),
    ("ATL-MIA", date(2026, 6, 30), 108),
    ("MAD-LIS", date(2026, 8, 8), 47),
)


def _config(**curve: int) -> BaselineConfig:
    return CONFIG.model_copy(update={"curve": CONFIG.curve.model_copy(update=curve)})


def _curve_for(history: pl.DataFrame, depart: date, *, as_of: date = FIXTURE_AS_OF, config=CONFIG):
    route_key = history["route_key"][0]
    origin, destination = route_key.split("-")
    return expected_curve(
        trip_shape=TripShape(origin=origin, destination=destination, depart_date=depart),
        as_of=as_of,
        distribution=build_distribution(history),
        bucket_distribution=build_bucket_distribution(history),
        config=config,
    )


def _hand_curve(
    values: list[int], *, dtd_now: int, depart: date = date(2026, 12, 1)
) -> CurveResult:
    """A curve from explicit values, listed from dtd_now downwards."""
    return CurveResult(
        as_of=depart - timedelta(days=dtd_now),
        depart_date=depart,
        dtd_now=dtd_now,
        points=tuple(
            CurvePoint(days_to_departure=dtd_now - i, amount_minor=v) for i, v in enumerate(values)
        ),
    )


def test_curve_length_short_horizon(fixture_history) -> None:
    curve = _curve_for(fixture_history("JFK-LHR"), FIXTURE_AS_OF + timedelta(days=30))

    assert len(curve.points) == 31
    assert curve.points[-1].days_to_departure == 0


def test_curve_length_long_horizon(fixture_history) -> None:
    curve = _curve_for(fixture_history("JFK-LHR"), FIXTURE_AS_OF + timedelta(days=200))

    assert len(curve.points) == 91
    assert curve.points[-1].days_to_departure == 110


def test_curve_is_descending_in_dtd(fixture_history) -> None:
    curve = _curve_for(fixture_history("LAX-NRT"), FIXTURE_AS_OF + timedelta(days=45))
    dtds = [p.days_to_departure for p in curve.points]

    assert dtds[0] == curve.dtd_now == 45
    assert curve.as_of == FIXTURE_AS_OF
    assert dtds == sorted(dtds, reverse=True)
    assert dtds == list(range(45, -1, -1))


def test_past_departure_is_empty_but_carries_dtd_now(fixture_history) -> None:
    curve = _curve_for(fixture_history("JFK-LHR"), FIXTURE_AS_OF - timedelta(days=2))

    assert curve.points == ()
    assert curve.dtd_now == -2


def _distribution(rows: list[tuple[str, int, int, int, float]]) -> pl.DataFrame:
    """(ap_bucket, month, dow, count, median) rows for JFK-LHR."""
    return pl.DataFrame(
        {
            "route_key": ["JFK-LHR"] * len(rows),
            "ap_bucket": [r[0] for r in rows],
            "travel_month": [r[1] for r in rows],
            "travel_dow": [r[2] for r in rows],
            "count": [r[3] for r in rows],
            "median": [r[4] for r in rows],
        }
    )


def _buckets(rows: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "route_key": ["JFK-LHR"] * len(rows),
            "ap_bucket": list(rows),
            "count": [1000] * len(rows),
            "median": list(rows.values()),
        }
    )


# 2026-10-01 is a Thursday (dow 3); as of 2026-09-29, dtd_now = 2, every point in "0-3".
DEPART = date(2026, 10, 1)
SHAPE = TripShape(origin="JFK", destination="LHR", depart_date=DEPART)


def test_month_and_dow_constant_across_curve() -> None:
    distribution = _distribution(
        [("0-3", 10, 3, 50, 50000.0), ("0-3", 10, 4, 50, 10.0), ("0-3", 11, 3, 50, 10.0)]
    )

    curve = expected_curve(
        trip_shape=SHAPE,
        as_of=date(2026, 9, 29),
        distribution=distribution,
        bucket_distribution=_buckets({"0-3": 20.0}),
        config=CONFIG,
    )

    assert [p.amount_minor for p in curve.points] == [50000, 50000, 50000]


def test_falls_back_to_bucket_median_on_thin_cell() -> None:
    thin = CONFIG.curve.min_cell_observations - 1
    distribution = _distribution([("0-3", 10, 3, thin, 50000.0)])

    curve = expected_curve(
        trip_shape=SHAPE,
        as_of=date(2026, 9, 29),
        distribution=distribution,
        bucket_distribution=_buckets({"0-3": 30000.0}),
        config=CONFIG,
    )

    assert [p.amount_minor for p in curve.points] == [30000, 30000, 30000]


def test_point_is_omitted_when_neither_cell_nor_bucket_exists() -> None:
    # as of 2026-09-25, dtd_now = 6: dtds 4..6 are "4-7" (no data), 0..3 are "0-3".
    curve = expected_curve(
        trip_shape=SHAPE,
        as_of=date(2026, 9, 25),
        distribution=_distribution([]),
        bucket_distribution=_buckets({"0-3": 30000.0}),
        config=CONFIG,
    )

    assert curve.dtd_now == 6
    assert [p.days_to_departure for p in curve.points] == [3, 2, 1, 0]
    assert curve.value_at(curve.dtd_now) is None


def test_smoothing_removes_the_staircase(fixture_history) -> None:
    history = fixture_history("SFO-LHR")
    depart = FIXTURE_AS_OF + timedelta(days=100)

    raw = _curve_for(history, depart, config=_config(smoothing_window_days=1))
    smoothed = _curve_for(history, depart)

    assert len({p.amount_minor for p in raw.points}) <= 9
    assert len({p.amount_minor for p in smoothed.points}) > len(
        {p.amount_minor for p in raw.points}
    )
    # Shrinking windows keep both endpoints.
    assert smoothed.points[0].days_to_departure == raw.points[0].days_to_departure == 100
    assert smoothed.points[-1].days_to_departure == raw.points[-1].days_to_departure == 10


def test_error_fares_do_not_move_expected_low(fixture_history) -> None:
    """The curve is built from medians, never minima: however deep an error fare is, it
    cannot move the curve or the expected low (SF-03-build §4.4)."""
    for route_key, fetched, dtd in ERROR_FARE_CELLS:
        history = fixture_history(route_key)
        depart = fetched + timedelta(days=dtd)
        is_outlier = (
            (pl.col("fetched_date") == fetched)
            & (pl.col("depart_date") == depart)
            & (pl.col("price_kind") == "calendar_cheapest")
        )
        deeper = history.with_columns(
            pl.when(is_outlier)
            .then(pl.lit(100, pl.Int64))
            .otherwise("amount_minor")
            .alias("amount_minor")
        )

        original = _curve_for(history, depart, as_of=fetched)
        with_deeper_outlier = _curve_for(deeper, depart, as_of=fetched)

        assert original == with_deeper_outlier, route_key
        assert find_expected_low(original, config=CONFIG, currency="USD") == find_expected_low(
            with_deeper_outlier, config=CONFIG, currency="USD"
        )


def test_expected_low_window_is_a_contiguous_calendar_range() -> None:
    #          dtd: 10     9      8     7     6     5     4      3      2      1      0
    curve = _hand_curve(
        [10000, 10000, 9000, 8000, 8050, 8070, 9500, 10000, 10000, 10000, 10000], dtd_now=10
    )

    low = find_expected_low(curve, config=CONFIG, currency="USD")

    assert low is not None
    assert low.amount_minor == 8000
    assert low.window_start <= low.window_end
    # dtds 7, 6, 5 are within 1% of 8000 -> calendar depart-7 .. depart-5.
    assert low.window_start == curve.depart_date - timedelta(days=7)
    assert low.window_end == curve.depart_date - timedelta(days=5)


def test_expected_low_none_only_when_no_future_point_clears_the_drop() -> None:
    # 0002 D4: the old "minimum at dtd_now still returns an ExpectedLow" is gone — a curve
    # that only rises from here has no eligible point, and neither does dtd_now == 0.
    rising = _hand_curve([8000, 9000, 10000], dtd_now=2)
    at_departure = _hand_curve([8000], dtd_now=0)
    shallow = _hand_curve([10000, 9500, 9400], dtd_now=2)
    dipping = _hand_curve([10000, 9000, 9500], dtd_now=2)

    assert find_expected_low(rising, config=CONFIG, currency="USD") is None
    assert find_expected_low(at_departure, config=CONFIG, currency="USD") is None
    assert find_expected_low(shallow, config=CONFIG, currency="USD") is None
    assert find_expected_low(dipping, config=CONFIG, currency="USD") is not None


def test_wait_candidates_are_strictly_after_as_of_and_within_horizon() -> None:
    horizon = CONFIG.verdict.wait.search_horizon_days
    dtd_now = horizon + 5
    values = [10000] + [5000] * dtd_now  # every future point is a deep dip
    curve = _hand_curve(values, dtd_now=dtd_now)

    dtds = {p.days_to_departure for p in wait_candidates(curve, CONFIG)}

    assert max(dtds) == dtd_now - 1
    assert min(dtds) == dtd_now - horizon


def test_drop_threshold_is_inclusive_and_exact() -> None:
    drop = CONFIG.verdict.wait.min_curve_drop_pct
    on_threshold = int(10000 * (100 - drop) / 100)

    assert wait_candidates(_hand_curve([10000, on_threshold], dtd_now=1), CONFIG)
    assert not wait_candidates(_hand_curve([10000, on_threshold + 1], dtd_now=1), CONFIG)


def test_expected_low_window_is_the_run_containing_the_argmin() -> None:
    # Two troughs inside the 1% band; the deeper one (7900 at dtd 3) owns the window.
    #          dtd: 9      8     7     6      5      4     3     2      1      0
    curve = _hand_curve([10000, 7950, 7960, 9500, 9500, 7950, 7900, 7940, 9500, 9500], dtd_now=9)

    low = find_expected_low(curve, config=CONFIG, currency="USD")

    assert low is not None
    assert low.amount_minor == 7900
    assert low.window_start == curve.depart_date - timedelta(days=4)
    assert low.window_end == curve.depart_date - timedelta(days=2)


def test_expected_low_tie_prefers_the_earliest_calendar_date() -> None:
    curve = _hand_curve([10000, 8000, 9500, 8000], dtd_now=3)

    low = find_expected_low(curve, config=CONFIG, currency="USD")

    assert low is not None
    assert low.window_start == curve.depart_date - timedelta(days=2)
    assert low.window_end == curve.depart_date - timedelta(days=2)
