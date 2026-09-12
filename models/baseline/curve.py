"""The expected price curve and the expected low (SF-06-build §4.3, decision 0002 D4, §6a).

``depart_date`` is fixed by the request, so ``travel_month`` and ``travel_dow`` are constant
across the whole curve. Only ``days_to_departure`` varies, and through it ``ap_bucket``.

Threshold comparisons are done in scaled arithmetic (``value * 100`` against
``now * (100 - pct)``), not ``now * (1 - pct / 100)``: the latter is not exact in binary
floating point and would send a point sitting exactly on the threshold the wrong way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from models.baseline.config import BaselineConfig
from models.baseline.percentile import round_half_up_float
from models.baseline.types import CurvePoint, CurveResult, ExpectedLow, TripShape
from models.features.buckets import ap_bucket

#: The expected-low window is every consecutive day within this many percent of the minimum
#: (SF-06-build §4.3 pins 1%; it is a presentation rule, not a verdict threshold).
EXPECTED_LOW_BAND_PCT: int = 1


@dataclass(frozen=True, slots=True)
class CurveIndex:
    """One route's distributions as plain lookups, built once per history so a caller that
    asks for many curves (the backtest) does not re-filter the frames each time."""

    route_key: str
    #: (ap_bucket, travel_month, travel_dow) -> (count, median)
    cells: dict[tuple[str, int, int], tuple[int, float]]
    #: ap_bucket -> median
    buckets: dict[str, float]

    @classmethod
    def build(
        cls, route_key: str, distribution: pl.DataFrame, bucket_distribution: pl.DataFrame
    ) -> CurveIndex:
        cells = {
            (row["ap_bucket"], row["travel_month"], row["travel_dow"]): (
                row["count"],
                row["median"],
            )
            for row in distribution.filter(pl.col("route_key") == route_key).iter_rows(named=True)
        }
        buckets = {
            row["ap_bucket"]: row["median"]
            for row in bucket_distribution.filter(pl.col("route_key") == route_key).iter_rows(
                named=True
            )
        }
        return cls(route_key=route_key, cells=cells, buckets=buckets)


def _smooth(values: Sequence[float], window: int) -> list[float]:
    """Centred rolling mean with shrinking windows at both ends (min_periods=1)."""
    if window <= 1:
        return list(values)
    left, right = (window - 1) // 2, window // 2
    out: list[float] = []
    for i in range(len(values)):
        chunk = values[max(0, i - left) : i + right + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def curve_from_index(
    index: CurveIndex, *, trip_shape: TripShape, as_of: date, config: BaselineConfig
) -> CurveResult:
    """``expected_curve`` over a prebuilt index. Same semantics, see there."""
    dtd_now = (trip_shape.depart_date - as_of).days
    empty = CurveResult(as_of=as_of, depart_date=trip_shape.depart_date, dtd_now=dtd_now)
    if dtd_now < 0:
        return empty

    month, dow = trip_shape.depart_date.month, trip_shape.depart_date.weekday()
    min_cell = config.curve.min_cell_observations
    raw_dtds: list[int] = []
    raw_values: list[float] = []
    for dtd in range(max(0, dtd_now - config.curve.horizon_days), dtd_now + 1):
        bucket = ap_bucket(dtd, config)
        cell = index.cells.get((bucket, month, dow))
        if cell is not None and cell[0] >= min_cell:
            value = cell[1]
        elif bucket in index.buckets:
            value = index.buckets[bucket]
        else:
            continue  # no data for this bucket at all: the point is omitted
        raw_dtds.append(dtd)
        raw_values.append(value)

    if not raw_values:
        return empty
    smoothed = _smooth(raw_values, config.curve.smoothing_window_days)
    points = tuple(
        CurvePoint(days_to_departure=dtd, amount_minor=max(1, round_half_up_float(value)))
        for dtd, value in zip(reversed(raw_dtds), reversed(smoothed), strict=True)
    )
    return CurveResult(
        as_of=as_of, depart_date=trip_shape.depart_date, dtd_now=dtd_now, points=points
    )


def expected_curve(
    *,
    trip_shape: TripShape,
    as_of: date,
    distribution: pl.DataFrame,
    bucket_distribution: pl.DataFrame,
    config: BaselineConfig,
) -> CurveResult:
    """The expected cheapest price for each remaining day before departure.

    1. ``dtd_now = (depart_date - as_of).days``; a past departure gives no points.
    2. Points cover ``days_to_departure`` from ``dtd_now`` down to
       ``max(0, dtd_now - curve.horizon_days)``, inclusive, descending.
    3. Raw value: the (route, ap_bucket(dtd), travel_month, travel_dow) median; a cell with
       fewer than ``curve.min_cell_observations`` rows (or none) falls back to the
       (route, ap_bucket) median; with neither, the point is omitted.
    4. The raw series is a staircase of at most one value per bucket; a centred rolling mean
       of ``curve.smoothing_window_days`` (shrinking at the ends) blends it into ramps.
       Values are rounded half-up to int minor units at this single point.
    """
    index = CurveIndex.build(trip_shape.route_key, distribution, bucket_distribution)
    return curve_from_index(index, trip_shape=trip_shape, as_of=as_of, config=config)


def wait_candidates(curve: CurveResult, config: BaselineConfig) -> list[CurvePoint]:
    """The points that satisfy ``wait`` condition 2 (0002 D4): eligible future points —
    ``days_to_departure`` in ``[max(0, dtd_now - wait.search_horizon_days), dtd_now - 1]``,
    strictly after ``as_of`` — whose value is at least ``wait.min_curve_drop_pct`` percent
    below the curve's own value at ``dtd_now``.

    Empty when ``dtd_now <= 0`` or the curve has no point at ``dtd_now``. This is the one
    definition of condition 2; ``decide_verdict`` and ``find_expected_low`` both use it.
    """
    now = curve.value_at(curve.dtd_now)
    if curve.dtd_now <= 0 or now is None:
        return []
    wait = config.verdict.wait
    lowest = max(0, curve.dtd_now - wait.search_horizon_days)
    ceiling = now * (100 - wait.min_curve_drop_pct)
    return [
        point
        for point in curve.points
        if lowest <= point.days_to_departure <= curve.dtd_now - 1
        and point.amount_minor * 100 <= ceiling
    ]


def find_expected_low(
    curve: CurveResult, *, config: BaselineConfig, currency: str
) -> ExpectedLow | None:
    """The expected low among ``wait_candidates`` — None iff there are none.

    It does not decide whether an expected low is shown: ``predict()`` calls it only for a
    ``wait`` verdict, which is the single decision point for ``expected_low``.

    ``amount_minor`` is the minimum candidate value. The window is the run of consecutive
    ``days_to_departure`` values, among the candidates, that contains the argmin and stays
    within 1% of the minimum; on a tie the argmin with the largest ``days_to_departure``
    (earliest calendar date) wins. Calendar dates map back as
    ``depart_date - days_to_departure``, so ``window_start <= window_end`` always.
    """
    candidates = wait_candidates(curve, config)
    if not candidates:
        return None
    minimum = min(point.amount_minor for point in candidates)
    # curve.points is descending in dtd, so the first minimum has the largest dtd.
    argmin = next(point for point in candidates if point.amount_minor == minimum)
    in_band = {
        point.days_to_departure
        for point in candidates
        if point.amount_minor * 100 <= minimum * (100 + EXPECTED_LOW_BAND_PCT)
    }
    high = low = argmin.days_to_departure
    while high + 1 in in_band:
        high += 1
    while low - 1 in in_band:
        low -= 1
    return ExpectedLow(
        amount_minor=minimum,
        currency=currency,
        window_start=curve.depart_date - timedelta(days=high),
        window_end=curve.depart_date - timedelta(days=low),
    )
