"""Verdict, confidence and the plain-language reason (SF-06-build §4.4, decision 0002 D4).

Every threshold comes from ``config/baseline.yaml``; none is a literal here (a test greps this
file for them).
"""

from __future__ import annotations

from datetime import date

from models.baseline.config import BaselineConfig
from models.baseline.curve import wait_candidates
from models.baseline.percentile import round_half_up_float
from models.baseline.types import Basis, Confidence, CurveResult, ExpectedLow, TripShape, Verdict


def curve_rise(curve: CurveResult, config: BaselineConfig) -> tuple[int, int] | None:
    """``(now, peak)``: the curve's value at ``dtd_now`` and its maximum over
    ``[max(0, dtd_now - curve.horizon_days), dtd_now]``. None when there is no current point."""
    now = curve.value_at(curve.dtd_now)
    if now is None:
        return None
    lowest = max(0, curve.dtd_now - config.curve.horizon_days)
    peak = max(
        point.amount_minor
        for point in curve.points
        if lowest <= point.days_to_departure <= curve.dtd_now
    )
    return now, peak


def decide_verdict(
    *, price_percentile: int | None, curve: CurveResult, config: BaselineConfig
) -> Verdict:
    """Decision 0002 D4, every comparison pinned. ``now`` is the curve's own value at
    ``dtd_now`` — never the quote.

    book_now  price_percentile <= verdict.book_now.max_percentile
              AND max(curve over [max(0, dtd_now - curve.horizon_days), dtd_now])
                  >= now * (1 + book_now.min_curve_rise_pct / 100)

    wait      price_percentile >= verdict.wait.min_percentile
              AND some eligible future point (dtd in
                  [max(0, dtd_now - wait.search_horizon_days), dtd_now - 1])
                  <= now * (1 - wait.min_curve_drop_pct / 100)       (see wait_candidates)

    neutral   otherwise, and unconditionally when there is no percentile, the curve is
              empty, it has no point at dtd_now, or dtd_now == 0.

    The two percentile conditions are disjoint for any sane config, so the order in which
    they are checked is documentation, not tie-breaking.
    """
    if price_percentile is None or not curve.points or curve.dtd_now <= 0:
        return Verdict.NEUTRAL
    rise = curve_rise(curve, config)
    if rise is None:
        return Verdict.NEUTRAL
    now, peak = rise

    book_now = config.verdict.book_now
    if price_percentile <= book_now.max_percentile and peak * 100 >= now * (
        100 + book_now.min_curve_rise_pct
    ):
        return Verdict.BOOK_NOW
    if price_percentile >= config.verdict.wait.min_percentile and wait_candidates(curve, config):
        return Verdict.WAIT
    return Verdict.NEUTRAL


def decide_confidence(
    *, cell_observations: int, cov: float | None, config: BaselineConfig
) -> Confidence:
    """low   if cell_observations < confidence.low.max_observations
             OR (cov is not None AND cov > confidence.low.max_cov)
    high  if cell_observations >= confidence.high.min_observations
             AND cov is not None AND cov < confidence.high.max_cov
    medium otherwise.
    `low` is checked first: SF-06 lists it first and it is the safe direction."""
    low, high = config.confidence.low, config.confidence.high
    if cell_observations < low.max_observations or (cov is not None and cov > low.max_cov):
        return Confidence.LOW
    if cell_observations >= high.min_observations and cov is not None and cov < high.max_cov:
        return Confidence.HIGH
    return Confidence.MEDIUM


def _day(value: date) -> str:
    return f"{value.day} {value:%b}"


def _percent(numerator: int, denominator: int) -> int:
    return round_half_up_float(100 * numerator / denominator)


def build_reason(
    *,
    verdict: Verdict,
    price_percentile: int | None,
    expected_low: ExpectedLow | None,
    curve: CurveResult,
    basis: Basis,
    trip_shape: TripShape,
    config: BaselineConfig,
    data_quality_note: str | None = None,
) -> str:
    """A plain-language sentence assembled from the numbers. No LLM, no randomness — same
    inputs, same string.

    The history clause is only ever a statement about the past ("cheaper than X% of the n
    observations we have"); the forecast clause describes the curve's own move from its
    current point, with the realised percentage rather than the config threshold (0002 D4).
    The sentence never claims a period that ``basis`` does not cover.

    book_now:
      "This fare is cheaper than {100 - p}% of the {n} observations we have for {route}
       booked {bucket} days out, and prices on this route usually rise about {rise}% from
       here. Book now."
    wait:
      "This fare is cheaper than {100 - p}% of the {n} observations we have for {route}
       booked {bucket} days out, but prices on this route usually dip about {drop}% around
       {window_start}-{window_end}. Waiting looks better."
    neutral:
      "This fare is cheaper than {100 - p}% of the {n} observations we have for {route}
       booked {bucket} days out, and we do not see a clear move either way in the next
       {days} days."  — or "..., and the flight leaves today." when dtd_now is 0.
    no percentile (unavailable data):
      "We can't call this fare yet. {data_quality_note}"
    """
    if price_percentile is None:
        return f"We can't call this fare yet. {data_quality_note or ''}".rstrip()

    history = (
        f"This fare is cheaper than {100 - price_percentile}% of the {basis.observations} "
        f"observations we have for {trip_shape.route_key} booked {basis.ap_bucket} days out"
    )
    now = curve.value_at(curve.dtd_now)

    if verdict == Verdict.BOOK_NOW:
        rise = curve_rise(curve, config)
        if rise is None:
            raise ValueError("a book_now verdict needs a current curve point")
        start, peak = rise
        return (
            f"{history}, and prices on this route usually rise about "
            f"{_percent(peak - start, start)}% from here. Book now."
        )
    if verdict == Verdict.WAIT:
        if expected_low is None or now is None:
            raise ValueError("a wait verdict needs an expected low and a current curve point")
        drop = _percent(now - expected_low.amount_minor, now)
        return (
            f"{history}, but prices on this route usually dip about {drop}% around "
            f"{_day(expected_low.window_start)}-{_day(expected_low.window_end)}. "
            "Waiting looks better."
        )
    if curve.dtd_now == 0:
        return f"{history}, and the flight leaves today."
    days = min(curve.dtd_now, config.verdict.wait.search_horizon_days)
    return f"{history}, and we do not see a clear move either way in the next {days} days."
