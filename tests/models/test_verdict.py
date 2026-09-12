"""Verdict, confidence and reason (SF-06-build §4.4 as amended by 0002 D4)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from models.baseline.config import BaselineConfig, load_baseline_config
from models.baseline.curve import find_expected_low
from models.baseline.types import Basis, Confidence, CurvePoint, CurveResult, TripShape, Verdict
from models.baseline.verdict import build_reason, decide_confidence, decide_verdict
from models.features.distributions import build_bucket_distribution
from tests.models.conftest import fixture_route_keys

CONFIG = load_baseline_config()
BOOK = CONFIG.verdict.book_now
WAIT = CONFIG.verdict.wait
DEPART = date(2026, 12, 1)


def _curve(values: list[int], *, dtd_now: int | None = None) -> CurveResult:
    """Values listed from dtd_now downwards."""
    now = len(values) - 1 if dtd_now is None else dtd_now
    return CurveResult(
        as_of=DEPART - timedelta(days=now),
        depart_date=DEPART,
        dtd_now=now,
        points=tuple(
            CurvePoint(days_to_departure=now - i, amount_minor=v) for i, v in enumerate(values)
        ),
    )


def _rising(pct: float) -> CurveResult:
    return _curve([10000, 10000, int(10000 * (100 + pct) / 100)])


def _dipping(pct: float) -> CurveResult:
    return _curve([10000, int(10000 * (100 - pct) / 100), 10000])


FLAT = _curve([10000, 10000, 10000])


def _verdict(percentile: int | None, curve: CurveResult, config: BaselineConfig = CONFIG):
    return decide_verdict(price_percentile=percentile, curve=curve, config=config)


def test_book_now_rule() -> None:
    rise = BOOK.min_curve_rise_pct
    cutoff = BOOK.max_percentile

    assert _verdict(cutoff, _rising(rise)) == Verdict.BOOK_NOW
    assert _verdict(cutoff - 1, _rising(rise)) == Verdict.BOOK_NOW
    assert _verdict(cutoff + 1, _rising(rise)) == Verdict.NEUTRAL
    assert _verdict(cutoff, _curve([10000, 10000, int(10000 * (100 + rise) / 100) - 1])) == (
        Verdict.NEUTRAL
    )


def test_book_now_rise_is_measured_from_the_current_point() -> None:
    # The curve is already at its peak: it falls from here, so there is no rise to beat,
    # however cheap the quote is against history (0002 D4, review P2).
    falling = _curve([12000, 11000, 10000])

    assert _verdict(0, falling) == Verdict.NEUTRAL


def test_wait_rule() -> None:
    drop = WAIT.min_curve_drop_pct
    cutoff = WAIT.min_percentile

    assert _verdict(cutoff, _dipping(drop)) == Verdict.WAIT
    assert _verdict(cutoff + 1, _dipping(drop)) == Verdict.WAIT
    assert _verdict(cutoff - 1, _dipping(drop)) == Verdict.NEUTRAL
    assert _verdict(100, _curve([10000, int(10000 * (100 - drop) / 100) + 1, 10000])) == (
        Verdict.NEUTRAL
    )


def test_wait_needs_the_curve_to_dip_not_just_a_dear_quote() -> None:
    # A quote at percentile 100 on a flat curve used to read as "wait": the difference
    # between the quote and the curve is a price level, not a predicted move (review P2).
    assert _verdict(100, FLAT) == Verdict.NEUTRAL


def test_wait_ignores_a_dip_beyond_the_search_horizon() -> None:
    dtd_now = WAIT.search_horizon_days + 3
    values = [10000] * (dtd_now + 1)
    values[-1] = 5000  # a deep dip at dtd 0, which is dtd_now - horizon - 3
    assert _verdict(100, _curve(values)) == Verdict.NEUTRAL

    values = [10000] * (dtd_now + 1)
    values[WAIT.search_horizon_days] = 5000  # exactly dtd_now - horizon: eligible
    assert _verdict(100, _curve(values)) == Verdict.WAIT


def test_neutral_when_neither() -> None:
    assert _verdict(40, _rising(50)) == Verdict.NEUTRAL
    assert _verdict(40, _dipping(50)) == Verdict.NEUTRAL
    assert _verdict(10, FLAT) == Verdict.NEUTRAL
    assert _verdict(90, FLAT) == Verdict.NEUTRAL


def test_neutral_on_empty_curve() -> None:
    empty = CurveResult(as_of=DEPART, depart_date=DEPART, dtd_now=5)

    assert _verdict(0, empty) == Verdict.NEUTRAL
    assert _verdict(100, empty) == Verdict.NEUTRAL


def test_neutral_at_departure_and_without_a_percentile_or_current_point() -> None:
    assert _verdict(0, _curve([10000], dtd_now=0)) == Verdict.NEUTRAL
    assert _verdict(None, _dipping(50)) == Verdict.NEUTRAL
    gap_at_now = CurveResult(
        as_of=DEPART - timedelta(days=3),
        depart_date=DEPART,
        dtd_now=3,
        points=(CurvePoint(days_to_departure=2, amount_minor=20000),
                CurvePoint(days_to_departure=1, amount_minor=1000)),
    )  # fmt: skip
    assert _verdict(100, gap_at_now) == Verdict.NEUTRAL
    assert _verdict(0, gap_at_now) == Verdict.NEUTRAL


def test_lowering_book_now_threshold_changes_verdict() -> None:
    stricter = CONFIG.model_copy(
        update={
            "verdict": CONFIG.verdict.model_copy(
                update={"book_now": BOOK.model_copy(update={"max_percentile": 10})}
            )
        }
    )

    assert _verdict(20, _rising(20)) == Verdict.BOOK_NOW
    assert _verdict(20, _rising(20), stricter) == Verdict.NEUTRAL


@pytest.mark.parametrize(
    ("observations", "cov", "expected"),
    [
        (99, 0.1, Confidence.LOW),
        (270, 0.1, Confidence.MEDIUM),
        (600, 0.1, Confidence.HIGH),
        (600, 0.4, Confidence.LOW),
        (600, None, Confidence.MEDIUM),
        (99, None, Confidence.LOW),
    ],
)
def test_confidence_bands(observations: int, cov: float | None, expected: Confidence) -> None:
    assert decide_confidence(cell_observations=observations, cov=cov, config=CONFIG) == expected


def test_raising_confidence_floor_changes_confidence() -> None:
    raised = CONFIG.model_copy(
        update={
            "confidence": CONFIG.confidence.model_copy(
                update={"high": CONFIG.confidence.high.model_copy(update={"min_observations": 700})}
            )
        }
    )

    assert decide_confidence(cell_observations=600, cov=0.1, config=CONFIG) == Confidence.HIGH
    assert decide_confidence(cell_observations=600, cov=0.1, config=raised) == Confidence.MEDIUM


def test_fixture_short_buckets_cap_at_medium(fixture_history) -> None:
    for route_key in fixture_route_keys():
        cells = build_bucket_distribution(fixture_history(route_key))
        for row in cells.filter(cells["ap_bucket"].is_in(["0-3", "4-7"])).iter_rows(named=True):
            confidence = decide_confidence(
                cell_observations=row["count"], cov=row["cov"], config=CONFIG
            )
            assert confidence != Confidence.HIGH, (route_key, row["ap_bucket"])


SHAPE = TripShape(origin="JFK", destination="LHR", depart_date=DEPART)
BASIS = Basis(
    observations=630,
    route_observations=10800,
    from_=date(2026, 6, 12),
    to=date(2026, 9, 9),
    sources=["travelpayouts"],
    ap_bucket="0-3",
    travel_month=12,
    travel_dow=1,
    cov=0.12,
)


def _reason(verdict: Verdict, percentile: int | None, curve: CurveResult, **extra) -> str:
    low = (
        find_expected_low(curve, config=CONFIG, currency="USD") if verdict == Verdict.WAIT else None
    )
    return build_reason(
        verdict=verdict,
        price_percentile=percentile,
        expected_low=low,
        curve=curve,
        basis=BASIS,
        trip_shape=SHAPE,
        config=CONFIG,
        **extra,
    )


HISTORY = (
    "This fare is cheaper than 80% of the 630 observations we have for JFK-LHR booked 0-3 days out"
)


def test_reason_templates_describe_the_curve_move() -> None:
    # A 12% rise is reported as 12%, not as the 5% threshold it cleared.
    assert _reason(Verdict.BOOK_NOW, 20, _curve([10000, 10500, 11200])) == (
        f"{HISTORY}, and prices on this route usually rise about 12% from here. Book now."
    )
    # 2026-12-01 minus dtd 1 is 30 Nov.
    assert _reason(Verdict.WAIT, 20, _curve([10000, 8800, 9900])) == (
        f"{HISTORY}, but prices on this route usually dip about 12% around 30 Nov-30 Nov. "
        "Waiting looks better."
    )
    assert _reason(Verdict.NEUTRAL, 20, FLAT) == (
        f"{HISTORY}, and we do not see a clear move either way in the next 2 days."
    )
    assert _reason(Verdict.NEUTRAL, 20, _curve([10000], dtd_now=0)) == (
        f"{HISTORY}, and the flight leaves today."
    )


def test_reason_without_a_percentile_never_states_one() -> None:
    text = _reason(Verdict.NEUTRAL, None, FLAT, data_quality_note="No observed price.")

    assert text == "We can't call this fare yet. No observed price."
    assert "%" not in text


def test_reason_is_deterministic() -> None:
    curve = _curve([10000, 8800, 9900])

    assert _reason(Verdict.WAIT, 70, curve) == _reason(Verdict.WAIT, 70, curve)
