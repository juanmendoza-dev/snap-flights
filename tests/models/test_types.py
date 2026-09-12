"""Prediction types and the invariants they enforce (SF-06-build §4.1, 0002 D3)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from models.baseline.types import (
    Basis,
    Confidence,
    CurrentPrice,
    CurvePoint,
    CurveResult,
    DataUnavailableReason,
    ExpectedLow,
    Prediction,
    PriceSource,
    TripShape,
    Verdict,
)

SHAPE = TripShape(origin="JFK", destination="LHR", depart_date=date(2026, 10, 1))
PRICE = CurrentPrice(
    amount_minor=42000,
    currency="USD",
    source=PriceSource.USER_SUPPLIED,
    as_of=datetime(2026, 9, 9, tzinfo=UTC),
)
EMPTY_BASIS = Basis(
    observations=0,
    route_observations=0,
    from_=None,
    to=None,
    sources=[],
    ap_bucket="22-30",
    travel_month=10,
    travel_dow=3,
    cov=None,
)
LOW = ExpectedLow(
    amount_minor=38000,
    currency="USD",
    window_start=date(2026, 9, 20),
    window_end=date(2026, 9, 22),
)


def _prediction(**overrides: object) -> Prediction:
    fields: dict[str, object] = {
        "trip_shape": SHAPE,
        "current_price": PRICE,
        "price_percentile": 50,
        "verdict": Verdict.NEUTRAL,
        "expected_curve": [],
        "expected_low": None,
        "confidence": Confidence.MEDIUM,
        "reason": "x",
        "basis": EMPTY_BASIS,
    }
    return Prediction(**(fields | overrides))


def test_trip_shape_route_key_and_malformed_shapes() -> None:
    assert SHAPE.route_key == "JFK-LHR"
    with pytest.raises(ValidationError):
        TripShape(origin="jfk", destination="LHR", depart_date=date(2026, 10, 1))
    with pytest.raises(ValidationError):
        TripShape(
            origin="JFK",
            destination="LHR",
            depart_date=date(2026, 10, 1),
            return_date=date(2026, 10, 9),
        )


def test_expected_low_iff_wait() -> None:
    _prediction(verdict=Verdict.WAIT, expected_low=LOW)
    with pytest.raises(ValidationError, match="iff"):
        _prediction(verdict=Verdict.WAIT, expected_low=None)
    with pytest.raises(ValidationError, match="iff"):
        _prediction(verdict=Verdict.NEUTRAL, expected_low=LOW)


def test_unavailable_data_is_neutral_low_with_a_note_and_no_percentile() -> None:
    ok = _prediction(
        current_price=None,
        price_percentile=None,
        confidence=Confidence.LOW,
        data_unavailable_reason=DataUnavailableReason.NO_CURRENT_PRICE,
        data_quality_note="No observed price for this route and date.",
    )
    assert ok.data_unavailable_reason == "no_current_price"

    with pytest.raises(ValidationError, match="no price_percentile"):
        _prediction(
            price_percentile=50,
            confidence=Confidence.LOW,
            data_unavailable_reason=DataUnavailableReason.NO_ROUTE_HISTORY,
            data_quality_note="x",
        )
    with pytest.raises(ValidationError, match="neutral / low"):
        _prediction(
            price_percentile=None,
            data_unavailable_reason=DataUnavailableReason.NO_ROUTE_HISTORY,
            data_quality_note="x",
        )
    with pytest.raises(ValidationError, match="needs a price"):
        _prediction(price_percentile=None)


def test_basis_serialises_from_by_alias() -> None:
    dumped = _prediction().model_dump(mode="json", by_alias=True)

    assert "from" in dumped["basis"]
    assert "from_" not in dumped["basis"]
    assert dumped["basis"]["from"] is None


def test_current_price_must_be_utc() -> None:
    with pytest.raises(ValidationError):
        CurrentPrice(
            amount_minor=1, currency="USD", source=PriceSource.STORE, as_of=datetime(2026, 9, 9)
        )


def test_expected_low_window_is_ordered() -> None:
    with pytest.raises(ValidationError):
        ExpectedLow(
            amount_minor=1,
            currency="USD",
            window_start=date(2026, 9, 22),
            window_end=date(2026, 9, 20),
        )


def test_curve_result_value_at() -> None:
    curve = CurveResult(
        as_of=date(2026, 9, 9),
        depart_date=date(2026, 9, 11),
        dtd_now=2,
        points=(
            CurvePoint(days_to_departure=2, amount_minor=10),
            CurvePoint(days_to_departure=0, amount_minor=9),
        ),
    )

    assert curve.value_at(2) == 10
    assert curve.value_at(1) is None
