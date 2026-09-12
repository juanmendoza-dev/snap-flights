"""The baseline model's types (SF-06-build §4.1, amended by decision 0002 D3).

``current_price``, ``price_percentile``, ``expected_low`` and ``basis.from``/``basis.to`` are
nullable: missing data comes back as an honest ``neutral`` / ``low`` with a
``data_unavailable_reason``, never a fabricated price or a sentinel percentile.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipeline.schema import Cabin, TripType
from pipeline.schema.record import IataCode

_STRICT = ConfigDict(frozen=True, extra="forbid")
_CURRENCY = Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")


class Verdict(StrEnum):
    BOOK_NOW = "book_now"
    WAIT = "wait"
    NEUTRAL = "neutral"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PriceSource(StrEnum):
    USER_SUPPLIED = "user_supplied"
    STORE = "store"


class DataUnavailableReason(StrEnum):
    """Why a prediction could not be fully populated (0002 D3). Machine-readable; the
    human sentence is ``data_quality_note``."""

    NO_CURRENT_PRICE = "no_current_price"
    NO_ROUTE_HISTORY = "no_route_history"
    THIN_ROUTE_HISTORY = "thin_route_history"
    DEPARTURE_IN_PAST = "departure_in_past"


class Money(BaseModel):
    model_config = _STRICT
    amount_minor: int = Field(gt=0)
    currency: str = _CURRENCY


class TripShape(BaseModel):
    """The MVP trip shape (L0 §8): one-way, economy, 1 passenger. Field names and types
    are the canonical ones from L0 §3."""

    model_config = _STRICT
    origin: IataCode
    destination: IataCode
    depart_date: date
    trip_type: TripType = TripType.ONE_WAY
    cabin: Cabin = Cabin.ECONOMY
    passengers: int = Field(default=1, ge=1)
    return_date: date | None = None

    @model_validator(mode="after")
    def _shape_is_consistent(self) -> TripShape:
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
        if self.trip_type == TripType.ONE_WAY and self.return_date is not None:
            raise ValueError("a one_way trip has no return_date")
        if self.trip_type == TripType.ROUND_TRIP and self.return_date is None:
            raise ValueError("a round_trip needs a return_date")
        if self.return_date is not None and self.return_date < self.depart_date:
            raise ValueError("return_date is before depart_date")
        return self

    @property
    def route_key(self) -> str:
        return f"{self.origin}-{self.destination}"


class CurrentPrice(BaseModel):
    model_config = _STRICT
    amount_minor: int = Field(gt=0)
    currency: str = _CURRENCY
    source: PriceSource
    as_of: datetime

    @field_validator("as_of")
    @classmethod
    def _utc_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("as_of must be timezone-aware UTC")
        return value


class CurvePoint(BaseModel):
    model_config = _STRICT
    days_to_departure: int = Field(ge=0)
    amount_minor: int = Field(gt=0)


class CurveResult(BaseModel):
    """What ``expected_curve()`` returns (0002 D4, SF-06-build §6a). ``as_of`` and
    ``dtd_now`` are carried explicitly rather than inferred from the first surviving point,
    because a point can be omitted when its cell has no data."""

    model_config = _STRICT
    as_of: date
    depart_date: date
    dtd_now: int
    points: tuple[CurvePoint, ...] = ()

    def value_at(self, days_to_departure: int) -> int | None:
        for point in self.points:
            if point.days_to_departure == days_to_departure:
                return point.amount_minor
        return None


class ExpectedLow(BaseModel):
    model_config = _STRICT
    amount_minor: int = Field(gt=0)
    currency: str = _CURRENCY
    window_start: date
    window_end: date

    @model_validator(mode="after")
    def _window_ordered(self) -> ExpectedLow:
        if self.window_end < self.window_start:
            raise ValueError("window_end is before window_start")
        return self


class Basis(BaseModel):
    """Everything the numbers were computed from — the 'why' panel and the audit trail."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)
    #: Rows in the (route, AP BUCKET) cell — the sample price_percentile was ranked in, NOT
    #: the whole route (that is route_observations).
    observations: int = Field(ge=0)
    #: Rows for the whole route in the history window.
    route_observations: int = Field(ge=0)
    #: min / max fetched_date actually used; null when no rows were used (0002 D3).
    from_: date | None = Field(alias="from")
    to: date | None
    #: Distinct `source` values, sorted.
    sources: list[str]
    #: Null only for a departure in the past, which has no AP bucket (SF-06-build §6a).
    ap_bucket: str | None
    travel_month: int = Field(ge=1, le=12)
    travel_dow: int = Field(ge=0, le=6)
    cov: float | None
    dropped_currency_mismatch: int = 0


class Prediction(BaseModel):
    model_config = _STRICT
    trip_shape: TripShape
    current_price: CurrentPrice | None
    price_percentile: int | None = Field(ge=0, le=100)
    verdict: Verdict
    expected_curve: list[CurvePoint]
    expected_low: ExpectedLow | None
    confidence: Confidence
    reason: str
    basis: Basis
    data_quality_note: str | None = None
    data_unavailable_reason: DataUnavailableReason | None = None

    @model_validator(mode="after")
    def _invariants(self) -> Prediction:
        """The contract SF-07 relies on (SF-06-build §8), enforced at construction."""
        if (self.expected_low is not None) != (self.verdict == Verdict.WAIT):
            raise ValueError("expected_low must be set iff verdict is wait")
        if self.data_unavailable_reason is not None:
            if self.verdict != Verdict.NEUTRAL or self.confidence != Confidence.LOW:
                raise ValueError("unavailable data must be neutral / low")
            if self.data_quality_note is None:
                raise ValueError("unavailable data needs a data_quality_note")
            if self.price_percentile is not None:
                raise ValueError("unavailable data has no price_percentile")
        elif self.price_percentile is None or self.current_price is None:
            raise ValueError("a populated prediction needs a price and a percentile")
        return self
