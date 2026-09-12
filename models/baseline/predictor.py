"""``predict()`` — SF-06's entry point (SF-06-build §4.5, decision 0002 D3/D4, §6a).

``predict()`` reads the clock once, loads one route's history once, and hands both to
``predict_from_context()``, which is pure. The backtest builds a ``RouteContext`` per
``(route, as_of)`` and reuses it across every departure date it scores, so the two paths
cannot disagree about what a prediction is.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time

import polars as pl

from models.baseline.config import BaselineConfig, load_baseline_config
from models.baseline.curve import CurveIndex, curve_from_index, find_expected_low
from models.baseline.percentile import percentile_of_sorted
from models.baseline.types import (
    Basis,
    Confidence,
    CurrentPrice,
    CurveResult,
    DataUnavailableReason,
    Money,
    Prediction,
    PriceSource,
    TripShape,
    Verdict,
)
from models.baseline.verdict import build_reason, decide_confidence, decide_verdict
from models.features.buckets import ap_bucket
from models.features.distributions import build_bucket_distribution, build_distribution
from models.features.observations import collapse_observations, load_route_history_with_stats
from pipeline.store import ReadFilters, SnapshotStore, default_store
from shared.clock import today_utc

NOTE_DEPARTURE_IN_PAST = "This departure date is already in the past."
NOTE_NO_ROUTE_HISTORY = "No price history for {route_key} yet."
NOTE_NO_HISTORY_IN_CURRENCY = "No price history for {route_key} in {currency}."
NOTE_NO_CURRENT_PRICE = "No observed price for this route and date."
NOTE_THIN = "Only {n} observations for this route at this booking window; not enough to call it."


@dataclass(frozen=True, slots=True)
class RouteContext:
    """Everything ``predict_from_context`` needs from one route's history."""

    route_key: str
    history: pl.DataFrame
    dropped_currency_mismatch: int
    currency: str | None
    #: ap_bucket -> amounts in that (route, AP bucket) cell, sorted ascending
    bucket_samples: dict[str, list[int]]
    #: ap_bucket -> CoV of that cell
    bucket_cov: dict[str, float | None]
    curve_index: CurveIndex
    from_: date | None
    to: date | None
    sources: list[str]

    @classmethod
    def build(
        cls, route_key: str, history: pl.DataFrame, *, dropped_currency_mismatch: int = 0
    ) -> RouteContext:
        buckets = build_bucket_distribution(history)
        samples = {
            str(key[0]): sorted(group["amount_minor"].to_list())
            for key, group in history.group_by("ap_bucket")
        }
        return cls(
            route_key=route_key,
            history=history,
            dropped_currency_mismatch=dropped_currency_mismatch,
            currency=None if history.is_empty() else history["currency"][0],
            bucket_samples=samples,
            bucket_cov=dict(zip(buckets["ap_bucket"], buckets["cov"], strict=True)),
            curve_index=CurveIndex.build(route_key, build_distribution(history), buckets),
            from_=history["fetched_date"].min() if not history.is_empty() else None,
            to=history["fetched_date"].max() if not history.is_empty() else None,
            sources=sorted(history["source"].unique().to_list()),
        )


def _latest_price_row(frame: pl.DataFrame, depart_date: date, as_of: date) -> dict | None:
    """The cheapest collapsed row for ``depart_date`` at the latest fetched_date <= as_of."""
    rows = frame.filter((pl.col("depart_date") == depart_date) & (pl.col("fetched_date") <= as_of))
    if rows.is_empty():
        return None
    latest = rows.filter(pl.col("fetched_date") == rows["fetched_date"].max())
    return latest.sort("amount_minor").row(0, named=True)


def _as_current_price(row: dict) -> CurrentPrice:
    return CurrentPrice(
        amount_minor=row["amount_minor"],
        currency=row["currency"],
        source=PriceSource.STORE,
        as_of=row["fetched_at"],
    )


def latest_observed_price(
    trip_shape: TripShape, *, as_of: date | None = None, store: SnapshotStore | None = None
) -> CurrentPrice | None:
    """The cheapest observation for this exact (route_key, depart_date) at the most recent
    fetched_date at or before as_of, after the prefer-itinerary collapse. `as_of` defaults
    to shared.clock.today_utc(), never the system clock. Returns None when the store has
    nothing for that trip shape.

    source = PriceSource.STORE; as_of = that row's fetched_at.
    This lives in models/, not api/, so SF-07 and the backtest resolve 'the current price'
    identically."""
    resolved_as_of = as_of if as_of is not None else today_utc()
    store = store if store is not None else default_store()
    if trip_shape.depart_date < resolved_as_of:
        return None
    raw = store.read_frame(
        ReadFilters(
            route_key=trip_shape.route_key,
            fetched_date_to=resolved_as_of,
            depart_date_from=trip_shape.depart_date,
            depart_date_to=trip_shape.depart_date,
        )
    )
    frame, _ = collapse_observations(raw)
    row = _latest_price_row(frame, trip_shape.depart_date, resolved_as_of)
    return _as_current_price(row) if row is not None else None


def _unavailable(
    *,
    trip_shape: TripShape,
    reason: DataUnavailableReason,
    note: str,
    current_price: CurrentPrice | None,
    curve: CurveResult,
    basis: Basis,
    config: BaselineConfig,
) -> Prediction:
    return Prediction(
        trip_shape=trip_shape,
        current_price=current_price,
        price_percentile=None,
        verdict=Verdict.NEUTRAL,
        expected_curve=list(curve.points),
        expected_low=None,
        confidence=Confidence.LOW,
        reason=build_reason(
            verdict=Verdict.NEUTRAL,
            price_percentile=None,
            expected_low=None,
            curve=curve,
            basis=basis,
            trip_shape=trip_shape,
            config=config,
            data_quality_note=note,
        ),
        basis=basis,
        data_quality_note=note,
        data_unavailable_reason=reason,
    )


def _empty_basis(trip_shape: TripShape, bucket: str | None) -> Basis:
    return Basis(
        observations=0,
        route_observations=0,
        from_=None,
        to=None,
        sources=[],
        ap_bucket=bucket,
        travel_month=trip_shape.depart_date.month,
        travel_dow=trip_shape.depart_date.weekday(),
        cov=None,
    )


def predict_from_context(
    trip_shape: TripShape,
    current_price: Money | None,
    *,
    as_of: date,
    config: BaselineConfig,
    context: RouteContext,
) -> Prediction:
    """``predict()`` over an already-loaded route. Pure: no clock, no store."""
    depart = trip_shape.depart_date
    dtd_now = (depart - as_of).days
    empty_curve = CurveResult(as_of=as_of, depart_date=depart, dtd_now=dtd_now)
    supplied = (
        CurrentPrice(
            amount_minor=current_price.amount_minor,
            currency=current_price.currency,
            source=PriceSource.USER_SUPPLIED,
            # Derived from as_of, not a second clock read, so the output is byte-identical
            # for fixed inputs.
            as_of=datetime.combine(as_of, time.min, tzinfo=UTC),
        )
        if current_price is not None
        else None
    )

    if dtd_now < 0:
        return _unavailable(
            trip_shape=trip_shape,
            reason=DataUnavailableReason.DEPARTURE_IN_PAST,
            note=NOTE_DEPARTURE_IN_PAST,
            current_price=supplied,
            curve=empty_curve,
            basis=_empty_basis(trip_shape, None),
            config=config,
        )

    bucket = ap_bucket(dtd_now, config)
    history_in_currency = context.currency is not None and (
        supplied is None or supplied.currency == context.currency
    )
    if not history_in_currency:
        note = (
            NOTE_NO_HISTORY_IN_CURRENCY.format(
                route_key=trip_shape.route_key, currency=supplied.currency
            )
            if context.currency is not None and supplied is not None
            else NOTE_NO_ROUTE_HISTORY.format(route_key=trip_shape.route_key)
        )
        return _unavailable(
            trip_shape=trip_shape,
            reason=DataUnavailableReason.NO_ROUTE_HISTORY,
            note=note,
            current_price=supplied,
            curve=empty_curve,
            basis=_empty_basis(trip_shape, bucket),
            config=config,
        )

    curve = curve_from_index(context.curve_index, trip_shape=trip_shape, as_of=as_of, config=config)
    sample = context.bucket_samples.get(bucket, [])
    cov = context.bucket_cov.get(bucket)
    basis = Basis(
        observations=len(sample),
        route_observations=context.history.height,
        from_=context.from_,
        to=context.to,
        sources=context.sources,
        ap_bucket=bucket,
        travel_month=depart.month,
        travel_dow=depart.weekday(),
        cov=cov,
        dropped_currency_mismatch=context.dropped_currency_mismatch,
    )

    resolved = supplied
    if resolved is None:
        row = _latest_price_row(context.history, depart, as_of)
        resolved = _as_current_price(row) if row is not None else None
    if resolved is None:
        return _unavailable(
            trip_shape=trip_shape,
            reason=DataUnavailableReason.NO_CURRENT_PRICE,
            note=NOTE_NO_CURRENT_PRICE,
            current_price=None,
            curve=curve,
            basis=basis,
            config=config,
        )

    if len(sample) < config.history.min_cell_observations:
        return _unavailable(
            trip_shape=trip_shape,
            reason=DataUnavailableReason.THIN_ROUTE_HISTORY,
            note=NOTE_THIN.format(n=len(sample)),
            current_price=resolved,
            curve=curve,
            basis=basis,
            config=config,
        )

    percentile = percentile_of_sorted(resolved.amount_minor, sample)
    verdict = decide_verdict(price_percentile=percentile, curve=curve, config=config)
    confidence = decide_confidence(cell_observations=len(sample), cov=cov, config=config)
    # The only place expected_low is decided, which is what makes "non-None iff wait" hold.
    expected_low = (
        find_expected_low(curve, config=config, currency=resolved.currency)
        if verdict == Verdict.WAIT
        else None
    )
    return Prediction(
        trip_shape=trip_shape,
        current_price=resolved,
        price_percentile=percentile,
        verdict=verdict,
        expected_curve=list(curve.points),
        expected_low=expected_low,
        confidence=confidence,
        reason=build_reason(
            verdict=verdict,
            price_percentile=percentile,
            expected_low=expected_low,
            curve=curve,
            basis=basis,
            trip_shape=trip_shape,
            config=config,
        ),
        basis=basis,
    )


def predict(
    trip_shape: TripShape,
    current_price: Money | None = None,
    *,
    as_of: date | None = None,
    config: BaselineConfig | None = None,
    store: SnapshotStore | None = None,
) -> Prediction:
    """SF-06's entry point. Never raises for thin or missing data — it returns a
    Prediction with verdict=neutral, confidence=low, a data_unavailable_reason and a
    data_quality_note (0002 D3). It raises only for a malformed TripShape, which pydantic
    rejects before this is called.

    Order of operations:
      1. as_of <- as_of or shared.clock.today_utc() — the only clock read in the call.
         config <- config or load_baseline_config().
      2. A departure before as_of is departure_in_past; no store read happens.
      3. history <- load_route_history(trip_shape.route_key, as_of=as_of, ...)
      4. Empty history (or none in the supplied currency) -> no_route_history.
      5. expected_curve and basis are built from the history.
      6. price <- current_price (source=user_supplied, as_of = as_of at 00:00 UTC) or the
         latest observed price; neither -> no_current_price (curve still returned).
      7. The (route, ap_bucket(dtd_now)) cell thinner than history.min_cell_observations
         -> thin_route_history (curve still returned).
      8. price_percentile, verdict, confidence; expected_low only for wait; reason.

    Deterministic: same store contents + same as_of + same config -> byte-identical
    Prediction.
    """
    resolved_as_of = as_of if as_of is not None else today_utc()
    config = config if config is not None else load_baseline_config()
    store = store if store is not None else default_store()

    if trip_shape.depart_date < resolved_as_of:
        history, dropped = collapse_observations(pl.DataFrame(), config)
    else:
        history, dropped = load_route_history_with_stats(
            trip_shape.route_key, as_of=resolved_as_of, config=config, store=store
        )
    context = RouteContext.build(trip_shape.route_key, history, dropped_currency_mismatch=dropped)
    return predict_from_context(
        trip_shape, current_price, as_of=resolved_as_of, config=config, context=context
    )
