"""Backtest samples, scoring and the report (SF-06-build §5 as replaced by decision 0002 D5,
D6 and §6a).

Three arms, one sample set:

- ``baseline`` — the model. ``book_now`` and ``neutral`` pay ``price_now``; ``wait`` pays
  from the executable policy in ``score_wait``, never the hindsight minimum.
- ``always_book_now`` — every verdict forced to ``book_now``.
- ``hindsight_oracle`` — pays the minimum of ``price_now`` and every price in the scored
  window. A lower bound no real user can reach; never the baseline's own score.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

from models.baseline.types import Confidence, Verdict

ARM_BASELINE = "baseline"
ARM_ALWAYS_BOOK_NOW = "always_book_now"
ARM_HINDSIGHT_ORACLE = "hindsight_oracle"
ORACLE_LABEL = "hindsight lower bound: pays the cheapest observed price; no real user can"

_FROZEN = ConfigDict(frozen=True, extra="forbid")


def _r(value: float) -> float:
    """Floats in the report are rounded so a regenerated file diffs cleanly."""
    return round(value, 6)


@dataclass(frozen=True, slots=True)
class Quote:
    fetched_date: date
    amount_minor: int


@dataclass(frozen=True, slots=True)
class WaitOutcome:
    paid_minor: int
    target_error_minor: int
    window_hit: bool


def score_wait(
    window: list[Quote], *, target_minor: int, window_start: date, window_end: date
) -> WaitOutcome:
    """The executable ``wait`` policy (0002 D5). ``window`` is every observed quote in the
    scored window, ascending by fetched_date, and is never empty.

    Pay the first quote at or below the target whose fetched_date is inside
    ``[window_start, window_end]``. If none appears, pay the last quote in the scored window
    — the deadline (departure day, capped at the backtest horizon, §6a)."""
    for quote in window:
        if window_start <= quote.fetched_date <= window_end and quote.amount_minor <= target_minor:
            return WaitOutcome(
                paid_minor=quote.amount_minor,
                target_error_minor=quote.amount_minor - target_minor,
                window_hit=True,
            )
    deadline = window[-1].amount_minor
    return WaitOutcome(
        paid_minor=deadline, target_error_minor=deadline - target_minor, window_hit=False
    )


class BacktestSample(BaseModel):
    model_config = _FROZEN
    route_key: str
    depart_date: date
    as_of: date
    days_to_departure: int
    verdict: Verdict
    confidence: Confidence
    price_percentile: int | None
    data_unavailable_reason: str | None
    price_now_minor: int
    #: min observed price over fetched_date in (as_of, min(as_of + horizon, depart_date)]
    window_min_minor: int
    #: what the baseline paid under its verdict
    paid_minor: int
    #: min(price_now, window_min): the hindsight oracle's price
    oracle_paid_minor: int
    #: baseline book_now: price_now <= window_min; wait: paid < price_now; neutral: None
    hit: bool | None
    #: wait only
    expected_low_minor: int | None = None
    target_error_minor: int | None = None
    window_hit: bool | None = None
    #: fraction of the (route, AP bucket) cell priced strictly above price_now; None when
    #: no percentile was computed
    realised_fraction_cheaper: float | None = None

    @property
    def regret_minor(self) -> int:
        return self.paid_minor - self.oracle_paid_minor

    @property
    def always_book_now_hit(self) -> bool:
        return self.price_now_minor <= self.window_min_minor

    @property
    def actionable(self) -> bool:
        return self.verdict != Verdict.NEUTRAL


class ArmMetrics(BaseModel):
    model_config = _FROZEN
    label: str
    is_lower_bound: bool
    samples: int
    #: over the baseline's actionable (book_now + wait) samples; None for the oracle, which
    #: cannot miss by construction
    hit_rate: float | None
    actionable_samples: int
    mean_regret_minor: float
    median_regret_minor: float
    mean_regret_minor_actionable: float | None
    total_paid_minor: int


class WaitMetrics(BaseModel):
    model_config = _FROZEN
    samples: int
    hit_rate: float | None
    window_hit_rate: float | None
    mean_target_error_minor: float | None
    median_abs_target_error_minor: float | None
    #: mean of (price_now - paid): positive means waiting saved money against booking now
    mean_saving_vs_book_now_minor: float | None


class RankConsistencyBin(BaseModel):
    """Historical-rank consistency (0002 D6) — NOT forecast calibration. The percentile is
    checked against the same trailing cell it was ranked in, so a well-implemented rank lands
    near ``1 - predicted_mid / 100``; it says nothing about future prices."""

    model_config = _FROZEN
    decile: int
    predicted_mid: float
    realised_fraction_cheaper: float | None
    samples: int


class RouteMetrics(BaseModel):
    model_config = _FROZEN
    samples: int
    verdict_counts: dict[str, int]
    baseline_mean_regret_minor: float
    always_book_now_mean_regret_minor: float


class BacktestReport(BaseModel):
    model_config = _FROZEN
    generated_at: datetime
    scenario: str | None
    scenario_version: int | None
    dataset_sha256: str | None
    config_digest: str
    fixture_mode: bool
    horizon_days: int
    stride_days: int
    as_of_dates: list[date]
    samples: int
    skipped_no_future_observation: int
    verdict_counts: dict[str, int]
    unavailable_counts: dict[str, int]
    arms: dict[str, ArmMetrics]
    wait: WaitMetrics
    historical_rank_consistency: list[RankConsistencyBin]
    by_route: dict[str, RouteMetrics]


def _mean(values: list[float] | list[int]) -> float | None:
    return _r(statistics.fmean(values)) if values else None


def _median(values: list[float] | list[int]) -> float | None:
    return _r(float(statistics.median(values))) if values else None


def _rate(flags: list[bool]) -> float | None:
    return _r(sum(flags) / len(flags)) if flags else None


def _arm(
    label: str,
    samples: list[BacktestSample],
    *,
    paid: list[int],
    hits: list[bool] | None,
    lower_bound: bool = False,
) -> ArmMetrics:
    regrets = [p - s.oracle_paid_minor for p, s in zip(paid, samples, strict=True)]
    actionable = [r for r, s in zip(regrets, samples, strict=True) if s.actionable]
    return ArmMetrics(
        label=label,
        is_lower_bound=lower_bound,
        samples=len(samples),
        hit_rate=_rate(hits) if hits is not None else None,
        actionable_samples=len(actionable),
        mean_regret_minor=_mean(regrets) or 0.0,
        median_regret_minor=_median(regrets) or 0.0,
        mean_regret_minor_actionable=_mean(actionable),
        total_paid_minor=sum(paid),
    )


def rank_consistency(samples: list[BacktestSample]) -> list[RankConsistencyBin]:
    bins: list[RankConsistencyBin] = []
    for decile in range(10):
        members = [
            s.realised_fraction_cheaper
            for s in samples
            if s.price_percentile is not None
            and s.realised_fraction_cheaper is not None
            and min(s.price_percentile // 10, 9) == decile
        ]
        bins.append(
            RankConsistencyBin(
                decile=decile,
                predicted_mid=float(decile * 10 + 5),
                realised_fraction_cheaper=_mean(members),
                samples=len(members),
            )
        )
    return bins


def summarise(
    samples: list[BacktestSample],
    *,
    generated_at: datetime,
    config_digest: str,
    fixture_mode: bool,
    horizon_days: int,
    stride_days: int,
    as_of_dates: list[date],
    skipped: int,
    scenario: str | None = None,
    scenario_version: int | None = None,
    dataset_sha256: str | None = None,
) -> BacktestReport:
    actionable = [s for s in samples if s.actionable]
    waits = [s for s in samples if s.verdict == Verdict.WAIT]

    arms = {
        ARM_BASELINE: _arm(
            "the baseline model",
            samples,
            paid=[s.paid_minor for s in samples],
            hits=[bool(s.hit) for s in actionable],
        ),
        ARM_ALWAYS_BOOK_NOW: _arm(
            "every verdict forced to book_now",
            samples,
            paid=[s.price_now_minor for s in samples],
            hits=[s.always_book_now_hit for s in actionable],
        ),
        ARM_HINDSIGHT_ORACLE: _arm(
            ORACLE_LABEL,
            samples,
            paid=[s.oracle_paid_minor for s in samples],
            hits=None,
            lower_bound=True,
        ),
    }

    by_route: dict[str, RouteMetrics] = {}
    for route_key in sorted({s.route_key for s in samples}):
        members = [s for s in samples if s.route_key == route_key]
        by_route[route_key] = RouteMetrics(
            samples=len(members),
            verdict_counts=dict(sorted(Counter(str(s.verdict) for s in members).items())),
            baseline_mean_regret_minor=_mean([s.regret_minor for s in members]) or 0.0,
            always_book_now_mean_regret_minor=_mean(
                [s.price_now_minor - s.oracle_paid_minor for s in members]
            )
            or 0.0,
        )

    return BacktestReport(
        generated_at=generated_at,
        scenario=scenario,
        scenario_version=scenario_version,
        dataset_sha256=dataset_sha256,
        config_digest=config_digest,
        fixture_mode=fixture_mode,
        horizon_days=horizon_days,
        stride_days=stride_days,
        as_of_dates=as_of_dates,
        samples=len(samples),
        skipped_no_future_observation=skipped,
        verdict_counts={v.value: sum(s.verdict == v for s in samples) for v in Verdict},
        unavailable_counts=dict(
            sorted(
                Counter(
                    s.data_unavailable_reason for s in samples if s.data_unavailable_reason
                ).items()
            )
        ),
        arms=arms,
        wait=WaitMetrics(
            samples=len(waits),
            hit_rate=_rate([bool(s.hit) for s in waits]),
            window_hit_rate=_rate([bool(s.window_hit) for s in waits]),
            mean_target_error_minor=_mean([s.target_error_minor or 0 for s in waits]),
            median_abs_target_error_minor=_median([abs(s.target_error_minor or 0) for s in waits]),
            mean_saving_vs_book_now_minor=_mean([s.price_now_minor - s.paid_minor for s in waits]),
        ),
        historical_rank_consistency=rank_consistency(samples),
        by_route=by_route,
    )
