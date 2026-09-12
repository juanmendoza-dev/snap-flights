"""The walk-forward backtest (SF-06-build §5, decision 0002 D5, §6a).

At each ``as_of`` the model sees only rows with ``fetched_date <= as_of`` — enforced by
passing ``as_of`` to ``load_route_history`` — and is scored against what the price actually
did afterwards. Ground truth is read separately, once per route, and never reaches the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_right
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from models.backtest.metrics import (
    BacktestReport,
    BacktestSample,
    Quote,
    score_wait,
    summarise,
)
from models.backtest.scenario import Scenario, load_scenario, verify_dataset
from models.baseline.config import REPO_ROOT, BaselineConfig, load_baseline_config
from models.baseline.predictor import RouteContext, predict_from_context
from models.baseline.types import Money, TripShape, Verdict
from models.features.observations import collapse_observations, load_route_history
from pipeline.store import ReadFilters, SnapshotStore, default_store
from shared.clock import now_utc


def config_digest(config: BaselineConfig) -> str:
    return hashlib.sha256(config.model_dump_json().encode()).hexdigest()


def _quotes_by_departure(truth: pl.DataFrame) -> dict[date, list[Quote]]:
    quotes: dict[date, list[Quote]] = {}
    for depart, fetched, amount in truth.select(
        "depart_date", "fetched_date", "amount_minor"
    ).iter_rows():
        quotes.setdefault(depart, []).append(Quote(fetched_date=fetched, amount_minor=amount))
    for series in quotes.values():
        series.sort(key=lambda q: q.fetched_date)
    return quotes


def _derived_as_of_dates(
    store: SnapshotStore, routes: Sequence[str], stride_days: int
) -> list[date]:
    frame = store.read_frame(ReadFilters(route_key=list(routes)))
    if frame.is_empty():
        return []
    fetched = frame["fetched_at"].dt.convert_time_zone("UTC").dt.date()
    first, last = fetched.min(), fetched.max()
    return [first + timedelta(days=d) for d in range(0, (last - first).days + 1, stride_days)]


def collect_samples(
    *,
    routes: Sequence[str],
    as_of_dates: Sequence[date],
    config: BaselineConfig,
    store: SnapshotStore,
    horizon_days: int,
    min_days_to_departure: int = 1,
    max_days_to_departure: int | None = None,
) -> tuple[list[BacktestSample], int]:
    """Score every ``(route, depart_date, as_of)`` with a quote on ``as_of`` and
    ``min_dtd <= dtd <= max_dtd``. Returns ``(samples, skipped)``; a sample with no quote in
    the scored window is skipped."""
    samples: list[BacktestSample] = []
    skipped = 0
    for route_key in routes:
        truth, _ = collapse_observations(store.read_frame(ReadFilters(route_key=route_key)), config)
        quotes = _quotes_by_departure(truth)
        origin, destination = route_key.split("-")

        for as_of in as_of_dates:
            history = load_route_history(route_key, as_of=as_of, config=config, store=store)
            context = RouteContext.build(route_key, history)

            for depart in sorted(quotes):
                dtd = (depart - as_of).days
                if dtd < min_days_to_departure or (
                    max_days_to_departure is not None and dtd > max_days_to_departure
                ):
                    continue
                series = quotes[depart]
                today = [q for q in series if q.fetched_date == as_of]
                if not today:
                    continue
                price_now = today[0].amount_minor
                window_end = min(as_of + timedelta(days=horizon_days), depart)
                start = bisect_right([q.fetched_date for q in series], as_of)
                window = [q for q in series[start:] if q.fetched_date <= window_end]
                if not window:
                    skipped += 1
                    continue

                prediction = predict_from_context(
                    TripShape(origin=origin, destination=destination, depart_date=depart),
                    Money(amount_minor=price_now, currency=truth["currency"][0]),
                    as_of=as_of,
                    config=config,
                    context=context,
                )
                window_min = min(q.amount_minor for q in window)
                oracle = min(price_now, window_min)
                sample = {
                    "route_key": route_key,
                    "depart_date": depart,
                    "as_of": as_of,
                    "days_to_departure": dtd,
                    "verdict": prediction.verdict,
                    "confidence": prediction.confidence,
                    "price_percentile": prediction.price_percentile,
                    "data_unavailable_reason": prediction.data_unavailable_reason,
                    "price_now_minor": price_now,
                    "window_min_minor": window_min,
                    "oracle_paid_minor": oracle,
                    "paid_minor": price_now,
                    "hit": None,
                }
                if prediction.verdict == Verdict.BOOK_NOW:
                    sample["hit"] = price_now <= window_min
                elif prediction.verdict == Verdict.WAIT:
                    low = prediction.expected_low
                    assert low is not None  # Prediction enforces expected_low iff wait
                    outcome = score_wait(
                        window,
                        target_minor=low.amount_minor,
                        window_start=low.window_start,
                        window_end=low.window_end,
                    )
                    sample |= {
                        "paid_minor": outcome.paid_minor,
                        "hit": outcome.paid_minor < price_now,
                        "expected_low_minor": low.amount_minor,
                        "target_error_minor": outcome.target_error_minor,
                        "window_hit": outcome.window_hit,
                    }
                if prediction.price_percentile is not None and prediction.basis.ap_bucket:
                    cell = context.bucket_samples[prediction.basis.ap_bucket]
                    above = len(cell) - bisect_right(cell, price_now)
                    sample["realised_fraction_cheaper"] = above / len(cell)
                samples.append(BacktestSample(**sample))
    return samples, skipped


def run_backtest(
    *,
    routes: Sequence[str] | None = None,
    config: BaselineConfig | None = None,
    store: SnapshotStore | None = None,
    stride_days: int | None = None,
    horizon_days: int | None = None,
    scenario: Scenario | None = None,
) -> BacktestReport:
    """Run the walk-forward backtest and summarise it.

    With a ``scenario`` the sample set is the frozen one: its routes (optionally narrowed by
    ``routes``), its as_of dates and dtd range, its horizon unless ``horizon_days`` is given,
    and the dataset SHA is checked first. Without one, routes default to every route in the
    store and as_of steps by ``stride_days`` across the available fetched_date range.
    """
    config = config if config is not None else load_baseline_config()
    store = store if store is not None else default_store()

    if scenario is not None:
        verify_dataset(scenario, store)
        chosen = [r for r in scenario.routes if routes is None or r in set(routes)]
        as_of_dates = list(scenario.as_of_dates)
        horizon = horizon_days if horizon_days is not None else scenario.horizon_days
        stride = scenario.stride_days
        dtd_range = (scenario.min_days_to_departure, scenario.max_days_to_departure)
    else:
        stride = stride_days if stride_days is not None else config.backtest.stride_days
        horizon = horizon_days if horizon_days is not None else config.backtest.horizon_days
        if routes is None:
            frame = store.read_frame()
            chosen = sorted(frame["route_key"].unique().to_list())
        else:
            chosen = list(routes)
        as_of_dates = _derived_as_of_dates(store, chosen, stride) if chosen else []
        dtd_range = (1, None)

    samples, skipped = collect_samples(
        routes=chosen,
        as_of_dates=as_of_dates,
        config=config,
        store=store,
        horizon_days=horizon,
        min_days_to_departure=dtd_range[0],
        max_days_to_departure=dtd_range[1],
    )
    return summarise(
        samples,
        generated_at=now_utc(),
        config_digest=config_digest(config),
        fixture_mode=store.settings.use_fixtures,
        horizon_days=horizon,
        stride_days=stride,
        as_of_dates=as_of_dates,
        skipped=skipped,
        scenario=scenario.name if scenario else None,
        scenario_version=scenario.version if scenario else None,
        dataset_sha256=scenario.dataset.sha256 if scenario else None,
    )


def report_json(report: BacktestReport) -> str:
    """Sorted keys, 2-space indent, trailing newline, so a regenerated report diffs cleanly."""
    return json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def write_report(
    report: BacktestReport, path: Path | None = None, *, config: BaselineConfig | None = None
) -> Path:
    """Writes JSON to config.backtest.report_path (repo-relative) unless ``path`` is given.
    E8's accuracy page reads this file."""
    if path is None:
        config = config if config is not None else load_baseline_config()
        path = REPO_ROOT / config.backtest.report_path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_json(report))
    return path


def main(argv: list[str] | None = None) -> int:
    """`uv run python -m models.backtest` — runs the frozen default scenario and writes the
    report. --routes narrows it, --horizon overrides its horizon, --scenario picks another
    file. --stride runs an ad-hoc walk over the store instead of a frozen scenario."""
    parser = argparse.ArgumentParser(prog="python -m models.backtest")
    parser.add_argument("--routes", help="comma-separated route keys")
    parser.add_argument("--stride", type=int, help="ad-hoc run: as_of every N days")
    parser.add_argument("--horizon", type=int, help="scored window in days")
    parser.add_argument("--scenario", type=Path, help="scenario YAML (default fixture-v1)")
    parser.add_argument("--out", type=Path, help="report path (default from config)")
    args = parser.parse_args(argv)

    routes = args.routes.split(",") if args.routes else None
    scenario = None if args.stride is not None else load_scenario(args.scenario)
    report = run_backtest(
        routes=routes, stride_days=args.stride, horizon_days=args.horizon, scenario=scenario
    )
    path = write_report(report, args.out)
    baseline = report.arms["baseline"]
    print(
        f"{report.samples} samples, verdicts {report.verdict_counts}, "
        f"baseline hit rate {baseline.hit_rate}, mean regret {baseline.mean_regret_minor} "
        f"-> {path}"
    )
    return 0
