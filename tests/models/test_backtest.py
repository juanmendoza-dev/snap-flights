"""The walk-forward backtest (SF-06-build §5 as replaced by decision 0002 D5/D6, §6a)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

import models.backtest.harness as harness
from models.backtest import run_backtest, write_report
from models.backtest.harness import collect_samples, main, report_json
from models.backtest.metrics import (
    ARM_ALWAYS_BOOK_NOW,
    ARM_BASELINE,
    ARM_HINDSIGHT_ORACLE,
    BacktestReport,
    BacktestSample,
    Quote,
    score_wait,
)
from models.backtest.scenario import (
    Scenario,
    ScenarioDatasetMismatchError,
    load_scenario,
    verify_dataset,
)
from models.baseline import Verdict, load_baseline_config
from pipeline.store import SnapshotStore
from shared.settings import load_data_settings
from tests.models.conftest import FIXTURE_AS_OF, fixture_route_keys

CONFIG = load_baseline_config()
REPO = Path(__file__).resolve().parents[2]
COMMITTED_REPORT = REPO / CONFIG.backtest.report_path

#: Historical-rank consistency tolerances (0002 D6), set from the first clean run of
#: scenario fixture-v1: the largest |realised - (1 - mid/100)| over the ten deciles was
#: 0.0065 (p90 bin: -0.0065). Roughly 3x headroom on that spread.
RANK_TOLERANCE = 0.02
RANK_TOLERANCE_P90 = 0.015

#: Routes whose samples the per-sample tests inspect: a tier-1 transatlantic, a
#: transpacific and a cheap intra-Europe route.
SAMPLE_ROUTES = ("JFK-LHR", "LAX-NRT", "MAD-LIS")


@pytest.fixture(scope="session")
def scenario() -> Scenario:
    return load_scenario()


@pytest.fixture(scope="session")
def fixture_report(fixture_store: SnapshotStore, scenario: Scenario) -> BacktestReport:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SNAP_TODAY", FIXTURE_AS_OF.isoformat())
        return run_backtest(scenario=scenario, store=fixture_store)


@pytest.fixture(scope="session")
def samples(fixture_store: SnapshotStore, scenario: Scenario) -> list[BacktestSample]:
    collected, _ = collect_samples(
        routes=SAMPLE_ROUTES,
        as_of_dates=scenario.as_of_dates,
        config=CONFIG,
        store=fixture_store,
        horizon_days=scenario.horizon_days,
        min_days_to_departure=scenario.min_days_to_departure,
        max_days_to_departure=scenario.max_days_to_departure,
    )
    return collected


# --- the three arms (0002 D5) --------------------------------------------------------------


@pytest.mark.slow
def test_report_has_three_arms(fixture_report: BacktestReport) -> None:
    assert set(fixture_report.arms) == {ARM_BASELINE, ARM_ALWAYS_BOOK_NOW, ARM_HINDSIGHT_ORACLE}
    assert fixture_report.arms[ARM_BASELINE].hit_rate is not None
    assert fixture_report.arms[ARM_ALWAYS_BOOK_NOW].hit_rate is not None
    assert fixture_report.wait.samples == fixture_report.verdict_counts["wait"]
    assert fixture_report.samples == sum(fixture_report.verdict_counts.values())


@pytest.mark.slow
def test_same_sample_set_for_every_arm(fixture_report: BacktestReport) -> None:
    arms = fixture_report.arms.values()

    assert {arm.samples for arm in arms} == {fixture_report.samples}
    actionable = fixture_report.verdict_counts["book_now"] + fixture_report.verdict_counts["wait"]
    assert {arm.actionable_samples for arm in arms} == {actionable}


@pytest.mark.slow
def test_oracle_is_a_lower_bound(fixture_report: BacktestReport) -> None:
    oracle = fixture_report.arms[ARM_HINDSIGHT_ORACLE]
    baseline = fixture_report.arms[ARM_BASELINE]
    always = fixture_report.arms[ARM_ALWAYS_BOOK_NOW]

    assert oracle.is_lower_bound and "lower bound" in oracle.label
    assert not baseline.is_lower_bound and not always.is_lower_bound
    assert oracle.hit_rate is None
    assert oracle.mean_regret_minor == 0.0
    assert oracle.total_paid_minor <= min(baseline.total_paid_minor, always.total_paid_minor)
    assert baseline.mean_regret_minor >= 0 and always.mean_regret_minor >= 0


def test_wait_pays_from_the_executable_policy() -> None:
    window = [
        Quote(date(2026, 9, 2), 30000),  # cheapest of all, but before the predicted window
        Quote(date(2026, 9, 5), 41000),
        Quote(date(2026, 9, 6), 39000),  # first in-window quote at or below target
        Quote(date(2026, 9, 7), 35000),
        Quote(date(2026, 9, 20), 45000),  # deadline
    ]

    hit = score_wait(
        window, target_minor=40000, window_start=date(2026, 9, 5), window_end=date(2026, 9, 8)
    )
    assert (hit.paid_minor, hit.target_error_minor, hit.window_hit) == (39000, -1000, True)

    miss = score_wait(
        window, target_minor=30000, window_start=date(2026, 9, 5), window_end=date(2026, 9, 8)
    )
    # The 30000 quote is outside the window; hindsight would have paid it, the policy cannot.
    assert (miss.paid_minor, miss.target_error_minor, miss.window_hit) == (45000, 15000, False)


def test_sample_payments_follow_the_verdict(samples: list[BacktestSample]) -> None:
    verdicts = {s.verdict for s in samples}
    assert Verdict.WAIT in verdicts and Verdict.BOOK_NOW in verdicts

    for s in samples:
        assert s.oracle_paid_minor == min(s.price_now_minor, s.window_min_minor)
        assert s.regret_minor >= 0
        if s.verdict == Verdict.WAIT:
            assert s.target_error_minor is not None and s.window_hit is not None
            assert s.expected_low_minor is not None
            assert s.target_error_minor == s.paid_minor - s.expected_low_minor
            assert s.hit == (s.paid_minor < s.price_now_minor)
        else:
            assert s.paid_minor == s.price_now_minor
            assert s.target_error_minor is None and s.window_hit is None
            if s.verdict == Verdict.BOOK_NOW:
                assert s.hit == (s.price_now_minor <= s.window_min_minor)
            else:
                assert s.hit is None


def test_walk_forward_never_reads_the_future(
    fixture_store: SnapshotStore, scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, date]] = []
    real = harness.load_route_history

    def spy(route_key: str, *, as_of: date, **kwargs):
        frame = real(route_key, as_of=as_of, **kwargs)
        calls.append((route_key, as_of))
        if not frame.is_empty():
            assert frame["fetched_date"].max() <= as_of
        return frame

    monkeypatch.setattr(harness, "load_route_history", spy)

    collected, _ = collect_samples(
        routes=["MAD-LIS"],
        as_of_dates=scenario.as_of_dates,
        config=CONFIG,
        store=fixture_store,
        horizon_days=scenario.horizon_days,
    )

    assert calls, "the harness must load history through load_route_history"
    assert {as_of for _, as_of in calls} == set(scenario.as_of_dates)
    loaded = set(calls)
    for sample in collected:
        assert (sample.route_key, sample.as_of) in loaded
        assert sample.as_of < sample.depart_date


# --- historical-rank consistency (0002 D6) -------------------------------------------------


@pytest.mark.slow
def test_historical_rank_consistency_within_tolerance(fixture_report: BacktestReport) -> None:
    bins = fixture_report.historical_rank_consistency
    assert [b.decile for b in bins] == list(range(10))

    for b in bins:
        assert b.samples > 0 and b.realised_fraction_cheaper is not None
        expected = 1 - b.predicted_mid / 100
        tolerance = RANK_TOLERANCE_P90 if b.decile == 9 else RANK_TOLERANCE
        assert abs(b.realised_fraction_cheaper - expected) <= tolerance, b


# --- report file ---------------------------------------------------------------------------


def test_report_json_is_stable(
    fixture_store: SnapshotStore, scenario: Scenario, monkeypatch
) -> None:
    monkeypatch.setenv("SNAP_TODAY", FIXTURE_AS_OF.isoformat())

    first = report_json(run_backtest(routes=["MAD-LIS"], scenario=scenario, store=fixture_store))
    second = report_json(run_backtest(routes=["MAD-LIS"], scenario=scenario, store=fixture_store))

    assert first == second
    assert first.endswith("}\n")


def test_report_written_to_configured_path(fixture_report: BacktestReport, tmp_path: Path) -> None:
    target = tmp_path / "reports" / "latest.json"
    config = CONFIG.model_copy(
        update={"backtest": CONFIG.backtest.model_copy(update={"report_path": str(target)})}
    )

    written = write_report(fixture_report, config=config)

    assert written == target
    assert target.read_text() == report_json(fixture_report)
    assert json.loads(target.read_text())["samples"] == fixture_report.samples


@pytest.mark.slow
def test_committed_report_is_current(fixture_report: BacktestReport) -> None:
    """models/backtest/reports/latest.json is what the current code produces on the frozen
    scenario. Regenerate with:
    SNAP_USE_FIXTURES=1 SNAP_TODAY=2026-09-09 uv run python -m models.backtest"""
    assert COMMITTED_REPORT.read_text() == report_json(fixture_report)


def test_main_writes_a_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNAP_USE_FIXTURES", "1")
    monkeypatch.setenv("SNAP_TODAY", FIXTURE_AS_OF.isoformat())
    out = tmp_path / "report.json"

    assert main(["--routes", "MAD-LIS", "--out", str(out)]) == 0

    report = json.loads(out.read_text())
    assert report["scenario"] == "fixture-v1"
    assert set(report["by_route"]) == {"MAD-LIS"}


# --- frozen scenarios ----------------------------------------------------------------------


def test_scenario_file_is_well_formed(scenario: Scenario) -> None:
    assert sorted(scenario.routes) == sorted(fixture_route_keys())
    steps = {
        (b - a).days for a, b in zip(scenario.as_of_dates, scenario.as_of_dates[1:], strict=False)
    }
    assert steps == {scenario.stride_days}
    assert scenario.as_of_dates[-1] + timedelta(days=scenario.stride_days) > FIXTURE_AS_OF


def test_scenario_refuses_a_changed_dataset(
    fixture_store: SnapshotStore, scenario: Scenario, tmp_path: Path
) -> None:
    verify_dataset(scenario, fixture_store)

    tampered = scenario.model_copy(
        update={"dataset": scenario.dataset.model_copy(update={"sha256": "0" * 64})}
    )
    with pytest.raises(ScenarioDatasetMismatchError, match="sha256"):
        verify_dataset(tampered, fixture_store)

    live = SnapshotStore(load_data_settings(repo_root=tmp_path, use_fixtures=False))
    with pytest.raises(ScenarioDatasetMismatchError, match="live"):
        run_backtest(scenario=scenario, store=live)
