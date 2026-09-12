"""predict() end to end (SF-06-build §4.5, §6, §8; decision 0002 D3)."""

from __future__ import annotations

import os
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import models.baseline.predictor as predictor_module
from models.baseline import (
    Confidence,
    DataUnavailableReason,
    Money,
    Prediction,
    PriceSource,
    TripShape,
    Verdict,
    latest_observed_price,
    load_baseline_config,
    predict,
)
from pipeline.store import SnapshotStore
from shared.settings import load_data_settings
from tests.models.conftest import FIXTURE_AS_OF, fixture_route_keys, make_row

CONFIG = load_baseline_config()
REPO = Path(__file__).resolve().parents[2]


def _shape(route_key: str, dtd: int, as_of: date = FIXTURE_AS_OF) -> TripShape:
    origin, destination = route_key.split("-")
    return TripShape(
        origin=origin, destination=destination, depart_date=as_of + timedelta(days=dtd)
    )


@pytest.fixture(scope="module")
def fixture_predictions(fixture_store: SnapshotStore) -> dict[tuple[str, int], Prediction]:
    """The 15 fixture routes x dtd 10 / 45 / 100, predicted once for the module."""
    return {
        (route_key, dtd): predict(_shape(route_key, dtd), as_of=FIXTURE_AS_OF, store=fixture_store)
        for route_key in fixture_route_keys()
        for dtd in (10, 45, 100)
    }


# --- populated case ------------------------------------------------------------------------


@pytest.mark.parametrize("route_key", fixture_route_keys())
@pytest.mark.parametrize("dtd", [10, 45, 100])
def test_predict_all_fixture_routes(route_key: str, dtd: int, fixture_predictions) -> None:
    prediction = fixture_predictions[(route_key, dtd)]

    assert prediction.data_unavailable_reason is None
    assert prediction.data_quality_note is None
    assert prediction.current_price is not None
    assert prediction.price_percentile is not None
    assert prediction.expected_curve
    assert prediction.reason
    assert prediction.basis.observations > 0
    assert prediction.basis.route_observations > 0
    assert prediction.basis.from_ is not None
    assert prediction.basis.to is not None
    assert prediction.basis.sources
    assert prediction.basis.ap_bucket is not None
    assert prediction.basis.cov is not None
    assert (prediction.expected_low is not None) == (prediction.verdict == Verdict.WAIT)


def test_fixture_exercises_more_than_one_verdict(fixture_predictions) -> None:
    verdicts = {p.verdict for p in fixture_predictions.values()}

    assert len(verdicts) >= 2, verdicts


# --- empty / thin case (0002 D3) -----------------------------------------------------------


def _thin_store(synthetic_store, n: int) -> SnapshotStore:
    depart = FIXTURE_AS_OF + timedelta(days=10)
    return synthetic_store(
        make_row(
            fetched_date=FIXTURE_AS_OF - timedelta(days=i),
            depart_date=depart,
            amount_minor=40000 + i,
        )
        for i in range(n)
    )


def test_predict_unavailable_data_is_honest(fixture_store, synthetic_store) -> None:
    user_price = Money(amount_minor=45000, currency="USD")
    far_future = FIXTURE_AS_OF + timedelta(days=170)  # past the fixture's last depart_date

    cases = {
        DataUnavailableReason.NO_ROUTE_HISTORY: predict(
            TripShape(origin="AAA", destination="BBB", depart_date=far_future),
            user_price,
            as_of=FIXTURE_AS_OF,
            store=fixture_store,
        ),
        DataUnavailableReason.NO_CURRENT_PRICE: predict(
            _shape("JFK-LHR", 170), as_of=FIXTURE_AS_OF, store=fixture_store
        ),
        DataUnavailableReason.THIN_ROUTE_HISTORY: predict(
            _shape("JFK-LHR", 10),
            user_price,
            as_of=FIXTURE_AS_OF,
            store=_thin_store(synthetic_store, 5),
        ),
        DataUnavailableReason.DEPARTURE_IN_PAST: predict(
            _shape("JFK-LHR", -1), user_price, as_of=FIXTURE_AS_OF, store=fixture_store
        ),
    }

    for reason, prediction in cases.items():
        assert prediction.data_unavailable_reason == reason
        assert prediction.verdict == Verdict.NEUTRAL
        assert prediction.confidence == Confidence.LOW
        assert prediction.price_percentile is None
        assert prediction.expected_low is None
        assert prediction.data_quality_note
        assert "%" not in prediction.reason

    no_history = cases[DataUnavailableReason.NO_ROUTE_HISTORY]
    assert no_history.current_price is not None
    assert no_history.current_price.source == PriceSource.USER_SUPPLIED
    assert no_history.current_price.amount_minor == 45000
    assert no_history.expected_curve == []
    assert (no_history.basis.observations, no_history.basis.route_observations) == (0, 0)
    assert no_history.basis.sources == []
    assert (no_history.basis.from_, no_history.basis.to) == (None, None)

    no_price = cases[DataUnavailableReason.NO_CURRENT_PRICE]
    assert no_price.current_price is None
    assert no_price.expected_curve, "the curve depends on history, not on the quote"

    thin = cases[DataUnavailableReason.THIN_ROUTE_HISTORY]
    assert thin.basis.observations == 5
    assert "Only 5 observations" in (thin.data_quality_note or "")

    past = cases[DataUnavailableReason.DEPARTURE_IN_PAST]
    assert past.basis.ap_bucket is None
    assert past.expected_curve == []
    assert past.current_price is not None and past.current_price.amount_minor == 45000


def test_thin_data_note_shape(synthetic_store) -> None:
    thin = CONFIG.history.min_cell_observations - 1
    prediction = predict(
        _shape("JFK-LHR", 10), as_of=FIXTURE_AS_OF, store=_thin_store(synthetic_store, thin)
    )

    assert prediction.verdict == Verdict.NEUTRAL
    assert prediction.confidence == Confidence.LOW
    assert prediction.data_quality_note is not None
    assert prediction.data_unavailable_reason == DataUnavailableReason.THIN_ROUTE_HISTORY
    # HTTP-safe: serialises cleanly the way SF-07 will.
    dumped = prediction.model_dump(mode="json", by_alias=True)
    assert dumped["basis"]["from"] == (FIXTURE_AS_OF - timedelta(days=thin - 1)).isoformat()
    assert dumped["data_unavailable_reason"] == "thin_route_history"


def test_price_in_another_currency_is_not_ranked(fixture_store) -> None:
    prediction = predict(
        _shape("JFK-LHR", 10),
        Money(amount_minor=40000, currency="EUR"),
        as_of=FIXTURE_AS_OF,
        store=fixture_store,
    )

    assert prediction.data_unavailable_reason == DataUnavailableReason.NO_ROUTE_HISTORY
    assert prediction.price_percentile is None
    assert "EUR" in (prediction.data_quality_note or "")


def test_departure_today_is_neutral_but_populated(synthetic_store) -> None:
    rows = [
        make_row(
            fetched_date=FIXTURE_AS_OF - timedelta(days=k),
            depart_date=FIXTURE_AS_OF - timedelta(days=k) + timedelta(days=k % 4),
            amount_minor=30000 + 100 * k,
        )
        for k in range(60)
    ]
    rows.append(make_row(fetched_date=FIXTURE_AS_OF, depart_date=FIXTURE_AS_OF, amount_minor=20000))
    store = synthetic_store(rows)

    prediction = predict(_shape("JFK-LHR", 0), as_of=FIXTURE_AS_OF, store=store)

    assert prediction.data_unavailable_reason is None
    assert prediction.price_percentile is not None
    assert prediction.verdict == Verdict.NEUTRAL
    assert prediction.reason.endswith("and the flight leaves today.")


# --- behaviour -----------------------------------------------------------------------------


def test_predict_is_deterministic(fixture_store, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNAP_TODAY", FIXTURE_AS_OF.isoformat())
    shape = _shape("LAX-NRT", 45)
    price = Money(amount_minor=90000, currency="USD")

    first = predict(shape, price, store=fixture_store).model_dump_json(by_alias=True)
    second = predict(shape, price, store=fixture_store).model_dump_json(by_alias=True)

    assert first == second


def test_predict_reads_the_clock_exactly_once(
    fixture_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake_today() -> date:
        calls.append(1)
        return FIXTURE_AS_OF

    monkeypatch.setattr(predictor_module, "today_utc", fake_today)

    prediction = predict(_shape("JFK-LHR", 45), store=fixture_store)

    assert calls == [1]
    assert prediction.basis.to == FIXTURE_AS_OF


def test_no_wall_clock_reads_under_models() -> None:
    offenders = [
        str(path.relative_to(REPO))
        for path in (REPO / "models").rglob("*.py")
        if re.search(r"date\.today\(|datetime\.now\(|datetime\.utcnow\(", path.read_text())
    ]

    assert offenders == []


def test_predict_without_current_price_uses_store(fixture_store) -> None:
    shape = _shape("JFK-LHR", 30)

    prediction = predict(shape, as_of=FIXTURE_AS_OF, store=fixture_store)
    latest = latest_observed_price(shape, as_of=FIXTURE_AS_OF, store=fixture_store)

    assert latest is not None
    assert prediction.current_price == latest
    assert latest.source == PriceSource.STORE
    assert latest.as_of == datetime(2026, 9, 9, 6, 0, tzinfo=UTC)


def test_latest_observed_price_walks_back_to_the_newest_fetch(synthetic_store) -> None:
    depart = date(2026, 10, 1)
    store = synthetic_store(
        [
            make_row(fetched_date=date(2026, 9, 1), depart_date=depart, amount_minor=30000),
            make_row(fetched_date=date(2026, 9, 5), depart_date=depart, amount_minor=35000),
            make_row(fetched_date=date(2026, 9, 12), depart_date=depart, amount_minor=20000),
        ]
    )
    shape = TripShape(origin="JFK", destination="LHR", depart_date=depart)

    latest = latest_observed_price(shape, as_of=FIXTURE_AS_OF, store=store)

    assert latest is not None
    assert latest.amount_minor == 35000
    assert latest_observed_price(shape, as_of=date(2026, 8, 1), store=store) is None


def test_predict_never_raises_on_unknown_route(fixture_store, synthetic_store) -> None:
    unknown = TripShape(
        origin="ZZZ", destination="YYY", depart_date=FIXTURE_AS_OF + timedelta(days=9)
    )

    assert predict(unknown, as_of=FIXTURE_AS_OF, store=fixture_store).verdict == Verdict.NEUTRAL
    assert predict(
        _shape("JFK-LHR", 9), as_of=FIXTURE_AS_OF, store=synthetic_store([])
    ).verdict == (Verdict.NEUTRAL)
    far = predict(_shape("JFK-LHR", 1000), as_of=FIXTURE_AS_OF, store=fixture_store)
    assert far.data_unavailable_reason == DataUnavailableReason.NO_CURRENT_PRICE


def test_malformed_trip_shape_raises() -> None:
    with pytest.raises(ValidationError):
        TripShape(origin="JF", destination="LHR", depart_date=FIXTURE_AS_OF)


def test_expected_low_only_set_when_wait(fixture_predictions) -> None:
    for prediction in fixture_predictions.values():
        assert (prediction.expected_low is not None) == (prediction.verdict == Verdict.WAIT)


# --- reason and basis consistency ----------------------------------------------------------

TEMPLATES: dict[Verdict, re.Pattern[str]] = {
    Verdict.BOOK_NOW: re.compile(
        r"^This fare is cheaper than \d+% of the \d+ observations we have for [A-Z]{3}-[A-Z]{3} "
        r"booked \S+ days out, and prices on this route usually rise about \d+% from here\. "
        r"Book now\.$"
    ),
    Verdict.WAIT: re.compile(
        r"^This fare is cheaper than \d+% of the \d+ observations we have for [A-Z]{3}-[A-Z]{3} "
        r"booked \S+ days out, but prices on this route usually dip about \d+% around "
        r"\d+ [A-Z][a-z]{2}-\d+ [A-Z][a-z]{2}\. Waiting looks better\.$"
    ),
    Verdict.NEUTRAL: re.compile(
        r"^This fare is cheaper than \d+% of the \d+ observations we have for [A-Z]{3}-[A-Z]{3} "
        r"booked \S+ days out, and (we do not see a clear move either way in the next \d+ days"
        r"|the flight leaves today)\.$"
    ),
}


def test_reason_matches_verdict_template(fixture_predictions) -> None:
    for prediction in fixture_predictions.values():
        assert TEMPLATES[prediction.verdict].match(prediction.reason), prediction.reason


def test_reason_percentile_matches_field(fixture_predictions) -> None:
    for prediction in fixture_predictions.values():
        match = re.search(r"cheaper than (\d+)% of the (\d+) observations", prediction.reason)
        assert match is not None
        assert prediction.price_percentile is not None
        assert int(match.group(1)) == 100 - prediction.price_percentile
        assert int(match.group(2)) == prediction.basis.observations
        assert prediction.basis.ap_bucket is not None
        assert f"booked {prediction.basis.ap_bucket} days out" in prediction.reason


def test_reason_never_claims_a_period_outside_basis(fixture_predictions) -> None:
    for prediction in fixture_predictions.values():
        assert not re.search(r"\byears?\b|\blast\b|\bmonths?\b", prediction.reason)


def test_basis_counts_match_cell_size(fixture_predictions, fixture_history) -> None:
    for (route_key, _dtd), prediction in fixture_predictions.items():
        history = fixture_history(route_key)
        cell = history.filter(history["ap_bucket"] == prediction.basis.ap_bucket)
        assert prediction.basis.observations == cell.height
        assert prediction.basis.route_observations == history.height


def test_basis_sources_match_rows_used(fixture_predictions, fixture_history) -> None:
    for (route_key, _dtd), prediction in fixture_predictions.items():
        expected = sorted(fixture_history(route_key)["source"].unique().to_list())
        assert prediction.basis.sources == expected
    assert fixture_predictions[("JFK-LHR", 10)].basis.sources == ["fastflights", "travelpayouts"]
    assert fixture_predictions[("ATL-MIA", 10)].basis.sources == ["travelpayouts"]


def test_basis_from_to_within_history_window(fixture_predictions) -> None:
    earliest = FIXTURE_AS_OF - timedelta(days=CONFIG.history.window_days)
    for prediction in fixture_predictions.values():
        assert prediction.basis.from_ is not None and prediction.basis.to is not None
        assert earliest <= prediction.basis.from_ <= prediction.basis.to <= FIXTURE_AS_OF
        assert prediction.basis.from_ == date(2026, 6, 12)


def test_works_in_fixture_mode_with_no_snapshots_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "fixtures").symlink_to(REPO / "data" / "fixtures")
    monkeypatch.setenv("SNAP_USE_FIXTURES", "1")
    settings = load_data_settings(repo_root=tmp_path)

    assert settings.use_fixtures
    assert not (tmp_path / "data" / "snapshots").exists()

    prediction = predict(_shape("JFK-LHR", 45), as_of=FIXTURE_AS_OF, store=SnapshotStore(settings))

    assert prediction.data_unavailable_reason is None
    assert os.environ["SNAP_USE_FIXTURES"] == "1"


# --- config-driven and frozen interface ----------------------------------------------------


def test_no_threshold_literals_in_module_source() -> None:
    for name in ("curve.py", "verdict.py"):
        source = (REPO / "models" / "baseline" / name).read_text()
        for literal in ("5.0", "7.0", "0.35", "0.18"):
            assert literal not in source, f"{literal} is a threshold literal in {name}"


def test_public_interface_frozen_for_sf07() -> None:
    """SF-06-build §8 item 1, plus the reason enum 0002 D3 added."""
    import models.baseline as baseline

    for name in (
        "predict", "latest_observed_price", "TripShape", "Money", "CurrentPrice", "Prediction",
        "CurvePoint", "ExpectedLow", "Basis", "Verdict", "Confidence", "PriceSource",
        "BaselineConfig", "load_baseline_config", "DataUnavailableReason",
    ):  # fmt: skip
        assert hasattr(baseline, name), name


def test_predict_signature_matches_section_8() -> None:
    import inspect

    params = inspect.signature(predict).parameters
    assert list(params) == ["trip_shape", "current_price", "as_of", "config", "store"]
    assert params["current_price"].default is None
    assert all(
        params[name].kind is inspect.Parameter.KEYWORD_ONLY for name in ("as_of", "config", "store")
    )
