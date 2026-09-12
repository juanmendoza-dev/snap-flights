"""Frozen evaluation scenarios (decision 0002 D5, SF-06-build §6a).

A scenario pins *which* samples the backtest scores — routes, as_of dates, the
days-to-departure range, the horizon — and the SHA-256 of the dataset it was written
against. It is versioned independently of the results, and the harness refuses to score a
dataset whose bytes have changed, so editing the fixture cannot quietly change the evaluation.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.baseline.config import REPO_ROOT
from pipeline.store import SnapshotStore

SCENARIOS_DIR: Path = Path(__file__).resolve().parent / "scenarios"
DEFAULT_SCENARIO_PATH: Path = SCENARIOS_DIR / "fixture-v1.yaml"


class ScenarioDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    #: Repo-relative path of the Parquet file the scenario was frozen against.
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    version: int = Field(ge=1)
    description: str
    dataset: ScenarioDataset
    routes: tuple[str, ...]
    as_of_dates: tuple[date, ...]
    stride_days: int = Field(ge=1)
    horizon_days: int = Field(ge=1)
    min_days_to_departure: int = Field(ge=1)
    max_days_to_departure: int

    @model_validator(mode="after")
    def _consistent(self) -> Scenario:
        if list(self.as_of_dates) != sorted(set(self.as_of_dates)):
            raise ValueError("as_of_dates must be strictly increasing")
        if self.max_days_to_departure < self.min_days_to_departure:
            raise ValueError("max_days_to_departure is below min_days_to_departure")
        if len(set(self.routes)) != len(self.routes):
            raise ValueError("routes must be unique")
        return self


class ScenarioDatasetMismatchError(RuntimeError):
    """The store's data is not the dataset the scenario was frozen against."""


def load_scenario(path: Path | None = None) -> Scenario:
    target = Path(path) if path is not None else DEFAULT_SCENARIO_PATH
    return Scenario.model_validate(yaml.safe_load(target.read_text()))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(scenario: Scenario, store: SnapshotStore) -> None:
    """Raise unless ``store`` reads exactly the dataset the scenario was frozen against:
    fixture mode on, the fixture file's bytes matching the recorded SHA, and no snapshot
    part files unioned in on top."""
    settings = store.settings
    expected = (REPO_ROOT / scenario.dataset.path).resolve()
    if not settings.use_fixtures:
        raise ScenarioDatasetMismatchError(
            f"scenario {scenario.name} scores the fixture; store is live"
        )
    if settings.fixture_parquet.resolve() != expected:
        raise ScenarioDatasetMismatchError(
            f"scenario {scenario.name} was frozen against {scenario.dataset.path}, "
            f"store reads {settings.fixture_parquet}"
        )
    actual = sha256_of(settings.fixture_parquet)
    if actual != scenario.dataset.sha256:
        raise ScenarioDatasetMismatchError(
            f"{scenario.dataset.path} has sha256 {actual}, scenario {scenario.name} "
            f"v{scenario.version} was frozen against {scenario.dataset.sha256}"
        )
    if any(settings.snapshots_root.glob("source=*/route_key=*/fetched_date=*/part-*.parquet")):
        raise ScenarioDatasetMismatchError("snapshot part files would be unioned into the fixture")
