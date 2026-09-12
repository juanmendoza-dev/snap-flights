# SF-06 — Build Spec: Baseline Percentile / Seasonality Model

**Tier:** L3 (build spec) · **Parent:** `specs/subfeatures/SF-06-baseline-prediction-model.md`
**Execution slot:** third, after SF-03. Blocks SF-07.

## Summary

Implements SF-06 exactly: feature builders over `store.read_frame()`, the baseline
`predict()`, and the walk-forward backtest. Descriptive statistics only — no training, no ML
dependency.

> **Amended by decision 0002 — read it before implementing §4 and §5.** The technical review
> (`specs/review/03-prediction.md`) found this spec self-contradictory in three places.
> Decision 0002 resolves them and **overrides** the following, which are not yet rewritten
> inline below:
>
> - **§4.1 types** — `current_price`, `price_percentile`, `expected_low`, `basis.from_/to`
>   are nullable; add `data_unavailable_reason`. A missing-data `predict()` returns
>   `neutral` / `low` with that reason set (0002 §D3), not `price_percentile = 50`.
> - **§4.3 `find_expected_low` / §4.4 `decide_verdict`** — the curve result carries `as_of`
>   and `dtd_now` explicitly (not inferred from the first surviving point). `wait` and
>   `book_now` compare the curve against **its own current point**, and `wait`'s eligible
>   points are strictly after `as_of`; `dtd_now == 0` is always `neutral` (0002 §D4).
> - **§5 scoring + "If the baseline does not beat always-`book_now`"** — the fixture is
>   never edited to force a win, and there is no win gate. Score the baseline plus
>   `always_book_now` and a labelled `hindsight_oracle` lower bound. `wait` pays from the
>   executable policy in 0002 §D5, not the hindsight minimum. Add `target_error_minor` and
>   `window_hit`. Scenarios freeze under `models/backtest/scenarios/`.
> - **§5 calibration + §6 test** — rename to historical-rank consistency (0002 §D6). Split
>   the "every field non-null" test into a populated-route case and an empty/thin case.
>
> P3/P4/P5/P8 from the review are deferred to E2 Phase 2 (0002 "Deferred") and do not block
> this build.

## Depends on

- SF-03 on `main`, and specifically the interfaces frozen in `SF-03-build.md` §9.
- Decision 0001. Stats library: **polars** (P0's project-wide choice).

## Files owned

```
models/__init__.py                     # (exists from P0)
models/features/__init__.py            # re-exports: ap_bucket, AP_BUCKETS, build_* functions
models/features/buckets.py             # AP bucket definitions + ap_bucket()
models/features/observations.py        # route history frame: load, prefer-itinerary collapse
models/features/distributions.py       # cell distributions (median, p10..p90, count, CoV)
models/baseline/__init__.py            # re-exports: predict, latest_observed_price, types
models/baseline/config.py              # BaselineConfig + load_baseline_config()
models/baseline/types.py               # TripShape, Money, CurvePoint, ExpectedLow, Basis, Prediction
models/baseline/percentile.py          # percentile_of()
models/baseline/curve.py               # expected_curve() + smoothing
models/baseline/verdict.py             # decide_verdict(), decide_confidence(), build_reason()
models/baseline/predictor.py           # predict(), latest_observed_price()
models/backtest/__init__.py            # re-exports: run_backtest, write_report
models/backtest/harness.py             # walk-forward loop
models/backtest/metrics.py             # hit rate, regret, percentile calibration
models/backtest/reports/.gitkeep       # latest.json lands here
config/baseline.yaml                   # OWNED HERE — every threshold SF-06 uses
tests/models/__init__.py
tests/models/conftest.py               # fixture-mode store + a small synthetic frame factory
tests/models/test_buckets.py
tests/models/test_observations.py
tests/models/test_distributions.py
tests/models/test_percentile.py
tests/models/test_curve.py
tests/models/test_verdict.py
tests/models/test_predict.py
tests/models/test_config.py
tests/models/test_backtest.py
```

**Ownership note:** SF-06's L2 *Files owned* section lists only `models/**` and
`tests/models/**`, but its body says "config in `config/baseline.yaml`". The file is assigned
here. See `specs/implementation/README.md` open question 4.

**Not owned:** anything under `pipeline/`, `api/`, `config/quality.yaml`, `config/routes.yaml`.

## 1. Exact file tree

The block above is the tree, one line per path, nothing else.

## 2. `config/baseline.yaml` — committed default

Every number in SF-06's rules lives here. No threshold is a literal in code.

```yaml
# Baseline prediction model (SF-06). Every threshold the model uses lives in this file.
# Defaults are the values written into specs/subfeatures/SF-06-baseline-prediction-model.md.
schema_version: 1

history:
  # Trailing window of fetched_date used to build every distribution.
  window_days: 365
  # A (route, AP bucket) cell thinner than this cannot produce a verdict:
  # predict() returns neutral / low with a data_quality_note (SF-07 "thin data").
  min_cell_observations: 30

advance_purchase_buckets:
  - {name: "0-3",   min_days: 0,  max_days: 3}
  - {name: "4-7",   min_days: 4,  max_days: 7}
  - {name: "8-14",  min_days: 8,  max_days: 14}
  - {name: "15-21", min_days: 15, max_days: 21}
  - {name: "22-30", min_days: 22, max_days: 30}
  - {name: "31-45", min_days: 31, max_days: 45}
  - {name: "46-60", min_days: 46, max_days: 60}
  - {name: "61-90", min_days: 61, max_days: 90}
  - {name: "90+",   min_days: 91, max_days: null}   # null = unbounded

curve:
  # expected_curve spans days_to_departure from dtd_now down to max(0, dtd_now - horizon_days).
  horizon_days: 90
  # Centred rolling mean over the raw per-day series; 1 disables smoothing.
  smoothing_window_days: 7
  # Cells with fewer than this many observations fall back to the (route, AP bucket)
  # median instead of the (route, AP bucket, month, DOW) median.
  min_cell_observations: 10

verdict:
  book_now:
    max_percentile: 25
    min_curve_rise_pct: 5.0
  wait:
    min_percentile: 60
    min_curve_drop_pct: 7.0
    search_horizon_days: 60

confidence:
  low:
    max_observations: 100     # strictly fewer than this -> low
    max_cov: 0.35             # CoV strictly greater than this -> low
  high:
    min_observations: 500     # at least this many AND
    max_cov: 0.18             # CoV strictly below this -> high

backtest:
  horizon_days: 60
  # Fetched dates are sampled every N days across the available history.
  stride_days: 7
  report_path: "models/backtest/reports/latest.json"
```

Loader:

```python
# models/baseline/config.py
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

class BucketSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    min_days: int = Field(ge=0)
    max_days: int | None = None

class HistoryConfig(BaseModel): window_days: int; min_cell_observations: int
class CurveConfig(BaseModel): horizon_days: int; smoothing_window_days: int; min_cell_observations: int
class BookNowConfig(BaseModel): max_percentile: int; min_curve_rise_pct: float
class WaitConfig(BaseModel): min_percentile: int; min_curve_drop_pct: float; search_horizon_days: int
class VerdictConfig(BaseModel): book_now: BookNowConfig; wait: WaitConfig
class ConfidenceBand(BaseModel): ...
class ConfidenceConfig(BaseModel): low: ConfidenceBand; high: ConfidenceBand
class BacktestConfig(BaseModel): horizon_days: int; stride_days: int; report_path: str

class BaselineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int
    history: HistoryConfig
    advance_purchase_buckets: tuple[BucketSpec, ...]
    curve: CurveConfig
    verdict: VerdictConfig
    confidence: ConfidenceConfig
    backtest: BacktestConfig

DEFAULT_CONFIG_PATH: Path  # {repo_root}/config/baseline.yaml

def load_baseline_config(path: Path | None = None) -> BaselineConfig:
    """Parse and validate config/baseline.yaml. Cached per resolved path.
    extra='forbid' everywhere: a typo'd key is an error, not a silently ignored setting."""
```

## 3. Feature builders

### 3.1 `models/features/buckets.py`

```python
AP_BUCKETS: tuple[str, ...] = (
    "0-3", "4-7", "8-14", "15-21", "22-30", "31-45", "46-60", "61-90", "90+",
)

def ap_bucket(days_to_departure: int, config: BaselineConfig | None = None) -> str:
    """Bucket name for a days-to-departure value. Bounds are inclusive on both ends.
    Negative days_to_departure raises ValueError — a departed flight has no AP bucket."""

def ap_bucket_expr(config: BaselineConfig | None = None) -> pl.Expr:
    """The same mapping as a polars expression over a `days_to_departure` column, so the
    frame path and the scalar path cannot drift. Tested for equivalence over 0..400."""
```

### 3.2 `models/features/observations.py`

```python
OBSERVATION_FRAME_COLUMNS: tuple[str, ...] = (
    "route_key", "source", "price_kind", "fetched_date", "depart_date",
    "days_to_departure", "ap_bucket", "travel_month", "travel_dow", "amount_minor",
    "currency", "fetched_at",
)

def load_route_history(
    route_key: str,
    *,
    as_of: date,
    config: BaselineConfig | None = None,
    store: SnapshotStore | None = None,
) -> pl.DataFrame:
    """All observations for one route with fetched_date in
    [as_of - config.history.window_days, as_of], via store.read_frame().

    Adds the derived columns:
      days_to_departure = (depart_date - fetched_date).days
      ap_bucket         = ap_bucket_expr()
      travel_month      = depart_date.month        (1-12, month OF TRAVEL)
      travel_dow        = depart_date.weekday()    (0=Mon .. 6=Sun, DOW OF TRAVEL)

    Rows with days_to_departure < 0 are dropped (the flight departed before we saw it).
    Then applies the prefer-itinerary collapse below.

    Returns a frame with exactly OBSERVATION_FRAME_COLUMNS, sorted by
    (depart_date, fetched_date). Empty history returns an empty frame with that schema.
    """
```

**Prefer-itinerary collapse, pinned.** SF-06 says "when both exist for the same
(route, depart_date, fetched_date), prefer `itinerary`". Implemented as: group by
`(route_key, depart_date, fetched_date)`; if any row in the group has
`price_kind == "itinerary"`, keep only the cheapest such row; otherwise keep the cheapest
`calendar_cheapest` row. One observation per group, always. Ties inside a group break on
`observation_id` ascending so the result is deterministic.

`store.read_frame()` has already excluded `data_quality = "rejected"` and deduplicated on
`observation_id` (SF-03-build §9). This function does not repeat either.

`currency`: rows whose `currency` differs from the group's modal currency are dropped, and
the count is reported in `basis`. The fixture is single-currency so this is a no-op today; it
exists so a future multi-currency store cannot silently mix units into a percentile.

### 3.3 `models/features/distributions.py`

```python
DISTRIBUTION_COLUMNS: tuple[str, ...] = (
    "route_key", "ap_bucket", "travel_month", "travel_dow",
    "count", "median", "p10", "p25", "p75", "p90", "mean", "std", "cov",
)

BUCKET_DISTRIBUTION_COLUMNS: tuple[str, ...] = (
    "route_key", "ap_bucket", "count", "median", "p10", "p25", "p75", "p90",
    "mean", "std", "cov",
)

def build_distribution(observations: pl.DataFrame) -> pl.DataFrame:
    """Per (route_key, ap_bucket, travel_month, travel_dow). Backs expected_curve."""

def build_bucket_distribution(observations: pl.DataFrame) -> pl.DataFrame:
    """Per (route_key, ap_bucket). Backs price_percentile and confidence — this is the
    'cell' SF-06's confidence rules refer to."""
```

- Percentiles use **linear interpolation** between order statistics
  (`pl.quantile(..., interpolation="linear")`) so p10/p90 are stable on small cells.
- `cov = std / mean`, population std (`ddof=0`), `null` when `count < 2` or `mean == 0`.
  A `null` CoV is treated as "not below the high threshold" and "not above the low
  threshold" — it can neither earn `high` nor force `low` on its own.
- `median`, `p*`, `mean`, `std` are floats in minor units; only values that reach a response
  are rounded to `int`, and rounding is `round-half-up` at the single point of conversion.

## 4. The baseline model

### 4.1 `models/baseline/types.py`

```python
from datetime import date, datetime
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field

from pipeline.schema import Cabin, TripType
from pipeline.schema.record import IataCode

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

class Money(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    amount_minor: int = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")

class TripShape(BaseModel):
    """The MVP trip shape (L0 §8): one-way, economy, 1 passenger. Field names and types
    are the canonical ones from L0 §3."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    origin: IataCode
    destination: IataCode
    depart_date: date
    trip_type: TripType = TripType.ONE_WAY
    cabin: Cabin = Cabin.ECONOMY
    passengers: int = Field(default=1, ge=1)
    return_date: date | None = None

    @property
    def route_key(self) -> str: ...   # f"{origin}-{destination}" (L0 §2)

class CurrentPrice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    amount_minor: int = Field(gt=0)
    currency: str
    source: PriceSource
    as_of: datetime            # tz-aware UTC

class CurvePoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    days_to_departure: int = Field(ge=0)
    amount_minor: int = Field(gt=0)

class ExpectedLow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    amount_minor: int = Field(gt=0)
    currency: str
    window_start: date
    window_end: date           # window_end >= window_start, validated

class Basis(BaseModel):
    """Everything the numbers were computed from — the 'why' panel and the audit trail."""
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)
    observations: int                       # rows in the (route, AP BUCKET) cell — the
                                            # sample price_percentile was ranked in, NOT the
                                            # whole route (that is route_observations)
    route_observations: int                 # rows for the whole route in the window
    from_: date = Field(alias="from")       # min(fetched_date) actually used
    to: date = Field(alias="to")            # max(fetched_date) actually used
    sources: list[str]                      # distinct `source` values, sorted
    ap_bucket: str
    travel_month: int
    travel_dow: int
    cov: float | None
    dropped_currency_mismatch: int = 0

class Prediction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    trip_shape: TripShape
    current_price: CurrentPrice
    price_percentile: int = Field(ge=0, le=100)
    verdict: Verdict
    expected_curve: list[CurvePoint]
    expected_low: ExpectedLow | None
    confidence: Confidence
    reason: str
    basis: Basis
    data_quality_note: str | None = None
```

`Basis` uses `from` as a wire name via an alias because it is a Python keyword; SF-07
serialises `by_alias=True` so the JSON matches SF-07's documented payload exactly.

### 4.2 `models/baseline/percentile.py`

```python
def percentile_of(amount_minor: int, sample: pl.Series | Sequence[int]) -> int:
    """Percentile RANK of amount_minor within sample, as an int in [0, 100].

    Definition, pinned (L0 and SF-06 leave the direction implicit; SF-07's example payload
    'price_percentile: 34' + 'cheaper than 66% of the last year' fixes it):

        percentile = round(100 * (below + 0.5 * equal) / n)

    where `below` counts sample values strictly less than amount_minor and `equal` counts
    values equal to it. LOW MEANS CHEAP. A price cheaper than everything scores 0; a price
    dearer than everything scores 100. "Cheaper than X% of history" = 100 - percentile.

    n == 0 raises ValueError; callers check cell size first.
    """
```

The mid-rank (`+ 0.5 * equal`) term matters on this fixture: prices are rounded to whole
currency units, so exact ties are common and a strict `below / n` would systematically
under-report. It is also what makes SF-06's calibration bullet ("a p90 call is beaten ~10% of
the time") hold.

### 4.3 `models/baseline/curve.py`

```python
def expected_curve(
    *,
    trip_shape: TripShape,
    as_of: date,
    distribution: pl.DataFrame,          # from build_distribution()
    bucket_distribution: pl.DataFrame,   # from build_bucket_distribution()
    config: BaselineConfig,
) -> list[CurvePoint]:
    """The expected cheapest price for each remaining day before departure."""
```

**Pinned semantics — read this before implementing.**

`depart_date` is **fixed** by the request. `travel_month` and `travel_dow` are therefore
derived once from `trip_shape.depart_date` and are constant across the whole curve. The only
thing that varies as the curve advances is `days_to_departure`, and through it `ap_bucket`.
Do **not** vary the departure date along the curve — that would answer a different question
("what if I flew a different day"), not SF-06's.

1. `dtd_now = (trip_shape.depart_date - as_of).days`. If `dtd_now < 0`, return `[]`.
2. The curve covers `days_to_departure` from `dtd_now` down to `max(0, dtd_now - config.curve.horizon_days)`, inclusive, **descending** — literally "the next 90 days". For a departure 30 days out the curve is 31 points (30 → 0), not 91.
3. Raw value at each `dtd`: the `median` of the `(route, ap_bucket(dtd), travel_month, travel_dow)` row of `distribution`. If that cell has `count < config.curve.min_cell_observations` (or is missing), fall back to the `(route, ap_bucket(dtd))` row of `bucket_distribution`. If that is missing too, the point is omitted.
4. Because `ap_bucket` is a step function of `dtd`, the raw series is a **staircase of at most 9 distinct values**. This is why SF-06 says "smoothed".
5. Smoothing: a centred rolling mean of width `config.curve.smoothing_window_days` over the raw series ordered by ascending `dtd`, with shrinking windows at both ends (`min_periods=1`) so the endpoints are not dropped. Output is rounded to `int` minor units at this single point.
6. The result is a daily series with the bucket steps blended into ramps — monotone within each step, continuous across step boundaries.

```python
def find_expected_low(
    curve: Sequence[CurvePoint], *, depart_date: date, config: BaselineConfig, currency: str,
) -> ExpectedLow | None:
    """Minimum of the curve restricted to the next config.verdict.wait.search_horizon_days
    days — i.e. days_to_departure in [dtd_now - horizon, dtd_now], where dtd_now is the
    first (highest) days_to_departure in `curve`.

    Returns None ONLY when `curve` is empty. It does not second-guess the verdict: whether
    an expected low is worth showing is decided once, in predict() step 9.

    window_start / window_end are CALENDAR dates, mapped back via
    calendar_date = depart_date - days_to_departure. The window is the contiguous run of
    days CONTAINING THE ARGMIN whose curve value is within 1% of the minimum; on a tie the
    argmin with the largest days_to_departure (the earliest calendar date) wins.
    window_start <= window_end always.
    """
```

`as_of` is deliberately **not** a parameter: `dtd_now` is already the first point of `curve`,
and taking both would let a caller pass an inconsistent pair and get silent nonsense.

The 1%-of-minimum band turns a single argmin day into the usable booking window SF-07's
payload and SF-08's banner both want, instead of a one-day point estimate. "Contiguous run
containing the argmin" is pinned because a smoothed U-shaped curve can have two separate runs
inside the 1% band, which would otherwise leave the window implementation-defined.

**Why this returns a value even when the verdict will not be `wait`.** `decide_verdict`
compares the window minimum against the **user's current price**; the curve's own minimum
could sit at `dtd_now` while still being far below that price. If `find_expected_low` also
tried to decide, the two rules could disagree and `expected_low is not None iff verdict ==
"wait"` (§8.6) would break. One decision point, in `predict()`, makes that guarantee true by
construction.

### 4.4 `models/baseline/verdict.py`

```python
def decide_verdict(
    *, price_percentile: int, current_amount_minor: int, curve: Sequence[CurvePoint],
    as_of: date, depart_date: date, config: BaselineConfig,
) -> Verdict:
    """SF-06's rules, with every comparison pinned:

    book_now  price_percentile <= verdict.book_now.max_percentile
              AND max(curve value over the whole curve) >= current * (1 + min_curve_rise_pct/100)
              ("the curve rises >= 5% from here")

    wait      price_percentile >= verdict.wait.min_percentile
              AND min(curve value over days_to_departure in [dtd_now - search_horizon_days,
                  dtd_now]) <= current * (1 - min_curve_drop_pct/100)

    neutral   otherwise, and unconditionally when the curve is empty.

    book_now is evaluated first. The two conditions cannot both hold (percentile <= 25 and
    >= 60 are disjoint), so order is documentation, not tie-breaking.
    """

def decide_confidence(*, cell_observations: int, cov: float | None,
                      config: BaselineConfig) -> Confidence:
    """low   if cell_observations < confidence.low.max_observations
                OR (cov is not None AND cov > confidence.low.max_cov)
       high  if cell_observations >= confidence.high.min_observations
                AND cov is not None AND cov < confidence.high.max_cov
       medium otherwise.
       `low` is checked first: SF-06 lists it first and it is the safe direction."""

def build_reason(*, verdict: Verdict, price_percentile: int, confidence: Confidence,
                 expected_low: ExpectedLow | None, current_amount_minor: int,
                 basis: Basis, trip_shape: TripShape) -> str:
    """A plain-language sentence assembled from the numbers. No LLM, no randomness —
    same inputs, same string.

    Templates (the only three; {} are substituted, nothing else varies):

    book_now:
      "This fare is cheaper than {100 - price_percentile}% of the {n} observations we have
       for {route_key} booked {ap_bucket} days out, and prices on this route usually rise
       about {rise_pct}% from here. Book now."

    wait:
      "This fare is cheaper than {100 - price_percentile}% of the {n} observations we have
       for {route_key} booked {ap_bucket} days out, but prices on this route usually dip
       about {drop_pct}% around {window_start:%-d %b}-{window_end:%-d %b}. Waiting looks
       better."

    neutral:
      "This fare is cheaper than {100 - price_percentile}% of the {n} observations we have
       for {route_key} booked {ap_bucket} days out, and we do not see a clear move either
       way in the next {horizon} days."

    The observation count and route come from `basis`; the sentence NEVER claims a period
    ("the last year") that basis.from_/basis.to does not cover. See
    specs/implementation/README.md open question 2.
    """
```

### 4.5 `models/baseline/predictor.py`

```python
def latest_observed_price(
    trip_shape: TripShape, *, as_of: date | None = None, store: SnapshotStore | None = None,
) -> CurrentPrice | None:
    """The cheapest observation for this exact (route_key, depart_date) at the most recent
    fetched_date at or before as_of, after the prefer-itinerary collapse. `as_of` defaults
    to shared.clock.today_utc() — never date.today() (P0 §Interfaces frozen 8). Returns
    None when the store has nothing for that trip shape.

    source = PriceSource.STORE; as_of = that row's fetched_at.
    This lives in models/, not api/, so SF-07 and the backtest resolve 'the current price'
    identically."""

def predict(
    trip_shape: TripShape,
    current_price: Money | None = None,
    *,
    as_of: date | None = None,
    config: BaselineConfig | None = None,
    store: SnapshotStore | None = None,
) -> Prediction:
    """SF-06's entry point. Never raises for thin or missing data — it returns a
    Prediction with verdict=neutral, confidence=low and a populated data_quality_note.

    Order of operations:
      1. as_of  <- as_of or shared.clock.today_utc(). config <- config or
         load_baseline_config(). This is the only clock read in the whole call.
      2. history <- load_route_history(trip_shape.route_key, as_of=as_of, ...)
      3. resolved <- current_price (source=user_supplied, as_of = the step-1 as_of date
         at 00:00:00+00:00 — derived, NOT a second clock read, so the Prediction stays
         byte-identical for fixed inputs) or latest_observed_price(...)
         If both are absent -> neutral / low, data_unavailable_reason = no_current_price,
         note "No observed price for this route and date." price_percentile = None
         (0002 D3), expected_low = None. The curve depends on history, not on the quote,
         so expected_curve is still built (§8.5).
      4. distribution / bucket_distribution from history.
      5. cell <- bucket_distribution row for (route_key, ap_bucket(dtd_now))
         If missing or cell.count < history.min_cell_observations ->
         thin-data Prediction, note "Only {n} observations for this route at this booking
         window; not enough to call it." Curve is still returned when it can be built.
      6. price_percentile <- percentile_of(resolved.amount_minor, cell sample)
      7. expected_curve <- expected_curve(...)
      8. verdict <- decide_verdict(...); confidence <- decide_confidence(...)
      9. expected_low <- find_expected_low(...) if verdict is `wait`, else None. This is
         the ONLY place that decision is made (SF-06's table says "if wait"), which is what
         makes the iff in §8.6 hold by construction.
     10. basis <- Basis(...); reason <- build_reason(...)

    Deterministic: same store contents + same as_of + same config -> byte-identical
    Prediction. No wall-clock read anywhere except the as_of default, and that read goes
    through shared.clock.today_utc(), so SNAP_TODAY pins the whole call.
    """
```

`predict()` is pure with respect to its arguments — it reads `shared.clock.today_utc()`
exactly once, in step 1, and never touches a clock after that — so SF-07 can cache on
`(trip_shape, current_price, as_of_date)` safely. No module under `models/` calls
`date.today()` or `datetime.now()`; `shared.clock` is the only sanctioned source (P0
§Interfaces frozen 8).

## 5. Backtest harness

**Report types.** Superseded by decision 0002 D5/D6 and §6a. `models/backtest/metrics.py`
defines them: `BacktestSample` (per-sample `paid_minor` from the verdict's policy,
`oracle_paid_minor`, and `target_error_minor` / `window_hit` on `wait`) and `BacktestReport`
with `arms` (`baseline`, `always_book_now`, `hindsight_oracle`), `wait`,
`historical_rank_consistency` (10 decile bins) and `by_route`. The report carries aggregates
only, not samples.

```python
# models/backtest/harness.py
def run_backtest(
    *,
    routes: Sequence[str] | None = None,     # default: every route in the store
    config: BaselineConfig | None = None,
    store: SnapshotStore | None = None,
    stride_days: int | None = None,          # default: config.backtest.stride_days
    horizon_days: int | None = None,         # default: config.backtest.horizon_days
) -> BacktestReport: ...

def write_report(report: BacktestReport, path: Path | None = None) -> Path:
    """Writes JSON to config.backtest.report_path. Sorted keys, 2-space indent, trailing
    newline, so a regenerated report diffs cleanly. E8's accuracy page reads this file."""

def main(argv: list[str] | None = None) -> int:
    """`uv run python -m models.backtest` — runs and writes the report. --routes, --stride,
    --horizon, --out."""
```

**Scoring, pinned** (SF-06 names the metrics but not the rule):

For each sampled `(route_key, depart_date, as_of)` where `as_of` steps by `stride_days`
across the available `fetched_date` range and `1 <= dtd_now`:

- The model sees only rows with `fetched_date <= as_of` (walk-forward; enforced by passing
  `as_of` through to `load_route_history`, and asserted by a test).
- `price_now` = the collapsed observation for `(route, depart_date, as_of)`.
- `best_future` = min collapsed price for `(route, depart_date)` over
  `fetched_date` in `(as_of, min(as_of + horizon_days, depart_date)]`. Samples with no
  future observation are skipped.
- **Paid prices, hits, regret and the reference arms: see §6a** (decision 0002 D5). `wait`
  pays from the executable policy, never the hindsight minimum. All three arms are scored on
  the same sample set.

**There is no win gate** (0002 D5). The fixture is never edited, and `config/baseline.yaml`
is never retuned, to make the baseline beat a reference arm. A loss is reported in
`latest.json` and in the commit that regenerates it.

**Historical-rank consistency** (0002 D6: not forecast calibration). Samples are binned by
`price_percentile` decile. Within a bin, `realised_fraction_cheaper` is the mean fraction of
the same (route, AP bucket) trailing cell the percentile was ranked in whose prices are
strictly above the scored price. A consistent p90 bin sits near 0.10. That is a property of
the rank, not a forward probability. Tolerances come from the first clean run and are
recorded in that commit's message.

## 6. Test list — mapped to SF-06's Done when

| SF-06 "Done when" bullet | Test |
|---|---|
| `predict()` returns a `Prediction` for every route in the fixture set — populated where data supports it, honest `neutral` / `low` where not (0002 D3) | **Populated case:** `tests/models/test_predict.py::test_predict_all_fixture_routes` (parametrised over the 15 route keys × 3 departure dates at dtd 10 / 45 / 100; asserts every field non-null, `data_unavailable_reason is None`, `expected_curve` non-empty, `basis.observations > 0`). **Empty / thin case:** `::test_predict_unavailable_data_is_honest` (no history, no price, thin cell, past departure: `neutral` / `low`, reason enum set, `price_percentile is None`, user price echoed). Plus `::test_predict_is_deterministic`, `::test_predict_without_current_price_uses_store`, `::test_predict_never_raises_on_unknown_route` |
| Historical-rank consistency holds on the fixture data (0002 D6 — not forward calibration) | `tests/models/test_backtest.py::test_historical_rank_consistency_within_tolerance` (tolerances set from the first clean run and recorded in commit 9's message), `tests/models/test_percentile.py::test_rank_definition`, `::test_ties_use_midrank`, `::test_low_percentile_means_cheap` |
| The backtest produces hit rate, regret and window error for the baseline and both reference arms (0002 D5 — no win gate) | `tests/models/test_backtest.py::test_report_has_three_arms`, `::test_same_sample_set_for_every_arm`, `::test_oracle_is_a_lower_bound`, `::test_wait_pays_from_the_executable_policy`, `::test_walk_forward_never_reads_the_future` (monkeypatches `load_route_history` and asserts no call receives `as_of` beyond the sample's) |
| `reason` and `basis` are populated and consistent with the numeric outputs | `tests/models/test_predict.py::test_reason_matches_verdict_template`, `::test_reason_percentile_matches_field` (the "cheaper than X%" in the string equals `100 - price_percentile`), `::test_reason_never_claims_a_period_outside_basis`, `::test_basis_counts_match_cell_size`, `::test_basis_sources_match_rows_used`, `::test_basis_from_to_within_history_window` |
| Everything works with `SNAP_USE_FIXTURES=1` and no `data/snapshots/` | `tests/models/conftest.py` builds every store fixture that way; `tests/models/test_predict.py::test_works_in_fixture_mode_with_no_snapshots_dir` asserts the directory is absent and the call still succeeds |
| All thresholds are config-driven | `tests/models/test_config.py::test_defaults_match_sf06_spec` (25 / 5.0 / 60 / 7.0 / 100 / 0.35 / 500 / 0.18), `::test_unknown_key_is_rejected`, `::test_missing_key_is_rejected`, `tests/models/test_verdict.py::test_lowering_book_now_threshold_changes_verdict`, `::test_raising_confidence_floor_changes_confidence`, `tests/models/test_predict.py::test_no_threshold_literals_in_module_source` (greps `models/baseline/curve.py` and `verdict.py` for the float forms `5.0`, `7.0`, `0.35`, `0.18` and fails if any appears; the integer thresholds are not grepped because `100` and `60` legitimately occur in percentile and date arithmetic) |

Supporting tests:

| Test | Asserts |
|---|---|
| `test_buckets.py::test_bucket_boundaries` | 0→`0-3`, 3→`0-3`, 4→`4-7`, 90→`61-90`, 91→`90+`, 400→`90+` |
| `test_buckets.py::test_expr_matches_scalar` | `ap_bucket_expr()` equals `ap_bucket()` for every value 0..400 |
| `test_buckets.py::test_negative_raises` | `ap_bucket(-1)` raises `ValueError` |
| `test_observations.py::test_prefer_itinerary_collapse` | on a tier-1 route the itinerary row wins over the calendar row for the same (depart, fetched) |
| `test_observations.py::test_collapse_is_one_row_per_group` | group counts are all 1 |
| `test_observations.py::test_derived_columns` | `days_to_departure`, `travel_month`, `travel_dow` correct for a known date; `travel_dow` is Mon=0 |
| `test_observations.py::test_travel_month_dow_come_from_depart_not_fetched` | the trap the curve depends on |
| `test_observations.py::test_history_window_respected` | rows older than `window_days` are excluded |
| `test_observations.py::test_frame_schema_is_frozen` | `OBSERVATION_FRAME_COLUMNS` exactly |
| `test_distributions.py::test_cell_counts_match_sf03_build_table` | per-(route, AP bucket) counts equal `SF-03-build.md` §8 |
| `test_distributions.py::test_cov_null_on_single_row` | |
| `test_distributions.py::test_percentiles_ordered` | p10 ≤ p25 ≤ median ≤ p75 ≤ p90 for every cell |
| `test_curve.py::test_curve_length_short_horizon` | departure 30 days out → 31 points |
| `test_curve.py::test_curve_length_long_horizon` | departure 200 days out → 91 points, ending at dtd 110 |
| `test_curve.py::test_curve_is_descending_in_dtd` | first point is `dtd_now` |
| `test_curve.py::test_month_and_dow_constant_across_curve` | asserts the distribution lookup uses one (month, dow) pair |
| `test_curve.py::test_smoothing_removes_the_staircase` | raw series has ≤ 9 distinct values, smoothed has more, and endpoints survive |
| `test_curve.py::test_falls_back_to_bucket_median_on_thin_cell` | |
| `test_curve.py::test_expected_low_window_is_a_contiguous_calendar_range` | `window_start <= window_end`, both map back through `depart_date - dtd` |
| `test_curve.py::test_expected_low_none_only_on_empty_curve` | a curve whose minimum is at `dtd_now` still returns an ExpectedLow |
| `test_curve.py::test_expected_low_window_is_the_run_containing_the_argmin` | a two-trough curve inside the 1% band picks the argmin's run |
| `test_curve.py::test_error_fares_do_not_move_expected_low` | injects the 12 fixture outliers' route/date and asserts the curve is unchanged (medians, never minima) |
| `test_verdict.py::test_book_now_rule`, `::test_wait_rule`, `::test_neutral_when_neither`, `::test_neutral_on_empty_curve` | the three rules, at and either side of each threshold |
| `test_verdict.py::test_confidence_bands` | `(99, 0.1) -> low`, `(270, 0.1) -> medium`, `(600, 0.1) -> high`, `(600, 0.4) -> low`, `(600, None) -> medium` |
| `test_verdict.py::test_fixture_short_buckets_cap_at_medium` | `0-3` and `4-7` never return `high` on the fixture (matches `SF-03-build.md` §8) |
| `test_predict.py::test_expected_low_only_set_when_wait` | |
| `test_predict.py::test_thin_data_note_shape` | thin cell → `neutral` / `low` / non-null note / HTTP-safe (no raise) |
| `test_backtest.py::test_report_json_is_stable` | two runs produce byte-identical JSON |
| `test_backtest.py::test_report_written_to_configured_path` | |

## 6a. Decision 0002 as built — choices 0002 leaves open

Pinned here so the code has one place to cite. Everything else in 0002 applies verbatim.

**Curve result.** `expected_curve()` returns `CurveResult(as_of, depart_date, dtd_now,
points)`; `Prediction.expected_curve` stays `list[CurvePoint]` (= `result.points`). A
`dtd_now` point that was omitted (no cell, no bucket fallback) means there is no current
curve value, and the verdict is `neutral`.

**Expected low.** `wait_candidates(curve, config)` returns the eligible future points that
clear `wait` condition 2. `decide_verdict` and `find_expected_low` both call it, so the two
cannot disagree. `find_expected_low` returns `None` iff that set is empty (which includes
`dtd_now == 0`). The window is the run of consecutive `days_to_departure` values containing
the argmin, within the candidates, whose value is within 1% of the minimum.

**Reason text.** `price_percentile` is `int | None`. The three templates substitute the
*realised* curve move (`(max - now) / now` for `book_now`, `(now - low) / now` for `wait`),
not the config threshold. A fourth template covers unavailable data and never states a
percentile.

**Unavailable data.** Reasons are checked in this order: `departure_in_past` →
`no_route_history` → `no_current_price` → `thin_route_history`. `basis` counts are `0` and
`sources` `[]` for the first two, where there is nothing to count. `no_current_price` and
`thin_route_history` have rows, so `basis` reports what was actually there. For a thin cell
that means the note's "Only {n} observations" and `basis.observations` are the same `n`.
`Basis.ap_bucket` is `str | None`, `None` only for `departure_in_past` (a departed flight
has no AP bucket, §3.1).

A caller-supplied price whose currency differs from the route history's is never ranked
against it (L0 forbids conversion). 0002's enum has no dedicated value for this, so it maps
to `no_route_history` ("no history in this unit"), with basis counts `0` and the note naming
the currency: "No price history for {route} in {currency}." SF-07 should show
`data_quality_note` for this branch, because the enum alone would wrongly suggest the route
has no data at all.

**Backtest scoring.**
- Scored window: `fetched_date` in `(as_of, min(as_of + horizon_days, depart_date)]`.
  Samples with no observation in it are skipped.
- `hindsight_oracle` pays `min(price_now, min price in window)`.
- `book_now` and `neutral` pay `price_now`. Neutral means "no reason to wait".
- `wait` pays the first in-window quote at or below `expected_low.amount_minor` with
  `fetched_date` in `[window_start, window_end]`. If none appears, it pays the last observed
  quote in the scored window: the deadline is the departure day, capped at the backtest
  horizon so that every arm is scored on the same observations and the oracle stays a true
  lower bound.
- Hit: `book_now` when `price_now <= min(window)`; `wait` when `paid < price_now`.
- `regret_minor = paid - oracle_paid`, which is never negative.
- `target_error_minor` and `window_hit` are set on `wait` samples only.

**Scenarios.** `models/backtest/scenarios/fixture-v1.yaml` freezes the routes, `as_of` dates,
`days_to_departure` range, horizon, and the SHA-256 of the dataset they were written against.
The harness refuses to score when the SHA does not match, so a fixture edit cannot silently
change the evaluation. `run_backtest()` keeps its §5 signature; `main()` loads the default
scenario.

## 7. Out of scope

Everything SF-06 lists (trained model, quantile regression, SHAP, HTTP, UI), plus: no writes
to the store, no `config/routes.yaml`, no route metadata — SF-06 works from `route_key`
alone and never needs to know a route's region or tier.

## 8. Interfaces frozen for downstream (what SF-07 may assume about SF-06)

1. `from models.baseline import predict, latest_observed_price, TripShape, Money,
   CurrentPrice, Prediction, CurvePoint, ExpectedLow, Basis, Verdict, Confidence,
   PriceSource, DataUnavailableReason, BaselineConfig, load_baseline_config`
   (`CurveResult` is exported too; SF-07 does not need it). Checked by
   `tests/models/test_predict.py::test_public_interface_frozen_for_sf07`.
2. `predict(trip_shape, current_price=None, *, as_of=None, config=None, store=None)
   -> Prediction` **never raises** for thin data, an unknown route, an empty store, or a
   departure date outside the fixture range. It raises only for a malformed `TripShape`,
   which SF-07 has already rejected with a 400.
3. `Prediction` is a Pydantic v2 model. `Prediction.model_dump(mode="json", by_alias=True)`
   produces JSON-ready values (dates as ISO strings, `basis.from_` serialised as `from`).
   SF-07 nests it, it does not re-derive it.
4. `price_percentile` is an `int` in `[0, 100]` where **low means cheap**, or `null` exactly
   when `data_unavailable_reason` is set (0002 D3). SF-07's `reason` and any UI must read
   "cheaper than `100 - price_percentile`% of history".
5. `expected_curve` is ordered **descending** by `days_to_departure` and has at most
   `config.curve.horizon_days + 1` points, none above `dtd_now`. A day whose AP bucket has no
   data at all is **omitted**, so the first point may be *below* `dtd_now` (§6a). Callers
   read the current value by `days_to_departure`, never by position, and a missing `dtd_now`
   point forces `neutral`. The curve is `[]` only when the departure date is in the past or
   the route has no usable history. On the committed fixture every route has all 9 buckets,
   so no point is ever omitted there.
6. `expected_low` is non-`None` **iff** `verdict == "wait"`, enforced by `Prediction`'s own
   validator.
7. Missing data (the four `data_unavailable_reason` values: thin data, unknown routes, no
   price, a past departure) comes back as `verdict="neutral"`, `confidence="low"`,
   `data_unavailable_reason` and `data_quality_note` both non-`None`. SF-07 maps this
   straight to its documented `200` response, with no special-casing in `api/`.
8. `latest_observed_price(trip_shape, as_of=None, store=None) -> CurrentPrice | None` is
   the only sanctioned way to fill an omitted `current_price`. Its `as_of` defaults to
   `shared.clock.today_utc()`. SF-07 must not query the store itself.
9. `predict()` reads the clock exactly once — `shared.clock.today_utc()`, as the `as_of`
   default — so caching on `(trip_shape, current_price, as_of)` is sound. SF-07 gets the
   same date from the same function for its cache key, so the two cannot disagree.
10. `config/baseline.yaml` is SF-06's file. SF-07 reads it only through
    `load_baseline_config()` and puts its own settings in `config/api.yaml`.
11. Confidence on the committed fixture: the `0-3` and `4-7` AP buckets cap at `medium`
    (270 / 360 observations vs. a 500 floor). SF-07's `GET /routes` must report that
    honestly rather than assuming every route can reach `high`.

## 9. Ordered commit plan

| # | Message | Contains |
|---|---|---|
| 1 | `Add the baseline config file and its loader` | `config/baseline.yaml`, `models/baseline/config.py`, `tests/models/test_config.py` |
| 2 | `Bucket days-to-departure for the feature builders` | `models/features/buckets.py`, `tests/models/test_buckets.py` |
| 3 | `Load route history and collapse to one row per day` | `models/features/observations.py`, `tests/models/conftest.py`, `tests/models/test_observations.py` |
| 4 | `Build the trailing price distributions` | `models/features/distributions.py`, `tests/models/test_distributions.py` |
| 5 | `Add the percentile rank and the prediction types` | `models/baseline/types.py`, `percentile.py`, `tests/models/test_percentile.py` |
| 6 | `Build and smooth the 90-day expected curve` | `models/baseline/curve.py`, `tests/models/test_curve.py` |
| 7 | `Decide the verdict, confidence and reason` | `models/baseline/verdict.py`, `tests/models/test_verdict.py` |
| 8 | `Wire it together behind predict()` | `models/baseline/predictor.py`, `__init__.py` files, `tests/models/test_predict.py` |
| 9 | `Score the baseline with a walk-forward backtest` | `models/backtest/**`, `tests/models/test_backtest.py`, the first `reports/latest.json` |

Push after each. Commits 1–4 are the feature layer and can be reviewed without reading the
model; 9 is the only one that commits a generated artefact.
