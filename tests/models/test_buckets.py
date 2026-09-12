"""Advance-purchase buckets (SF-06-build §3.1)."""

from __future__ import annotations

import polars as pl
import pytest

from models.baseline.config import load_baseline_config
from models.features.buckets import AP_BUCKETS, ap_bucket, ap_bucket_expr


@pytest.mark.parametrize(
    ("dtd", "expected"),
    [(0, "0-3"), (3, "0-3"), (4, "4-7"), (90, "61-90"), (91, "90+"), (400, "90+")],
)
def test_bucket_boundaries(dtd: int, expected: str) -> None:
    assert ap_bucket(dtd) == expected


def test_expr_matches_scalar() -> None:
    frame = pl.DataFrame({"days_to_departure": list(range(401))})

    via_expr = frame.select(ap_bucket_expr()).to_series().to_list()

    assert via_expr == [ap_bucket(d) for d in range(401)]


def test_negative_raises() -> None:
    with pytest.raises(ValueError):
        ap_bucket(-1)


def test_negative_is_null_in_the_expression() -> None:
    frame = pl.DataFrame({"days_to_departure": [-1]})

    assert frame.select(ap_bucket_expr()).item() is None


def test_constant_matches_config() -> None:
    names = tuple(b.name for b in load_baseline_config().advance_purchase_buckets)

    assert names == AP_BUCKETS
