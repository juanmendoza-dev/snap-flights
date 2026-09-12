"""Percentile rank (SF-06-build §4.2)."""

from __future__ import annotations

import polars as pl
import pytest

from models.baseline.percentile import (
    percentile_of,
    percentile_of_sorted,
    round_half_up,
    round_half_up_float,
)


def test_rank_definition() -> None:
    sample = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # 3 below, 0 equal, n=10 -> 30.
    assert percentile_of(350, sample) == 30
    # 2 below, 1 equal -> 100 * 2.5 / 10 = 25.
    assert percentile_of(300, sample) == 25


def test_ties_use_midrank() -> None:
    sample = [100, 200, 200, 200, 300]

    # 1 below, 3 equal: (1 + 1.5) / 5 = 50, where a strict below/n would say 20.
    assert percentile_of(200, sample) == 50
    assert percentile_of(200, [200] * 7) == 50


def test_low_percentile_means_cheap() -> None:
    sample = list(range(1000, 2000, 10))

    assert percentile_of(1, sample) == 0
    assert percentile_of(10**6, sample) == 100
    assert percentile_of(1100, sample) < percentile_of(1900, sample)


def test_rounding_is_half_up() -> None:
    # 1 below, 0 equal, n = 8 -> 12.5 -> 13 (banker's rounding would give 12).
    assert percentile_of(150, [100, 200, 300, 400, 500, 600, 700, 800]) == 13
    assert round_half_up(25, 2) == 13
    assert round_half_up(27, 2) == 14
    assert round_half_up_float(12.5) == 13
    assert round_half_up_float(13.5) == 14
    assert round_half_up_float(13.49) == 13


def test_empty_sample_raises() -> None:
    with pytest.raises(ValueError):
        percentile_of(100, [])


def test_accepts_a_series_and_matches_the_sorted_path() -> None:
    values = [500, 100, 400, 100, 300]

    assert percentile_of(300, pl.Series(values)) == percentile_of(300, values)
    assert percentile_of(300, values) == percentile_of_sorted(300, sorted(values))
