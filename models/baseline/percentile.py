"""Percentile rank of a price within a sample (SF-06-build §4.2)."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence

import polars as pl


def round_half_up(numerator: int, denominator: int) -> int:
    """``numerator / denominator`` rounded half-up, in exact integer arithmetic, for
    non-negative inputs. Python's ``round()`` is banker's rounding, which would send a rank
    of 12.5 to 12 and 13.5 to 14."""
    if denominator <= 0 or numerator < 0:
        raise ValueError("round_half_up needs numerator >= 0 and denominator > 0")
    return (2 * numerator + denominator) // (2 * denominator)


def round_half_up_float(value: float) -> int:
    """The single float-to-minor-units conversion (SF-06-build §3.3): half-up."""
    if value < 0:
        raise ValueError("amounts are never negative")
    return int(value + 0.5)


def percentile_of_sorted(amount_minor: int, ordered: Sequence[int]) -> int:
    """``percentile_of`` over a sample that is already sorted ascending. O(log n)."""
    n = len(ordered)
    if n == 0:
        raise ValueError("cannot rank a price in an empty sample")
    below = bisect_left(ordered, amount_minor)
    equal = bisect_right(ordered, amount_minor) - below
    # 100 * (below + 0.5 * equal) / n, kept in integers: (200*below + 100*equal) / (2n).
    return round_half_up(200 * below + 100 * equal, 2 * n)


def percentile_of(amount_minor: int, sample: pl.Series | Sequence[int]) -> int:
    """Percentile RANK of amount_minor within sample, as an int in [0, 100].

    Definition, pinned (L0 and SF-06 leave the direction implicit; SF-07's example payload
    'price_percentile: 34' + 'cheaper than 66% of the last year' fixes it):

        percentile = round(100 * (below + 0.5 * equal) / n)

    where `below` counts sample values strictly less than amount_minor and `equal` counts
    values equal to it. LOW MEANS CHEAP. A price cheaper than everything scores 0; a price
    dearer than everything scores 100. "Cheaper than X% of history" = 100 - percentile.
    `round` is half-up.

    n == 0 raises ValueError; callers check cell size first.
    """
    values = sample.to_list() if isinstance(sample, pl.Series) else list(sample)
    return percentile_of_sorted(amount_minor, sorted(values))
