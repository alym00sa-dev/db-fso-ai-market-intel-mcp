"""Small numeric helpers shared by analysis.py, compute_fit.py, and scorecard.py."""

from __future__ import annotations

from typing import Any


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_percentiles(rows: list[dict[str, Any]], value_key: str, percentile_key: str) -> None:
    """In place: add a 0-1 percentile rank for value_key, computed within `rows`."""
    n = len(rows)
    if n <= 1:
        for row in rows:
            row[percentile_key] = 0.5
        return
    ordered = sorted(range(n), key=lambda i: rows[i][value_key])
    for rank, idx in enumerate(ordered):
        rows[idx][percentile_key] = rank / (n - 1)
