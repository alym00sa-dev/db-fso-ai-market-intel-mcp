"""Fixed analysis tools: the value-add layer over plain retrieval.

model_roi ranks frontier models by capability-per-dollar from
model_benchmarks. perception_vs_performance reconciles
arena_agent_leaderboard (user perception) with model_benchmarks (measured
capability) to flag models the market rates differently than benchmarks do.

arena_agent_leaderboard and model_benchmarks share no key: the same model
shows up under different display names in each table (e.g. arena's "Claude
Fable 5 (High)" vs benchmarks' "Claude Fable 5 (Adaptive Reasoning, Max
Effort...)"). MODEL_NAME_OVERRIDES + normalize_model_name() below do that
reconciliation explicitly rather than relying on an exact string match.
"""

from __future__ import annotations

import re
from typing import Any

from server import retrieval

# Exact-match overrides for model name pairs normalize_model_name()'s
# heuristic can't reconcile on its own (i.e. names that differ by more than
# a parenthetical suffix). Populate this as real mismatches turn up when
# running perception_vs_performance against live data. Keys and values
# should both already be lowercased/whitespace-collapsed, matching what
# normalize_model_name() produces before the override lookup.
MODEL_NAME_OVERRIDES: dict[str, str] = {}


def normalize_model_name(raw_name: str | None) -> str:
    """Reduce a model's display name to a canonical key for joining across tables.

    Lowercases, strips whitespace, drops any trailing parenthetical suffix
    (reasoning effort, mode, thinking budget, etc. — the part that tends to
    vary between arena and benchmark naming), then applies
    MODEL_NAME_OVERRIDES for pairs the heuristic can't reconcile.
    """
    if not raw_name:
        return ""
    name = raw_name.strip().lower()
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return MODEL_NAME_OVERRIDES.get(name, name)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_percentiles(rows: list[dict[str, Any]], value_key: str, percentile_key: str) -> None:
    """In place: add a 0-1 percentile rank for value_key, computed within `rows`."""
    n = len(rows)
    if n <= 1:
        for row in rows:
            row[percentile_key] = 0.5
        return
    ordered = sorted(range(n), key=lambda i: rows[i][value_key])
    for rank, idx in enumerate(ordered):
        rows[idx][percentile_key] = rank / (n - 1)


def model_roi(
    model_column: str = "model_name",
    score_column: str = "composite_score",
    price_column: str = "price_per_1m_tokens_blended",
    limit: int = 50,
) -> dict[str, Any]:
    """Rank frontier models by capability-per-dollar (score_column / price_column).

    Reads model_benchmarks. Rows missing a score or price, or with a
    non-positive price, are skipped rather than distorting the ranking.

    The default column names are a best guess — if model_benchmarks uses
    different names in your warehouse, call
    describe_dataset('model_benchmarks') and pass the correct ones via
    model_column/score_column/price_column. A name that doesn't exist
    raises a clear error rather than silently returning nothing.
    """
    data = retrieval.query_dataset(
        "model_benchmarks",
        columns=[model_column, score_column, price_column],
        limit=limit,
    )

    ranked = []
    for row in data["rows"]:
        model = row.get(model_column)
        score = _to_float(row.get(score_column))
        price = _to_float(row.get(price_column))
        if model is None or score is None or price is None or price <= 0:
            continue
        ranked.append(
            {
                "model": model,
                "score": score,
                "price": price,
                "capability_per_dollar": score / price,
            }
        )

    ranked.sort(key=lambda r: r["capability_per_dollar"], reverse=True)
    return {
        "score_column": score_column,
        "price_column": price_column,
        "ranking": ranked,
    }


def perception_vs_performance(
    arena_model_column: str = "model_name",
    arena_score_column: str = "score",
    benchmark_model_column: str = "model_name",
    benchmark_score_column: str = "composite_score",
    limit: int = 200,
    overrated_threshold: float = 0.15,
) -> dict[str, Any]:
    """Compare user perception (arena) against measured capability (benchmarks).

    Joins arena_agent_leaderboard to model_benchmarks by normalized model
    name (see normalize_model_name) since the two tables share no key and
    use different display-name conventions for the same model. For each
    matched model, computes its percentile rank within the matched set on
    both arena score and benchmark score, then flags models whose arena
    percentile exceeds its benchmark percentile by more than
    `overrated_threshold` as "overrated" (the market rates it above what
    benchmarks support), and the reverse as "underrated".

    The default column names are a best guess for each table's schema — if
    wrong, call describe_dataset('arena_agent_leaderboard') /
    describe_dataset('model_benchmarks') and pass the correct names.
    """
    arena = retrieval.query_dataset(
        "arena_agent_leaderboard",
        columns=[arena_model_column, arena_score_column],
        limit=limit,
    )
    benchmarks = retrieval.query_dataset(
        "model_benchmarks",
        columns=[benchmark_model_column, benchmark_score_column],
        limit=limit,
    )

    benchmark_by_key: dict[str, dict[str, Any]] = {}
    for row in benchmarks["rows"]:
        key = normalize_model_name(row.get(benchmark_model_column))
        score = _to_float(row.get(benchmark_score_column))
        if not key or score is None:
            continue
        benchmark_by_key[key] = {"model": row.get(benchmark_model_column), "score": score}

    matched = []
    unmatched_arena_models = []
    for row in arena["rows"]:
        key = normalize_model_name(row.get(arena_model_column))
        arena_score = _to_float(row.get(arena_score_column))
        if not key or arena_score is None:
            continue
        bench = benchmark_by_key.get(key)
        if bench is None:
            unmatched_arena_models.append(row.get(arena_model_column))
            continue
        matched.append(
            {
                "model_key": key,
                "arena_model_name": row.get(arena_model_column),
                "benchmark_model_name": bench["model"],
                "arena_score": arena_score,
                "benchmark_score": bench["score"],
            }
        )

    _add_percentiles(matched, "arena_score", "arena_percentile")
    _add_percentiles(matched, "benchmark_score", "benchmark_percentile")

    for row in matched:
        gap = row["arena_percentile"] - row["benchmark_percentile"]
        row["perception_gap"] = gap
        if gap > overrated_threshold:
            row["flag"] = "overrated"
        elif gap < -overrated_threshold:
            row["flag"] = "underrated"
        else:
            row["flag"] = "in_line"

    matched.sort(key=lambda r: r["perception_gap"], reverse=True)

    return {
        "matched": matched,
        "unmatched_arena_models": unmatched_arena_models,
    }
