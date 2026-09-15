"""Fixed analysis tools: the value-add layer over plain retrieval.

model_roi ranks frontier models by capability-per-dollar from
model_benchmarks. cost_efficiency_frontier finds the Pareto-optimal models on
capability vs cost.
"""

from __future__ import annotations

from typing import Any

from server import retrieval
from server.stats import to_float as _to_float

__all__ = [
    "model_roi",
    "cost_efficiency_frontier",
]


def model_roi(
    model_column: str = "name",
    score_column: str = "intelligence_index",
    price_column: str = "price_blended_3to1",
    limit: int = 50,
) -> dict[str, Any]:
    """Rank frontier models by capability-per-dollar (score_column / price_column).

    Reads model_benchmarks. Rows missing a score or price, or with a
    non-positive price, are skipped rather than distorting the ranking.

    Defaults match the current (premium AA) model_benchmarks schema. If your
    warehouse uses different names, call describe_dataset('model_benchmarks')
    and pass the correct ones via model_column/score_column/price_column. A
    name that doesn't exist raises a clear error rather than silently
    returning nothing.
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


def cost_efficiency_frontier(
    model_column: str = "name",
    score_column: str = "intelligence_index",
    cost_column: str = "cost_per_task_total",
    limit: int = 200,
) -> dict[str, Any]:
    """Pareto frontier of capability (score_column) vs cost (cost_column).

    Reads model_benchmarks. A model is "on the frontier" if no other model
    is both cheaper-or-equal AND at least as capable — i.e. nothing strictly
    dominates it. Deterministic: sort by cost ascending, keep every model
    whose score is a new best-so-far (a classic skyline scan), everything
    else is "dominated" (some cheaper-or-equal model already matches or
    beats its capability). Sharper than model_roi's single ratio because it
    doesn't force capability and cost onto one linear scale.

    Rows missing a score or cost, or with non-positive cost, are skipped.
    """
    data = retrieval.query_dataset(
        "model_benchmarks",
        columns=[model_column, score_column, cost_column],
        limit=limit,
    )

    candidates = []
    for row in data["rows"]:
        model = row.get(model_column)
        score = _to_float(row.get(score_column))
        cost = _to_float(row.get(cost_column))
        if model is None or score is None or cost is None or cost <= 0:
            continue
        candidates.append({"model": model, "score": score, "cost": cost})

    # Ties on cost: prefer the higher score first, so a same-cost model with
    # lower capability is correctly seen as dominated rather than sneaking
    # onto the frontier ahead of its better-scoring, same-priced peer.
    candidates.sort(key=lambda r: (r["cost"], -r["score"]))

    frontier: list[dict[str, Any]] = []
    dominated: list[dict[str, Any]] = []
    best_score_so_far = float("-inf")
    for row in candidates:
        if row["score"] > best_score_so_far:
            frontier.append(row)
            best_score_so_far = row["score"]
        else:
            dominated.append(row)

    return {
        "score_column": score_column,
        "cost_column": cost_column,
        "frontier": frontier,
        "dominated": dominated,
    }
