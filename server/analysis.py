"""Fixed analysis tools: the value-add layer over plain retrieval.

capability_cost_pareto finds the Pareto-optimal models on capability vs
per-1M-token price.
"""

from __future__ import annotations

from typing import Any

from server import retrieval
from server.stats import to_float as _to_float

__all__ = [
    "capability_cost_pareto",
]


def capability_cost_pareto(
    model_column: str = "name",
    score_column: str = "intelligence_index",
    cost_column: str = "price_blended_3to1",
    limit: int = 200,
) -> dict[str, Any]:
    """Pareto frontier of capability (score_column) vs cost (cost_column).

    Reads model_benchmarks. A model is "on the frontier" if no other model
    is both cheaper-or-equal AND at least as capable — i.e. nothing strictly
    dominates it. Deterministic: sort by cost ascending, keep every model
    whose score is a new best-so-far (a classic skyline scan) — this is the
    step-function "highest capability at or below this cost" frontier.
    Everything else is "dominated" (some cheaper-or-equal model already
    matches or beats its capability).

    score_column is the capability axis and can be any column in
    model_benchmarks (e.g. intelligence_index, coding_index, gpqa_diamond) —
    it's validated live against the warehouse schema, not hardcoded, so pass
    whatever capability metric the question is actually about.

    cost_column defaults to price_blended_3to1 — a genuine $/1M-tokens price
    (blended 3:1 input:output ratio) — and should stay a price_* column
    (price_input, price_output, price_blended_7to2to1 are also valid).
    Avoid cost_per_task_total here: that's AA's $ cost to run their eval
    suite, not a per-token price, and mixing the two would make the frontier
    incomparable across models.

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
