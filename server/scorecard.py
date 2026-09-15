"""model_scorecard — flagship Level-3 triangulation tool (hybrid).

Joins three independent signals about the same model — measured capability
(model_benchmarks), stated preference (arena_agent_leaderboard), revealed
usage (openrouter_rankings) — plus cost, and flags the gaps between them
(underadopted / hyped / economical). The join, percentiles, and gap-flagging
are Python because they must be reproducible; the narrative "so what" is left
to Claude / an MD recipe that calls this tool and interprets the card (see
recipes/model_scorecard.md). Full spec: planning/03-model-scorecard.md.

Self-host cost never does its own VRAM math — it delegates to
server.compute_fit.compute_fit(), per planning/05-compute-fit-calculator.md.
"""

from __future__ import annotations

from typing import Any

from server import retrieval
from server.compute_fit import ComputeFitError, compute_fit
from server.reconciliation import index_by_key, normalize_model_name
from server.stats import add_percentiles, to_float

_PEER_LIMIT = 500
# refresh_benchmarks.ipynb pulls evaluations dynamically from the AA Pro API, so
# the exact live eval column names aren't fixed in code anywhere — these are
# plausible candidates (some confirmed live via describe_dataset, some carried
# over from the pre-premium schema in genie_context.md as a fallback). Only
# whichever of these actually exist get selected (see _measured_signal).
_KEY_EVAL_CANDIDATES = [
    "intelligence_index",
    "gpqa_diamond",
    "gpqa",
    "hle",
    "mmlu_pro",
    "math_500",
    "livecodebench",
    "coding_index",
    "math_index",
]
# "org" confirmed live (refresh_benchmarks.ipynb) — NOT "organization".
_IDENTITY_CANDIDATES = ["name", "slug", "org", "params_total_b", "release_date", "is_open_weights"]
_COST_CANDIDATES = ["price_blended_3to1", "cost_per_task_total"]
_GAP_THRESHOLD = 0.15


class ModelScorecardError(ValueError):
    """Raised when the requested model can't be found in model_benchmarks at all."""


def _available_columns(dataset: str) -> set[str]:
    return {c["name"] for c in retrieval.describe_dataset(dataset)["columns"]}


def _measured_signal(model: str) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Returns (matched benchmark row, full peer row set, canonical model key)."""
    available = _available_columns("model_benchmarks")
    identity_cols = [c for c in _IDENTITY_CANDIDATES if c in available]
    eval_cols = [c for c in _KEY_EVAL_CANDIDATES if c in available]
    cost_cols = [c for c in _COST_CANDIDATES if c in available]
    columns = list(dict.fromkeys(identity_cols + eval_cols + cost_cols))

    peers = retrieval.query_dataset("model_benchmarks", columns=columns, limit=_PEER_LIMIT)["rows"]
    name_col = "name" if "name" in available else identity_cols[0]
    by_key = index_by_key(peers, name_col)

    key = normalize_model_name(model)
    row = by_key.get(key)
    if row is None and "slug" in available:
        by_slug_key = index_by_key(peers, "slug")
        row = by_slug_key.get(key)
    if row is None:
        raise ModelScorecardError(
            f"'{model}' not found in model_benchmarks (checked name and slug, "
            f"normalized). Call query_dataset('model_benchmarks', ...) to find the "
            "right identifier."
        )
    return row, peers, key


def _stated_signal(model_key: str) -> dict[str, Any]:
    # NOTE: arena_agent_leaderboard's model column is "model", not "model_name" —
    # load_arena_agent.ipynb renames the raw HF dataset's model_name -> model on
    # load. Confirmed against both that notebook and genie_context.md.
    available = _available_columns("arena_agent_leaderboard")
    if "model" not in available or "score" not in available:
        return {"available": False, "reason": "arena_agent_leaderboard schema unrecognized"}

    arena_rows = retrieval.query_dataset(
        "arena_agent_leaderboard", columns=["model", "score"], limit=_PEER_LIMIT
    )["rows"]
    scored = [r for r in arena_rows if to_float(r.get("score")) is not None]
    for r in scored:
        r["score"] = to_float(r["score"])
    add_percentiles(scored, "score", "percentile")

    by_key = index_by_key(scored, "model")
    row = by_key.get(model_key)
    if row is None:
        return {"available": False, "reason": "not on arena"}
    return {
        "available": True,
        "arena_model_name": row["model"],
        "score": row["score"],
        "percentile": row["percentile"],
    }


def _revealed_signal(model_key: str) -> dict[str, Any]:
    available = _available_columns("openrouter_rankings")
    required = {"date", "model_permaslug", "total_tokens", "is_aggregate", "rank"}
    if not required.issubset(available):
        return {"available": False, "reason": "openrouter_rankings schema unrecognized"}

    latest = retrieval.query_dataset(
        "openrouter_rankings",
        columns=["date"],
        filters={"is_aggregate": False},
        order_by="date",
        descending=True,
        limit=1,
    )["rows"]
    if not latest:
        return {"available": False, "reason": "not tracked on OpenRouter"}
    latest_date = latest[0]["date"]

    day_rows = retrieval.query_dataset(
        "openrouter_rankings",
        columns=["model_permaslug", "total_tokens", "rank"],
        filters={"date": latest_date, "is_aggregate": False},
        order_by="rank",
        limit=retrieval.MAX_LIMIT,
    )["rows"]
    by_key = index_by_key(day_rows, "model_permaslug", is_permaslug=True)
    row = by_key.get(model_key)
    if row is None:
        return {"available": False, "reason": "not tracked on OpenRouter"}

    total_ranked = len(day_rows)
    percentile = 1.0 - ((row["rank"] - 1) / (total_ranked - 1)) if total_ranked > 1 else 0.5

    history = retrieval.query_dataset(
        "openrouter_rankings",
        columns=["date", "model_permaslug", "rank"],
        filters={"model_permaslug": row["model_permaslug"], "is_aggregate": False},
        order_by="date",
        descending=True,
        limit=14,
    )["rows"]
    trend = "flat"
    if len(history) > 1:
        oldest_rank, newest_rank = history[-1]["rank"], history[0]["rank"]
        if newest_rank < oldest_rank:
            trend = "up"
        elif newest_rank > oldest_rank:
            trend = "down"

    return {
        "available": True,
        "as_of_date": latest_date,
        "model_permaslug": row["model_permaslug"],
        "rank": row["rank"],
        "total_tokens": row["total_tokens"],
        "total_ranked_that_day": total_ranked,
        "percentile": percentile,
        "recent_trend": trend,
        "trend_window_days": len(history),
    }


def _cost_and_selfhost(benchmark_row: dict[str, Any], resolved_name: str) -> dict[str, Any]:
    cost = {
        "price_blended_3to1": to_float(benchmark_row.get("price_blended_3to1")),
        "cost_per_task_total": to_float(benchmark_row.get("cost_per_task_total")),
    }
    is_open_weights = benchmark_row.get("is_open_weights")
    if not is_open_weights:
        cost["self_host"] = {
            "self_hostable": False,
            "reason": "Model is closed-weight (API-only) — self-host isn't an option.",
        }
        return cost
    try:
        cost["self_host"] = compute_fit(model=resolved_name)
    except ComputeFitError as exc:
        cost["self_host"] = {"self_hostable": None, "reason": str(exc)}
    return cost


def model_scorecard(model: str) -> dict[str, Any]:
    """One model card triangulating measured capability + stated preference + revealed
    usage + cost for `model` (any recognizable name or slug).

    Measured capability and its percentile come from model_benchmarks, scored
    against all currently-benchmarked models as the peer set. Stated
    preference comes from arena_agent_leaderboard (or "not on arena" if
    absent). Revealed usage comes from the most recent openrouter_rankings
    snapshot, excluding is_aggregate rollups (or "not tracked on OpenRouter"
    if absent). Cost includes price_blended_3to1/cost_per_task_total, plus a
    self-host VRAM/GPU-fit read (via compute_fit) if the model is
    open-weight.

    Flags (underadopted / hyped / economical) are derived deterministically
    from percentile gaps between signals, and only computed for signal pairs
    that are both available.
    """
    benchmark_row, peers, model_key = _measured_signal(model)
    resolved_name = benchmark_row.get("name") or benchmark_row.get("slug") or model

    scored_peers = [p for p in peers if to_float(p.get("intelligence_index")) is not None]
    for p in scored_peers:
        p["intelligence_index"] = to_float(p["intelligence_index"])
    add_percentiles(scored_peers, "intelligence_index", "percentile")
    measured_percentile = next(
        (p["percentile"] for p in scored_peers if normalize_model_name(p.get("name")) == model_key),
        None,
    )

    key_evals = {
        col: to_float(benchmark_row[col]) for col in _KEY_EVAL_CANDIDATES if col in benchmark_row
    }

    stated = _stated_signal(model_key)
    revealed = _revealed_signal(model_key)
    cost = _cost_and_selfhost(benchmark_row, resolved_name)

    flags: list[str] = []
    gaps: dict[str, float | None] = {"stated_vs_measured": None, "revealed_vs_measured": None}
    if measured_percentile is not None:
        if stated.get("available"):
            gap = stated["percentile"] - measured_percentile
            gaps["stated_vs_measured"] = gap
            if gap > _GAP_THRESHOLD:
                flags.append("hyped")
        if revealed.get("available"):
            gap = revealed["percentile"] - measured_percentile
            gaps["revealed_vs_measured"] = gap
            if gap < -_GAP_THRESHOLD:
                flags.append("underadopted")
            elif gap > _GAP_THRESHOLD and measured_percentile < 0.7:
                flags.append("economical")

    return {
        "identity": {
            "name": benchmark_row.get("name"),
            "slug": benchmark_row.get("slug"),
            "organization": benchmark_row.get("org"),
            "is_open_weights": benchmark_row.get("is_open_weights"),
            "release_date": benchmark_row.get("release_date"),
            "params_total_b": to_float(benchmark_row.get("params_total_b")),
        },
        "measured": {
            "intelligence_index": to_float(benchmark_row.get("intelligence_index")),
            "percentile": measured_percentile,
            "peer_set_size": len(scored_peers),
            "key_evals": key_evals,
        },
        "stated": stated,
        "revealed": revealed,
        "cost": cost,
        "gaps": gaps,
        "flags": flags,
    }
