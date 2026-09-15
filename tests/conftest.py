"""Shared fixtures for the MCP unit test suite.

There's no live Databricks warehouse in this dev environment (no CLI auth, no
warehouse id) — these tests never hit a real one. Instead they stub the
boundary at two levels, matching the two things worth testing separately:

- `patch_get_columns` stubs retrieval._get_columns (the INFORMATION_SCHEMA
  lookup) so retrieval.query_dataset's own SQL-building/validation logic runs
  for real and can be asserted on.
- `stub_query_dataset` stubs retrieval.query_dataset itself, for testing the
  tools built on top of it (analysis, compute_fit, scorecard) without also
  re-testing retrieval's SQL construction in every one of those tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from server import retrieval

# Column shapes matching the real (premium AA rebuild) schema described in
# planning/00-overview.md and planning/01-sync-fixes.md.
FAKE_COLUMNS: dict[str, list[str]] = {
    "model_benchmarks": [
        "name",
        "slug",
        "organization",
        "params_total_b",
        "context_window_tokens",
        "release_date",
        "is_open_weights",
        "intelligence_index",
        "gpqa_diamond",
        "hle",
        "price_blended_3to1",
        "cost_per_task_total",
    ],
    "arena_agent_leaderboard": ["model_name", "score"],
    "openrouter_rankings": [
        "date",
        "model_permaslug",
        "total_tokens",
        "is_aggregate",
        "period",
        "rank",
        "captured_at",
    ],
    "ornn_gpu_prices": [
        "date",
        "gpu_name",
        "index_value_usd_hr",
        "source",
        "source_updated_at",
        "captured_at",
    ],
}


def _columns_payload(names: list[str]) -> list[dict[str, Any]]:
    return [{"name": n, "type": "string", "comment": None} for n in names]


@pytest.fixture
def patch_get_columns(monkeypatch):
    """Make retrieval._get_columns return FAKE_COLUMNS (or a caller override)."""

    def _apply(overrides: dict[str, list[str]] | None = None):
        columns = {**FAKE_COLUMNS, **(overrides or {})}

        def fake_get_columns(dataset):
            return _columns_payload(columns[dataset.name])

        monkeypatch.setattr(retrieval, "_get_columns", fake_get_columns)
        return columns

    return _apply


@pytest.fixture
def stub_query_dataset(monkeypatch):
    """Patch a module's `retrieval.query_dataset` with a routing function.

    Usage: stub_query_dataset(analysis, {"model_benchmarks": {"rows": [...]}})
    routes to a dict of {dataset_name: fixed_response} or {dataset_name: callable}.
    """

    def _apply(target_module, routes: dict[str, Any]):
        def fake_query_dataset(
            dataset, columns=None, filters=None, limit=100, order_by=None, descending=False
        ):
            route = routes[dataset]
            if callable(route):
                return route(
                    dataset=dataset,
                    columns=columns,
                    filters=filters,
                    limit=limit,
                    order_by=order_by,
                    descending=descending,
                )
            return route

        monkeypatch.setattr(target_module.retrieval, "query_dataset", fake_query_dataset)

    return _apply


@pytest.fixture
def stub_describe_dataset(monkeypatch):
    """Patch a module's `retrieval.describe_dataset` to return fixed column names."""

    def _apply(target_module, columns_by_dataset: dict[str, list[str]]):
        def fake_describe_dataset(dataset):
            return {"columns": _columns_payload(columns_by_dataset[dataset])}

        monkeypatch.setattr(target_module.retrieval, "describe_dataset", fake_describe_dataset)

    return _apply


@pytest.fixture
def fake_db_run_query(monkeypatch):
    """Patch server.db.run_query to return a canned QueryResult-shaped object."""
    from server import db

    def _apply(columns: list[str], rows: list[list[Any]]):
        result = db.QueryResult(columns=columns, rows=rows)
        monkeypatch.setattr(db, "run_query", lambda sql, params=None: result)
        return result

    return _apply
