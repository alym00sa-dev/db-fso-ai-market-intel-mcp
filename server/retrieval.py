"""Meta and parameterized retrieval tools over the datasets.py whitelist.

These are the tools an MCP client uses to discover what's queryable
(list_datasets, describe_dataset) and to pull rows back (query_dataset).
No free-form SQL: every table comes from the static whitelist, every column
is checked against the table's live column list before use, and every
filter *value* is bound as a query parameter — never interpolated into SQL.
"""

from __future__ import annotations

import time
from typing import Any

from server import db
from server.datasets import CATALOG, SCHEMA, DATASETS, Dataset, get_dataset

DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
_COLUMN_CACHE_TTL_SECONDS = 300.0

_column_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


class UnknownColumnError(Exception):
    """Raised when a requested column isn't in a dataset's live schema."""


def _quote_ident(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def _get_columns(dataset: Dataset) -> list[dict[str, Any]]:
    """Live column list for `dataset`, from INFORMATION_SCHEMA.COLUMNS.

    Columns aren't hardcoded in datasets.py so this can't drift out of sync
    with the warehouse schema. Cached briefly in-process to avoid a round
    trip on every call.
    """
    cached = _column_cache.get(dataset.name)
    now = time.monotonic()
    if cached and now - cached[0] < _COLUMN_CACHE_TTL_SECONDS:
        return cached[1]

    result = db.run_query(
        f"""
        SELECT column_name, data_type, comment
        FROM {_quote_ident(CATALOG)}.information_schema.columns
        WHERE table_schema = :table_schema AND table_name = :table_name
        ORDER BY ordinal_position
        """,
        {"table_schema": SCHEMA, "table_name": dataset.table},
    )
    columns = [
        {"name": row["column_name"], "type": row["data_type"], "comment": row["comment"]}
        for row in result.as_dicts()
    ]
    if not columns:
        raise UnknownColumnError(
            f"INFORMATION_SCHEMA returned no columns for '{dataset.full_name}'. "
            "Check that the table exists and this app's warehouse identity has "
            "SELECT/USE grants on it."
        )
    _column_cache[dataset.name] = (now, columns)
    return columns


def list_datasets() -> list[dict[str, Any]]:
    """List every dataset this server can query.

    Returns each dataset's name, category, description, and whether it
    carries time-series history (vs. a point-in-time snapshot). Call this
    first, then describe_dataset(name) for column-level detail, or
    query_dataset(name, ...) to pull rows.
    """
    return [
        {
            "name": d.name,
            "category": d.category,
            "description": d.description,
            "has_history": d.has_history,
        }
        for d in DATASETS.values()
    ]


def describe_dataset(dataset: str) -> dict[str, Any]:
    """Describe one dataset: its full table name, history flag, and live columns.

    `dataset` must be one of the names returned by list_datasets() — anything
    else raises a clear error rather than guessing. The column list (name,
    type, comment) is read live from the warehouse, so it always matches
    what's actually queryable right now.
    """
    ds = get_dataset(dataset)
    return {
        "name": ds.name,
        "table": ds.full_name,
        "category": ds.category,
        "description": ds.description,
        "has_history": ds.has_history,
        "columns": _get_columns(ds),
    }


def query_dataset(
    dataset: str,
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Retrieve rows from one whitelisted dataset.

    Args:
        dataset: One of the names from list_datasets().
        columns: Subset of column names to return; omit or pass an empty
            list for all columns. Every name is checked against the
            dataset's live column list — unknown columns raise an error
            rather than being silently dropped.
        filters: Equality filters as {column: value}, or membership filters
            as {column: [v1, v2, ...]}. Values are always bound as query
            parameters, never interpolated into SQL.
        limit: Max rows to return (default 100, capped at 1000).

    Returns:
        {"dataset": name, "columns": [...], "rows": [{column: value, ...}, ...]}
    """
    ds = get_dataset(dataset)
    available = _get_columns(ds)
    available_names = {c["name"] for c in available}

    if columns:
        unknown = [c for c in columns if c not in available_names]
        if unknown:
            raise UnknownColumnError(
                f"Unknown column(s) {unknown} for dataset '{dataset}'. "
                f"Call describe_dataset('{dataset}') for the valid column list."
            )
        selected_columns = list(columns)
    else:
        selected_columns = [c["name"] for c in available]

    select_clause = ", ".join(_quote_ident(c) for c in selected_columns)

    where_clauses: list[str] = []
    params: dict[str, Any] = {}
    for i, (column, value) in enumerate((filters or {}).items()):
        if column not in available_names:
            raise UnknownColumnError(
                f"Unknown filter column '{column}' for dataset '{dataset}'. "
                f"Call describe_dataset('{dataset}') for the valid column list."
            )
        if isinstance(value, (list, tuple)):
            placeholders = []
            for j, v in enumerate(value):
                param_name = f"f{i}_{j}"
                params[param_name] = v
                placeholders.append(f":{param_name}")
            where_clauses.append(f"{_quote_ident(column)} IN ({', '.join(placeholders)})")
        else:
            param_name = f"f{i}"
            params[param_name] = value
            where_clauses.append(f"{_quote_ident(column)} = :{param_name}")

    safe_limit = max(1, min(int(limit), MAX_LIMIT))

    sql = (
        f"SELECT {select_clause} FROM "
        f"{_quote_ident(CATALOG)}.{_quote_ident(SCHEMA)}.{_quote_ident(ds.table)}"
    )
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += f" LIMIT {safe_limit}"

    result = db.run_query(sql, params)
    return {
        "dataset": ds.name,
        "columns": result.columns,
        "rows": result.as_dicts(),
    }
