"""Runs parameterized SQL against the attached Databricks SQL Warehouse.

This module only knows how to bind *values* safely, via the Statement
Execution API's named parameters (``:name`` placeholders). Table and column
identifiers must already be validated against the server/datasets.py
whitelist by the caller before they reach ``run_query`` — this module does
not (and cannot) sanitize identifiers, only parameter values.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    Disposition,
    Format,
    StatementParameterListItem,
    StatementState,
)

logger = logging.getLogger("fso_ai_market_intel_mcp.db")

_POLL_INTERVAL_SECONDS = 1.0
_MAX_WAIT_SECONDS = 60.0
_WAREHOUSE_ID_ENV_VAR = "DATABRICKS_WAREHOUSE_ID"


class QueryError(RuntimeError):
    """Raised when a warehouse query fails, times out, or can't be configured."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]


@lru_cache(maxsize=1)
def _client() -> WorkspaceClient:
    return WorkspaceClient()


@lru_cache(maxsize=1)
def _actor() -> str:
    """Best-effort identity for audit logging. Never blocks a query on failure."""
    try:
        return _client().current_user.me().user_name or "unknown"
    except Exception:
        logger.warning("could not resolve current_user for audit logging", exc_info=True)
        return "unknown"


def _warehouse_id() -> str:
    warehouse_id = os.environ.get(_WAREHOUSE_ID_ENV_VAR)
    if not warehouse_id:
        raise QueryError(
            f"{_WAREHOUSE_ID_ENV_VAR} environment variable is not set. "
            "Attach a SQL Warehouse to this app and set the env var to its id."
        )
    return warehouse_id


def _param_type(value: Any) -> str:
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "BIGINT"
    if isinstance(value, float):
        return "DOUBLE"
    return "STRING"


def _to_statement_parameters(
    params: dict[str, Any] | None,
) -> list[StatementParameterListItem]:
    if not params:
        return []
    return [
        StatementParameterListItem(
            name=name,
            type=_param_type(value),
            value=None if value is None else str(value),
        )
        for name, value in params.items()
    ]


def _log(event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "when": datetime.now(timezone.utc).isoformat(),
        "who": _actor(),
        **fields,
    }
    logger.info(json.dumps(payload, default=str))


def run_query(sql: str, params: dict[str, Any] | None = None) -> QueryResult:
    """Run a parameterized SQL statement against the attached SQL Warehouse.

    `sql` must use named placeholders (`:name`) for every value supplied via
    `params` — never format user-controlled values directly into `sql`.
    """
    warehouse_id = _warehouse_id()
    client = _client()
    statement_parameters = _to_statement_parameters(params)

    _log("query_start", sql=sql, param_names=list((params or {}).keys()))

    response = client.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        parameters=statement_parameters,
        disposition=Disposition.INLINE,
        format=Format.JSON_ARRAY,
        wait_timeout="30s",
    )

    statement_id = response.statement_id
    deadline = time.monotonic() + _MAX_WAIT_SECONDS
    while response.status and response.status.state in (
        StatementState.PENDING,
        StatementState.RUNNING,
    ):
        if time.monotonic() > deadline:
            _log("query_timeout", statement_id=statement_id)
            raise QueryError(
                f"Query timed out after {_MAX_WAIT_SECONDS}s (statement_id={statement_id})"
            )
        time.sleep(_POLL_INTERVAL_SECONDS)
        response = client.statement_execution.get_statement(statement_id)

    if not response.status or response.status.state != StatementState.SUCCEEDED:
        error_message = (
            response.status.error.message
            if response.status and response.status.error
            else "unknown error"
        )
        _log("query_failed", statement_id=statement_id, error=error_message)
        raise QueryError(f"Query failed: {error_message}")

    columns: list[str] = []
    if response.manifest and response.manifest.schema and response.manifest.schema.columns:
        columns = [c.name for c in response.manifest.schema.columns]

    rows: list[list[Any]] = []
    if response.result and response.result.data_array:
        rows = response.result.data_array

    _log("query_success", statement_id=statement_id, row_count=len(rows))

    return QueryResult(columns=columns, rows=rows)
