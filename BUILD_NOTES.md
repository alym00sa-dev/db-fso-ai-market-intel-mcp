# Build notes — steps 2, 3, 4

Covers `server/datasets.py`, `server/db.py`, `server/retrieval.py`,
`server/analysis.py`, and the `main.py` registration. Read the "review
these first" section before trusting `model_roi` / `perception_vs_performance`
output.

## What each file does

**`server/datasets.py`** — The whitelist. A `Dataset` entry per table (all
17): dataset name, bare table name, category, description, and a
`has_history` flag. `full_name` builds the fully-qualified
`fso_market_intelligence.frontier_labs.<table>` name. Deliberately does
**not** hardcode column lists — see `retrieval.py`. `get_dataset()` is the
fail-closed lookup: unknown name raises `UnknownDatasetError` with a
pointer to `list_datasets()`.

**`server/db.py`** — `run_query(sql, params)`. Runs SQL against the SQL
Warehouse via `WorkspaceClient().statement_execution`, using named
parameters (`:name` placeholders bound via `StatementParameterListItem`) —
never string-interpolated values. Polls `get_statement` until the statement
leaves `PENDING`/`RUNNING` (60s cap). Warehouse id comes from the
`DATABRICKS_WAREHOUSE_ID` env var; missing var raises `QueryError`
immediately rather than guessing a default. Every call logs a structured
JSON line (`who` = resolved current-user identity, `what` = the SQL and
param *names* only — not values, to avoid putting filter values in logs —
plus row counts / errors, `when` = UTC timestamp).

**`server/retrieval.py`** — The meta + retrieval tools:
- `list_datasets()` — name/category/description/has_history for all 17.
- `describe_dataset(dataset)` — the above plus the live column list
  (name, type, comment), read from `INFORMATION_SCHEMA.COLUMNS` and cached
  in-process for 5 minutes.
- `query_dataset(dataset, columns, filters, limit)` — validates `dataset`
  against the whitelist and every `columns`/`filters` key against the live
  column list (fail closed, clear error naming the bad column). Filter
  *values* are always bound as query parameters; `filters` supports both
  equality (`{"col": v}`) and membership (`{"col": [v1, v2]}` → `IN (...)`).
  `limit` is clamped to [1, 1000].

**`server/analysis.py`** — The two fixed analysis tools:
- `model_roi(...)` — reads `model_benchmarks` via `query_dataset`, computes
  `score / price` per model, skips rows with a missing score/price or
  non-positive price, returns models ranked by capability-per-dollar.
- `perception_vs_performance(...)` — reads `arena_agent_leaderboard` and
  `model_benchmarks`, joins them by `normalize_model_name()` (lowercase,
  collapse whitespace, strip a trailing `(...)` suffix, then check
  `MODEL_NAME_OVERRIDES` for anything the heuristic can't reconcile), then
  ranks matched models by percentile on each side and flags a model
  `"overrated"` if its arena percentile exceeds its benchmark percentile by
  more than `overrated_threshold` (default 0.15), `"underrated"` for the
  reverse gap, else `"in_line"`. Unmatched arena models are returned
  separately rather than silently dropped.

**`server/main.py`** — Unchanged in spirit: still just imports and
registers. Added `mcp.tool(retrieval.list_datasets)`,
`describe_dataset`, `query_dataset`, `analysis.model_roi`,
`analysis.perception_vs_performance` alongside the existing `health`.

## Verified locally (no warehouse access from here)

I don't have Databricks credentials in this environment, so I could not
run a real query end-to-end. What I *did* verify, against the real
`databricks-sdk` and `fastmcp` packages in a throwaway virtualenv (never
against your actual warehouse):

- Every file imports cleanly; `datasets.py` has exactly 17 entries.
- `server/main.py` boots as a real streamable-HTTP server and responds
  correctly to an MCP `initialize` handshake, with all 6 tools
  (`health`, `list_datasets`, `describe_dataset`, `query_dataset`,
  `model_roi`, `perception_vs_performance`) registered and their
  docstrings intact.
- `databricks-sdk`'s actual `StatementResponse`/`StatementStatus`/
  `ResultManifest`/`ResultSchema`/`ColumnInfo`/`StatementParameterListItem`
  field names match what `db.py` uses (I inspected the SDK source rather
  than guessing).
- `analysis.normalize_model_name()` correctly unifies the exact example
  from CLAUDE.md — `"Claude Fable 5 (High)"` and `"Claude Fable 5 (Adaptive
  Reasoning, Max Effort, Extended Thinking)"` both normalize to
  `"claude fable 5"`.
- `model_roi` / `perception_vs_performance`'s row-level logic (skipping
  bad rows, percentile math, flagging) tested against hand-built fake rows
  standing in for `query_dataset`'s output.

## What you need to configure in Databricks before these work

1. **Attach a SQL Warehouse** to the app and set `DATABRICKS_WAREHOUSE_ID`
   in the app's env vars to that warehouse's id. `db.py` raises a clear
   `QueryError` on startup-of-first-query if this is missing — it will not
   silently pick a default warehouse.
2. **Grants**: the app's identity (its service principal, or whoever it
   runs as) needs `USE CATALOG` on `fso_market_intelligence`, `USE SCHEMA`
   on `frontier_labs`, `SELECT` on all 17 tables, and — importantly —
   enough access to query `fso_market_intelligence.information_schema.columns`
   (this is what `describe_dataset` and every `query_dataset` call use to
   get the live column list; without it, every dataset will fail with
   "INFORMATION_SCHEMA returned no columns").
3. The audit logging in `db.py` calls `WorkspaceClient().current_user.me()`
   to resolve a "who" for its structured logs. This is best-effort (it
   falls back to `"unknown"` and logs a warning rather than failing the
   query) but for it to resolve a real identity, the app's identity needs
   whatever minimal Users-API permission `current_user.me()` requires in
   your workspace.

## Review these first — genuine unknowns

I do not have access to your actual warehouse schema, so two things in
here are informed guesses rather than verified facts. Both are designed to
fail closed with an actionable error if wrong, not to fail silently:

1. **`has_history` flags in `datasets.py`.** CLAUDE.md doesn't define what
   "has history" means precisely. I set it `True` for tables that are
   explicitly weekly/time-series or multi-edition by description
   (`hiring_weekly`, `hiring_weekly_breakdown`, `model_benchmarks` — daily
   pulls — and the four GIRAI tables that carry an edition dimension), and
   `False` for what read as point-in-time snapshots (job postings, arena/
   SAHARA/Oxford tables, country reference data). Worth a quick sanity
   check against the real tables.

2. **Column-name defaults in `analysis.py`.** `model_roi` assumes
   `model_benchmarks` has columns named `model_name`, `composite_score`,
   `price_per_1m_tokens_blended`. `perception_vs_performance` assumes
   `arena_agent_leaderboard` has `model_name`/`score` and reuses the same
   `model_benchmarks` assumptions. These are reasonable guesses given
   CLAUDE.md's prose descriptions, but I could not confirm them against
   the real schema. If wrong, calling either tool will raise
   `UnknownColumnError` naming exactly which column doesn't exist — at
   that point call `describe_dataset('model_benchmarks')` /
   `describe_dataset('arena_agent_leaderboard')` for the real names and
   pass them in via the tools' `model_column`/`score_column`/
   `price_column`/`arena_score_column`/etc. arguments (all overridable,
   on purpose, for exactly this reason).

3. **`MODEL_NAME_OVERRIDES` in `analysis.py` is empty.** The
   parenthetical-stripping heuristic handles the one example CLAUDE.md
   gives, but I have no real data to check it against every model in both
   tables. Once you run `perception_vs_performance` for real, check
   `unmatched_arena_models` in the result — any model that should have
   matched but didn't means its benchmark-table name doesn't reduce to the
   same key, and needs an entry added to `MODEL_NAME_OVERRIDES`.
