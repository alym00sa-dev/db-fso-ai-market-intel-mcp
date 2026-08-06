# CLAUDE.md — Custom MCP Server for AI Market Intelligence

This file briefs Claude Code on what we're building and what to create. Read it
fully before generating files.

## What we're building

A **custom MCP server** hosted as a **Databricks App**. It exposes tools that let
an MCP client (Claude) query and analyze AI-market data in Unity Catalog tables.
It replaces what our Databricks Genie space does under the hood (querying our
tables) but as our own server with purpose-built, deterministic tools — no Genie,
no LLM writing SQL.

- Catalog: `fso_market_intelligence`
- Schema: `frontier_labs`
- Coverage: **ALL tables in the schema** (the full set below), not a subset.

## The full dataset (all 17 tables)

Hiring / frontier labs:
- `hiring_jobs` — individual job postings at frontier labs (title, category, status).
- `hiring_weekly` — weekly open/new/removed jobs per company, incl. social-impact roles.
- `hiring_weekly_breakdown` — weekly job counts per company by category/sub_area/theme/vertical.

Model capability & perception:
- `model_benchmarks` — benchmark results + metadata for frontier models (composite scores, pricing).
- `arena_agent_leaderboard` — LMArena Agent Arena overall scores/ranks for agentic models.
- `arena_agent_signals` — LMArena sub-dimension scores (steerability, tool hallucination, etc.).

African-NLP (SAHARA):
- `sahara_leaderboard` — overall and cluster scores on the SAHARA African-NLP benchmark.
- `sahara_scores` — granular SAHARA scores by model, task, and language.
- `sahara_task_scores` — model scores for specific SAHARA tasks.
- `sahara_language_scores` — model scores for individual African languages.

Government AI readiness (Oxford):
- `oxford_ai_readiness` — Oxford Insights Government AI Readiness Index scores/pillars per country.
- `oxford_readiness_dimensions` — Oxford readiness sub-dimension scores per country.

Responsible AI (GIRAI) + country reference:
- `girai_scores` — overall, dimension, and pillar GIRAI scores/rankings per country.
- `girai_pillar_dimension_scores` — GIRAI scores by country, pillar, and dimension.
- `girai_indicators` — GIRAI indicator-level scores per country and edition.
- `girai_editions_comparison` — GIRAI indicator status/initiatives compared 2024 vs 2026.
- `girai_country_classifications` — country reference data: region, income group, GDP, development status.

## Architecture (decided — do not re-litigate)

Three kinds of tools, all deterministic:

1. **Meta tools** — `list_datasets()`, `describe_dataset(dataset)`. Self-describe the
   full catalog so the client knows what exists. These must cover ALL 17 tables.
2. **Parameterized retrieval tools** — `query_dataset(dataset, columns, filters, limit)`.
   Works across ALL tables. Dynamic in *what* it returns; SQL built from a whitelist +
   parameters. NO free-form SQL, NO string interpolation of user input into queries.
3. **Fixed analysis tools** — `model_roi(...)`, `perception_vs_performance(...)`.
   These touch only the specific tables they need (see step 4). They are the
   value-add layer over plain retrieval.

Deferred (do NOT build yet): free-form text-to-SQL fallback. Genie already covers
open-ended exploration.

## Critical technical constraints

- **Transport MUST be streamable HTTP** (`transport="streamable-http"`), NOT stdio.
  Databricks Apps require this. #1 cause of a deploy that runs but can't be reached.
- App listens on port 8000 by default (`DATABRICKS_APP_PORT` env var).
- Data access via `databricks-sdk`:
  `WorkspaceClient().statement_execution.execute_statement(...)` against an attached
  **SQL Warehouse** (warehouse id from env var — do NOT hardcode).
- Full table names live in the whitelist; ids/creds come from env vars — never hardcode.
- Dependency management uses `uv`.

## Files to create

\`\`\`
.
├── app.yaml               # command: ['uv','run','custom-server']
├── requirements.txt       # uv, mcp, databricks-sdk
├── pyproject.toml         # [project.scripts] custom-server = "server.main:main"
├── .gitignore             # .env, __pycache__/, .DS_Store, .databricks/, .venv/
├── README.md              # short: what this is + the deploy loop
└── server/
    ├── __init__.py        # empty
    ├── main.py            # FastMCP init (streamable-http), registers all tools
    ├── db.py              # run_query(sql, params) via WorkspaceClient + statement_execution
    ├── datasets.py        # THE WHITELIST: all 17 tables -> {full table name, allowed columns, has_history}
    ├── retrieval.py       # query_dataset (uses datasets.py + db.py)
    └── analysis.py        # model_roi, perception_vs_performance
\`\`\`

## Build order (do this sequentially, test between steps)

1. **Skeleton first.** Create app.yaml, requirements.txt, pyproject.toml,
   .gitignore, server/__init__.py, and a minimal server/main.py with ONLY a
   health() tool. Goal: a deployable server proving the pipeline before any data
   logic. STOP here; the human deploys + verifies health works.
2. **Data layer.** Build datasets.py (the whitelist covering ALL 17 tables — the
   single source of truth for allowed tables/columns) and db.py (run_query, with
   structured logging of who/what/when for audit).
3. **Meta + retrieval tools.** list_datasets, describe_dataset (read the whitelist),
   and query_dataset (parameterized, validated against the whitelist). Works for all
   17 tables. Register in main.py.
4. **Analysis tools.** model_roi (capability-per-dollar from model_benchmarks) and
   perception_vs_performance (join arena_agent_leaderboard to model_benchmarks,
   flag over/under-rated). NOTE: arena and benchmark tables have NO shared key — model
   names differ ("Claude Fable 5 (High)" vs "Claude Fable 5 (Adaptive Reasoning, Max
   Effort...)"). Build an explicit name-normalization map in analysis.py. This
   reconciliation is real work and is WHY these are tools, not queries.

## Safety rules for query building

- Every dataset the client can touch must be in the datasets.py whitelist (all 17).
- columns args validated against the per-dataset allowlist before use.
- filters become parameterized WHERE clauses — never f-string user values into SQL.
- Fail closed: unknown dataset or column → clear error, not a guess.

## Getting column names for the whitelist

datasets.py needs the real columns per table. Two options:
- Run DESCRIBE fso_market_intelligence.frontier_labs.<table> for each table and
  record the columns, OR
- In describe_dataset, query INFORMATION_SCHEMA.COLUMNS at runtime so the server
  reads columns live instead of hardcoding them. (Cleaner, less to maintain — prefer
  this if straightforward.)

## Definition of "done" for today

Step 1 (health server) deploys green and is reachable at <app-url>/mcp. Everything
after that is additive. Do NOT try to build all four steps before the first deploy
succeeds. The Claude connector is created LATER by IT using the endpoint URL — not
part of today's scope.
