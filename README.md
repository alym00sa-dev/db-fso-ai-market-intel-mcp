# FSO AI Market Intelligence MCP Server

Custom MCP server, hosted as a Databricks App, exposing deterministic tools
over the `fso_market_intelligence.frontier_labs` Unity Catalog schema.

See `CLAUDE.md` for the full build plan.

## Deploy loop

1. `databricks apps deploy` (or push via the configured CI/CD path) to ship
   `app.yaml` + the `server/` package to the Databricks App.
2. Confirm the app is reachable at `<app-url>/mcp` (streamable HTTP transport).
3. Iterate: edit `server/`, redeploy, re-verify.
