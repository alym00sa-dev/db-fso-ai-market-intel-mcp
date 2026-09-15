import os

from fastmcp import FastMCP

from server import analysis, compute_fit, equity, retrieval, scorecard

INSTRUCTIONS = """\
AI Market Intelligence over fso_market_intelligence.frontier_labs (16 datasets).
Three kinds of tools — use the cheapest one that answers the question:

1. Query (list_datasets, describe_dataset, query_dataset) — conversational access
   to any of the 16 datasets. Start here for "what does the data say about X".
2. Deterministic calculators (compute_fit, capability_cost_pareto,
   regional_equity_scores) — reproducible math. Same inputs always give the same
   answer.
3. Analyses (model_scorecard) — cross-source triangulation with deterministic
   joins/flags; treat the flags as a starting point for your own interpretation,
   not a final verdict.

Known caveats, always worth surfacing to the user:
- model_benchmarks is OVERWRITTEN DAILY (has_history=False) — use release_date as
  the time axis for any "over time" question, not load date.
- openrouter_rankings: filter is_aggregate=false for single-model reads; the
  is_aggregate=true rows are author/family rollups.
- ornn_gpu_prices / compute_fit only cover 5 GPUs (4 datacenter + 1 consumer RTX
  5090) — not the full hardware market, and compute_fit's self-host numbers are an
  upper-bound estimate at the stated precision (real usage with quantization will
  usually cost less).
- Closed-weight models (is_open_weights=false) have no self-host read — API cost
  only.

For the two flagship L3 reads, follow the orchestration playbooks in this repo's
recipes/ folder rather than improvising the structure: recipes/model_scorecard.md
(how to turn a model_scorecard() call into a stakeholder-ready read) and
recipes/global_equity_representation.md (how to weave regional_equity_scores +
SAHARA into a regional/language equity read — regional_equity_scores supplies the
deterministic readiness/responsibility-by-region join, but the language-coverage
and open-vs-closed matching steps are still a multi-call orchestration, not one tool).
"""

mcp = FastMCP("fso-ai-market-intel-mcp", instructions=INSTRUCTIONS)


@mcp.tool()
def health() -> str:
    """Report that the server is up and reachable."""
    return "ok"


mcp.tool(retrieval.list_datasets)
mcp.tool(retrieval.describe_dataset)
mcp.tool(retrieval.query_dataset)
mcp.tool(analysis.capability_cost_pareto)
mcp.tool(compute_fit.compute_fit)
mcp.tool(equity.regional_equity_scores)
mcp.tool(scorecard.model_scorecard)


def main() -> None:
    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
