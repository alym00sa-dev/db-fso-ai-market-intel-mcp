import os

from fastmcp import FastMCP

from server import analysis, retrieval

mcp = FastMCP("fso-ai-market-intel-mcp")


@mcp.tool()
def health() -> str:
    """Report that the server is up and reachable."""
    return "ok"


mcp.tool(retrieval.list_datasets)
mcp.tool(retrieval.describe_dataset)
mcp.tool(retrieval.query_dataset)
mcp.tool(analysis.model_roi)
mcp.tool(analysis.perception_vs_performance)


def main() -> None:
    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
