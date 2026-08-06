import os

from fastmcp import FastMCP

mcp = FastMCP("fso-ai-market-intel-mcp")


@mcp.tool()
def health() -> str:
    """Report that the server is up and reachable."""
    return "ok"


def main() -> None:
    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
