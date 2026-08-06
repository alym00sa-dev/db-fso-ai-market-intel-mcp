import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fso-ai-market-intel-mcp")


@mcp.tool()
def health() -> str:
    """Report that the server is up and reachable."""
    return "ok"


def main() -> None:
    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
