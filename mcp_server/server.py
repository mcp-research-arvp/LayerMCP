from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.tool_impls import calculator, customer_lookup, github_search

mcp = FastMCP("ResearchTools", json_response=True)
mcp.tool()(calculator)
mcp.tool()(customer_lookup)
mcp.tool()(github_search)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
