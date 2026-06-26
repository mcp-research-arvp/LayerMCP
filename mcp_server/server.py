from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.tool_impls import (
    calculator,
    customer_lookup,
    github_search,
    read_code_file,
    stock_price_api,
    ticket_router,
    unit_converter,
)

mcp = FastMCP("ResearchTools", json_response=True)
mcp.tool()(calculator)
mcp.tool()(customer_lookup)
mcp.tool()(github_search)
mcp.tool()(stock_price_api)
mcp.tool()(unit_converter)
mcp.tool()(read_code_file)
mcp.tool()(ticket_router)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
