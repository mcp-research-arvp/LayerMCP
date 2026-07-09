from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.enterprise_tools import (
    check_policy,
    create_support_ticket,
    get_order,
    search_knowledge_base,
    update_order_status,
)
from mcp_server.math_tools import (
    convert_units,
    differentiate_expression,
    expand_expression,
    factor_expression,
    simplify_expression,
    solve_equation,
)
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
mcp.tool()(simplify_expression)
mcp.tool()(solve_equation)
mcp.tool()(factor_expression)
mcp.tool()(expand_expression)
mcp.tool()(differentiate_expression)
mcp.tool()(convert_units)
mcp.tool()(get_order)
mcp.tool()(update_order_status)
mcp.tool()(create_support_ticket)
mcp.tool()(search_knowledge_base)
mcp.tool()(check_policy)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
