from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.coding_tools import (
    code_list_files,
    code_read_file,
    code_search_text,
    git_diff,
    git_log,
    git_show,
    git_status,
)
from mcp_server.enterprise_tools import (
    check_policy,
    create_support_ticket,
    get_order,
    search_knowledge_base,
    update_order_status,
)
from mcp_server.finance_tools import (
    finance_extract_pdf_tables,
    finance_get_company_facts,
    finance_get_filing_section,
    finance_get_financial_statement,
    finance_get_market_quote,
    finance_get_market_time_series,
    finance_lookup_company,
    finance_parse_xbrl,
    finance_query_table,
    finance_search_filings,
)
from mcp_server.math_tools import (
    base_arithmetic,
    convert_units,
    differentiate_expression,
    expand_expression,
    factor_expression,
    gcd_lcm,
    integer_factorization,
    modular_arithmetic,
    simplify_expression,
    solve_equation,
)
from mcp_server.retail_tools import (
    cancel_pending_order,
    exchange_delivered_order_items,
    find_user_id_by_email,
    find_user_id_by_name_zip,
    get_order_details,
    get_product_details,
    get_user_details,
    modify_pending_order_address,
    modify_pending_order_items,
    modify_user_address,
    return_delivered_order_items,
    transfer_to_human_agents,
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
mcp.tool()(integer_factorization)
mcp.tool()(gcd_lcm)
mcp.tool()(modular_arithmetic)
mcp.tool()(base_arithmetic)
mcp.tool()(get_order)
mcp.tool()(update_order_status)
mcp.tool()(create_support_ticket)
mcp.tool()(search_knowledge_base)
mcp.tool()(check_policy)
mcp.tool()(find_user_id_by_email)
mcp.tool()(find_user_id_by_name_zip)
mcp.tool()(get_user_details)
mcp.tool()(get_order_details)
mcp.tool()(get_product_details)
mcp.tool()(cancel_pending_order)
mcp.tool()(modify_pending_order_items)
mcp.tool()(modify_pending_order_address)
mcp.tool()(modify_user_address)
mcp.tool()(return_delivered_order_items)
mcp.tool()(exchange_delivered_order_items)
mcp.tool()(transfer_to_human_agents)
mcp.tool()(code_list_files)
mcp.tool()(code_read_file)
mcp.tool()(code_search_text)
mcp.tool()(git_log)
mcp.tool()(git_show)
mcp.tool()(git_diff)
mcp.tool()(git_status)
mcp.tool()(finance_lookup_company)
mcp.tool()(finance_search_filings)
mcp.tool()(finance_get_filing_section)
mcp.tool()(finance_get_company_facts)
mcp.tool()(finance_get_financial_statement)
mcp.tool()(finance_parse_xbrl)
mcp.tool()(finance_query_table)
mcp.tool()(finance_extract_pdf_tables)
mcp.tool()(finance_get_market_quote)
mcp.tool()(finance_get_market_time_series)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
