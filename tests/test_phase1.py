from __future__ import annotations

from pathlib import Path
import unittest

from evaluation.evaluate import load_benchmark
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
from mcp_server.server import mcp
from mcp_server.tool_impls import (
    calculator,
    customer_lookup,
    github_search,
    read_code_file,
    stock_price_api,
    ticket_router,
    unit_converter,
)
from models.qwen_router import HALLUCINATED_TOOL, _extract_tool_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ToolFixtureTests(unittest.TestCase):
    def test_calculator_safe_arithmetic(self) -> None:
        self.assertEqual(calculator("2 + 3 * 4")["result"], 14)

    def test_calculator_rejects_unsafe_expressions(self) -> None:
        with self.assertRaises(ValueError):
            calculator("__import__('os').system('echo unsafe')")

    def test_stock_price_api_deterministic_output(self) -> None:
        self.assertEqual(stock_price_api("aapl")["price"], 214.35)

    def test_unit_converter_deterministic_output(self) -> None:
        result = unit_converter(10, "km", "miles")
        self.assertEqual(result["converted_value"], 6.2137)

    def test_customer_lookup_deterministic_output(self) -> None:
        self.assertEqual(customer_lookup("12345")["status"], "premium")

    def test_github_search_deterministic_output(self) -> None:
        result = github_search("authentication bugs")
        self.assertEqual(result["source"], "offline-fixture")
        self.assertIn("authentication bugs", result["results"][0]["title"])

    def test_read_code_file_deterministic_output(self) -> None:
        result = read_code_file("src/auth.py")
        self.assertIn("authenticate", result["content"])

    def test_ticket_router_deterministic_output(self) -> None:
        self.assertEqual(
            ticket_router("duplicate invoice charge")["category"],
            "billing",
        )

    def test_simplify_expression_deterministic_output(self) -> None:
        result = simplify_expression("(x*x + 2*x + 1)/(x + 1)")
        self.assertEqual(result["simplified"], "x + 1")
        self.assertEqual(result["source"], "sympy")

    def test_symbolic_tools_reject_invalid_expressions_cleanly(self) -> None:
        with self.assertRaises(ValueError):
            simplify_expression("")

        with self.assertRaises(ValueError):
            simplify_expression("x +")

        with self.assertRaises(ValueError):
            solve_equation("x + 1", "x")

    def test_solve_equation_simple_equation(self) -> None:
        result = solve_equation("x**2 - 4 = 0")
        self.assertEqual(result["variable"], "x")
        self.assertEqual(result["solutions"], ["-2", "2"])

    def test_factor_expand_and_differentiate_examples(self) -> None:
        self.assertEqual(
            factor_expression("x**2 - 1")["factored"],
            "(x - 1)*(x + 1)",
        )
        self.assertEqual(
            expand_expression("(x + 2)*(x - 3)")["expanded"],
            "x**2 - x - 6",
        )
        self.assertEqual(
            differentiate_expression("x**3 + 2*x")["derivative"],
            "3*x**2 + 2",
        )

    def test_convert_units_pint_output(self) -> None:
        result = convert_units(10, "meter", "centimeter")
        self.assertEqual(result["converted_value"], 1000.0)
        self.assertEqual(result["source"], "pint")

    def test_enterprise_order_lookup_and_update(self) -> None:
        order = get_order("ord-1001")
        self.assertEqual(order["customer_id"], "CUST-1001")
        self.assertEqual(order["status"], "processing")

        update = update_order_status("ORD-1001", "shipped")
        self.assertEqual(update["previous_status"], "processing")
        self.assertEqual(update["status"], "shipped")
        self.assertTrue(update["updated"])

    def test_support_ticket_creation_is_deterministic(self) -> None:
        first = create_support_ticket("cust-1001", "Invoice missing", "high")
        second = create_support_ticket("CUST-1001", "Invoice missing", "high")
        self.assertEqual(first["ticket_id"], second["ticket_id"])
        self.assertEqual(first["priority"], "high")
        self.assertEqual(first["status"], "open")

    def test_search_knowledge_base_retrieves_matching_article(self) -> None:
        result = search_knowledge_base("duplicate invoice refund")
        self.assertEqual(result["results"][0]["article_id"], "KB-002")
        self.assertEqual(result["results"][0]["category"], "billing")

    def test_check_policy_allow_and_deny_cases(self) -> None:
        allowed = check_policy("refund", {"amount": 50})
        denied = check_policy("refund", {"amount": 150})
        self.assertTrue(allowed["allowed"])
        self.assertEqual(allowed["reason"], "refund_amount_within_limit")
        self.assertFalse(denied["allowed"])
        self.assertEqual(denied["reason"], "refund_amount_exceeds_limit")

    def test_invalid_enterprise_inputs_are_handled_cleanly(self) -> None:
        with self.assertRaises(ValueError):
            update_order_status("ORD-1001", "lost")

        with self.assertRaises(ValueError):
            create_support_ticket("CUST-1001", "Need help", "critical")

    def test_mcp_server_exposes_expanded_tool_names(self) -> None:
        expected_tools = {
            "calculator",
            "customer_lookup",
            "github_search",
            "stock_price_api",
            "unit_converter",
            "read_code_file",
            "ticket_router",
            "simplify_expression",
            "solve_equation",
            "factor_expression",
            "expand_expression",
            "differentiate_expression",
            "convert_units",
            "get_order",
            "update_order_status",
            "create_support_ticket",
            "search_knowledge_base",
            "check_policy",
        }
        actual_tools = set(mcp._tool_manager._tools)
        self.assertTrue(expected_tools.issubset(actual_tools))


class BenchmarkLoaderTests(unittest.TestCase):
    def test_benchmark_loader_handles_old_schema(self) -> None:
        samples = load_benchmark(PROJECT_ROOT / "benchmark" / "tool_routing.json")
        self.assertEqual(samples[0].id, "sample_0001")
        self.assertEqual(samples[0].domain, "unknown")
        self.assertEqual(samples[0].expected_args, {"expression": "25 * 17"})

    def test_benchmark_loader_handles_new_schema(self) -> None:
        samples = load_benchmark(PROJECT_ROOT / "benchmark" / "tool_routing_smoke.json")
        self.assertEqual(samples[0].id, "smoke_finance_001")
        self.assertEqual(samples[0].domain, "finance")
        self.assertEqual(samples[0].expected_args, {"ticker": "AAPL"})

    def test_evaluator_can_load_all_benchmark_files(self) -> None:
        benchmark_paths = [
            PROJECT_ROOT / "benchmark" / "tool_routing.json",
            PROJECT_ROOT / "benchmark" / "tool_routing_smoke.json",
            PROJECT_ROOT / "benchmark" / "tool_routing_controlled.json",
        ]
        lengths = [len(load_benchmark(path)) for path in benchmark_paths]
        self.assertEqual(lengths, [4, 8, 40])


class RouterParserTests(unittest.TestCase):
    def test_router_parser_exact_tool_name_matching(self) -> None:
        self.assertEqual(
            _extract_tool_name("calculator", ["calculator", "github_search"]),
            "calculator",
        )

    def test_router_parser_json_like_tool_output(self) -> None:
        self.assertEqual(
            _extract_tool_name('{"tool": "calculator"}', ["calculator", "github_search"]),
            "calculator",
        )

    def test_router_parser_returns_hallucinated_tool_for_unknown_output(self) -> None:
        self.assertEqual(
            _extract_tool_name("use the calculator please", ["calculator", "github_search"]),
            HALLUCINATED_TOOL,
        )


if __name__ == "__main__":
    unittest.main()
