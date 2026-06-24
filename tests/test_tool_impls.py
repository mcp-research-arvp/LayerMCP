from __future__ import annotations

import unittest

from mcp_server.tool_impls import calculator, customer_lookup, github_search


class ToolImplTests(unittest.TestCase):
    def test_calculator_evaluates_basic_arithmetic(self) -> None:
        self.assertEqual(
            calculator("25 * 17"),
            {"expression": "25 * 17", "result": 425},
        )

    def test_calculator_rejects_unsupported_syntax(self) -> None:
        with self.assertRaises(ValueError):
            calculator("__import__('os').system('echo unsafe')")

    def test_calculator_rejects_division_by_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "Division by zero"):
            calculator("1 / 0")

    def test_customer_lookup_is_deterministic(self) -> None:
        self.assertEqual(
            customer_lookup("12345"),
            {
                "customer_id": "12345",
                "status": "premium",
                "source": "offline-fixture",
            },
        )

    def test_customer_lookup_rejects_invalid_ids(self) -> None:
        with self.assertRaises(ValueError):
            customer_lookup("../secret")

    def test_github_search_returns_offline_results(self) -> None:
        result = github_search("authentication bugs")

        self.assertEqual(result["query"], "authentication bugs")
        self.assertEqual(result["source"], "offline-fixture")
        self.assertEqual(result["results"][0]["repository"], "example/research-mcp")


if __name__ == "__main__":
    unittest.main()
