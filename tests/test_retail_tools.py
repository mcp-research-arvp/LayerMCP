from __future__ import annotations

import asyncio
import inspect
import unittest

from mcp_server.retail_state import reset_retail_state
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
from mcp_server.server import mcp


RETAIL_TOOL_NAMES = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "cancel_pending_order",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_user_address",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "transfer_to_human_agents",
}

ENTERPRISE_V1_TOOL_NAMES = {
    "customer_lookup",
    "ticket_router",
    "get_order",
    "update_order_status",
    "create_support_ticket",
    "search_knowledge_base",
    "check_policy",
}


class RetailToolTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_retail_state()

    def _run_registered_tool(self, name: str, arguments: dict) -> object:
        return asyncio.run(mcp._tool_manager._tools[name].run(arguments))

    def _run_isolated_registered_tool(self, name: str, arguments: dict) -> object:
        reset_retail_state()
        return self._run_registered_tool(name, arguments)

    def test_identity_lookup_tools(self) -> None:
        self.assertEqual(
            find_user_id_by_email("MEI.KOVACS@example.com"),
            "USER-MEI",
        )
        self.assertEqual(
            find_user_id_by_name_zip("yusuf", "ROSSI", "19122"),
            "USER-YUSUF",
        )
        with self.assertRaises(ValueError):
            find_user_id_by_email("missing@example.com")
        with self.assertRaises(ValueError):
            find_user_id_by_name_zip("Yusuf", "Rossi", "00000")

    def test_user_order_and_product_details(self) -> None:
        user = get_user_details("USER-YUSUF")
        self.assertEqual(user["email"], "yusuf.rossi@example.com")
        self.assertEqual(len(user["payment_methods"]), 3)
        self.assertIn("RET-1001", user["order_ids"])

        order = get_order_details("RET-1001")
        self.assertEqual(order["status"], "pending")
        self.assertEqual(order["total"], 140.0)

        product = get_product_details("PROD-KEYBOARD")
        self.assertEqual(product["name"], "Mechanical Keyboard")
        self.assertEqual(len(product["variants"]), 3)

    def test_cancel_pending_order_mutates_and_reset_restores(self) -> None:
        result = cancel_pending_order("RET-1002", "ordered by mistake")
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["cancel_reason"], "ordered by mistake")
        self.assertEqual(get_order_details("RET-1002")["status"], "cancelled")
        gift_card = next(
            method
            for method in get_user_details("USER-YUSUF")["payment_methods"]
            if method["payment_method_id"] == "GIFT-YUSUF"
        )
        self.assertEqual(gift_card["balance"], 245.0)

        reset_retail_state()
        self.assertEqual(get_order_details("RET-1002")["status"], "pending")
        restored_gift_card = next(
            method
            for method in get_user_details("USER-YUSUF")["payment_methods"]
            if method["payment_method_id"] == "GIFT-YUSUF"
        )
        self.assertEqual(restored_gift_card["balance"], 200.0)

    def test_cancel_pending_order_rejects_invalid_state_and_reason(self) -> None:
        with self.assertRaises(ValueError):
            cancel_pending_order("RET-1004", "no longer needed")
        with self.assertRaises(ValueError):
            cancel_pending_order("RET-1001", "changed my mind")

    def test_modify_pending_order_items_mutates(self) -> None:
        result = modify_pending_order_items(
            "RET-1001",
            ["ITEM-KB-LINEAR"],
            ["ITEM-KB-CLICKY"],
            "GIFT-YUSUF",
        )
        self.assertEqual(result["status"], "pending (item modified)")
        self.assertEqual(result["modification_price_difference"], 15.0)
        self.assertIn("ITEM-KB-CLICKY", [item["item_id"] for item in result["items"]])
        gift_card = next(
            method
            for method in get_user_details("USER-YUSUF")["payment_methods"]
            if method["payment_method_id"] == "GIFT-YUSUF"
        )
        self.assertEqual(gift_card["balance"], 185.0)

        reset_retail_state()
        self.assertIn(
            "ITEM-KB-LINEAR",
            [item["item_id"] for item in get_order_details("RET-1001")["items"]],
        )

    def test_modify_pending_order_items_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            modify_pending_order_items("RET-1001", ["ITEM-KB-LINEAR"], [], "CARD-YUSUF")
        with self.assertRaises(ValueError):
            modify_pending_order_items(
                "RET-1001",
                ["ITEM-KB-LINEAR"],
                ["ITEM-KB-SILENT"],
                "CARD-YUSUF",
            )
        with self.assertRaises(ValueError):
            modify_pending_order_items(
                "RET-1001",
                ["ITEM-KB-LINEAR"],
                ["ITEM-BOTTLE-1000"],
                "CARD-YUSUF",
            )
        with self.assertRaises(ValueError):
            modify_pending_order_items(
                "RET-1001",
                ["ITEM-LAMP-USB"],
                ["ITEM-LAMP-BATTERY"],
                "CARD-YUSUF",
            )
        with self.assertRaises(ValueError):
            modify_pending_order_items(
                "RET-1001",
                ["ITEM-KB-LINEAR"],
                ["ITEM-KB-CLICKY"],
                "CARD-MEI",
            )
        with self.assertRaises(ValueError):
            modify_pending_order_items(
                "RET-1001",
                ["ITEM-KB-LINEAR"],
                ["ITEM-KB-LINEAR"],
                "CARD-YUSUF",
            )

    def test_address_mutations_are_independent_and_resettable(self) -> None:
        modify_pending_order_address(
            "RET-1001",
            "10 Order Lane",
            "Unit 2",
            "Boston",
            "MA",
            "US",
            "02110",
        )
        order = get_order_details("RET-1001")
        user = get_user_details("USER-YUSUF")
        self.assertEqual(order["address"]["address1"], "10 Order Lane")
        self.assertEqual(user["address"]["address1"], "123 Pine Street")

        modify_user_address(
            "USER-YUSUF",
            "20 Profile Road",
            "",
            "New York",
            "NY",
            "US",
            "10001",
        )
        order = get_order_details("RET-1001")
        user = get_user_details("USER-YUSUF")
        self.assertEqual(order["address"]["address1"], "10 Order Lane")
        self.assertEqual(user["address"]["address1"], "20 Profile Road")

        reset_retail_state()
        self.assertEqual(get_order_details("RET-1001")["address"]["address1"], "123 Pine Street")
        self.assertEqual(get_user_details("USER-YUSUF")["address"]["address1"], "123 Pine Street")

    def test_item_modified_pending_order_address_can_still_be_modified(self) -> None:
        modify_pending_order_items(
            "RET-1001",
            ["ITEM-KB-LINEAR"],
            ["ITEM-KB-CLICKY"],
            "GIFT-YUSUF",
        )
        result = modify_pending_order_address(
            "RET-1001",
            "30 Modified Order Lane",
            "",
            "Boston",
            "MA",
            "US",
            "02111",
        )
        self.assertEqual(result["status"], "pending (item modified)")
        self.assertEqual(result["address"]["address1"], "30 Modified Order Lane")

    def test_order_address_rejects_non_pending_order(self) -> None:
        with self.assertRaises(ValueError):
            modify_pending_order_address(
                "RET-2001",
                "10 Order Lane",
                "",
                "Boston",
                "MA",
                "US",
                "02110",
            )

    def test_return_delivered_order_items(self) -> None:
        before_history = list(get_order_details("RET-2001")["payment_history"])
        result = return_delivered_order_items(
            "RET-2001",
            ["ITEM-BOTTLE-500"],
            "GIFT-MEI",
        )
        self.assertEqual(result["status"], "return requested")
        self.assertEqual(result["return_items"], ["ITEM-BOTTLE-500"])
        self.assertEqual(result["return_payment_method_id"], "GIFT-MEI")
        self.assertEqual(result["payment_history"], before_history)
        gift_card = next(
            method
            for method in get_user_details("USER-MEI")["payment_methods"]
            if method["payment_method_id"] == "GIFT-MEI"
        )
        self.assertEqual(gift_card["balance"], 25.0)

        reset_retail_state()
        self.assertEqual(get_order_details("RET-2001")["status"], "delivered")

    def test_return_delivered_order_items_allows_original_payment_method(self) -> None:
        result = return_delivered_order_items(
            "RET-2001",
            ["ITEM-BOTTLE-500"],
            "CARD-MEI",
        )
        self.assertEqual(result["status"], "return requested")
        self.assertEqual(result["return_payment_method_id"], "CARD-MEI")

    def test_return_delivered_order_items_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            return_delivered_order_items("RET-1001", ["ITEM-BOTTLE-500"], "GIFT-YUSUF")
        with self.assertRaises(ValueError):
            return_delivered_order_items("RET-2001", ["ITEM-LAMP-USB"], "GIFT-MEI")
        with self.assertRaises(ValueError):
            return_delivered_order_items("RET-2001", ["ITEM-BOTTLE-500"], "GIFT-YUSUF")
        with self.assertRaises(ValueError):
            return_delivered_order_items("RET-2001", ["ITEM-BOTTLE-500"], "PAYPAL-MEI")

    def test_exchange_delivered_order_items(self) -> None:
        before_history = list(get_order_details("RET-2001")["payment_history"])
        result = exchange_delivered_order_items(
            "RET-2001",
            ["ITEM-KB-LINEAR"],
            ["ITEM-KB-CLICKY"],
            "CARD-MEI",
        )
        self.assertEqual(result["status"], "exchange requested")
        self.assertEqual(result["exchange_items"], ["ITEM-KB-LINEAR"])
        self.assertEqual(result["exchange_new_items"], ["ITEM-KB-CLICKY"])
        self.assertEqual(result["exchange_price_difference"], 15.0)
        self.assertEqual(result["payment_history"], before_history)

        reset_retail_state()
        self.assertEqual(get_order_details("RET-2001")["status"], "delivered")

    def test_exchange_delivered_order_items_does_not_mutate_payment_balance(self) -> None:
        before_balance = next(
            method
            for method in get_user_details("USER-MEI")["payment_methods"]
            if method["payment_method_id"] == "GIFT-MEI"
        )["balance"]
        result = exchange_delivered_order_items(
            "RET-2001",
            ["ITEM-BOTTLE-500"],
            ["ITEM-BOTTLE-1000"],
            "GIFT-MEI",
        )
        self.assertEqual(result["exchange_price_difference"], 12.0)
        after_balance = next(
            method
            for method in get_user_details("USER-MEI")["payment_methods"]
            if method["payment_method_id"] == "GIFT-MEI"
        )["balance"]
        self.assertEqual(after_balance, before_balance)

    def test_exchange_delivered_order_items_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            exchange_delivered_order_items(
                "RET-1001",
                ["ITEM-KB-LINEAR"],
                ["ITEM-KB-CLICKY"],
                "CARD-YUSUF",
            )
        with self.assertRaises(ValueError):
            exchange_delivered_order_items(
                "RET-2001",
                ["ITEM-KB-LINEAR"],
                ["ITEM-BOTTLE-1000"],
                "CARD-MEI",
            )
        with self.assertRaises(ValueError):
            exchange_delivered_order_items(
                "RET-2001",
                ["ITEM-KB-LINEAR"],
                ["ITEM-KB-SILENT"],
                "CARD-MEI",
            )
        with self.assertRaises(ValueError):
            exchange_delivered_order_items(
                "RET-2001",
                ["ITEM-KB-LINEAR"],
                ["ITEM-KB-CLICKY"],
                "CARD-YUSUF",
            )

    def test_transfer_to_human_agents(self) -> None:
        result = transfer_to_human_agents("Undo a cancelled order")
        self.assertTrue(result["transfer_requested"])
        self.assertEqual(result["summary"], "Undo a cancelled order")
        self.assertEqual(result["status"], "pending_human_agent")

    def test_registered_retail_tools_preserve_state_across_sequential_calls(self) -> None:
        self._run_registered_tool(
            "modify_user_address",
            {
                "user_id": "USER-YUSUF",
                "address1": "42 Workflow Way",
                "address2": "Suite 8",
                "city": "Denver",
                "state": "CO",
                "country": "US",
                "zip": "80202",
            },
        )
        user = self._run_registered_tool("get_user_details", {"user_id": "USER-YUSUF"})

        self.assertEqual(user["address"]["address1"], "42 Workflow Way")
        self.assertEqual(user["address"]["address2"], "Suite 8")
        self.assertEqual(user["address"]["zip"], "80202")

    def test_registered_retail_state_changing_samples_are_independent(self) -> None:
        arguments = {"order_id": "RET-1002", "reason": "ordered by mistake"}
        first = self._run_isolated_registered_tool("cancel_pending_order", arguments)
        second = self._run_isolated_registered_tool("cancel_pending_order", arguments)

        self.assertEqual(first["status"], "cancelled")
        self.assertEqual(second["status"], "cancelled")

    def test_registered_retail_sample_order_does_not_affect_success(self) -> None:
        cancel_arguments = {"order_id": "RET-1001", "reason": "ordered by mistake"}
        modify_arguments = {
            "order_id": "RET-1001",
            "item_ids": ["ITEM-KB-LINEAR"],
            "new_item_ids": ["ITEM-KB-CLICKY"],
            "payment_method_id": "GIFT-YUSUF",
        }

        first_sequence = [
            self._run_isolated_registered_tool("cancel_pending_order", cancel_arguments),
            self._run_isolated_registered_tool("modify_pending_order_items", modify_arguments),
        ]
        second_sequence = [
            self._run_isolated_registered_tool("modify_pending_order_items", modify_arguments),
            self._run_isolated_registered_tool("cancel_pending_order", cancel_arguments),
        ]

        self.assertEqual([result["status"] for result in first_sequence], ["cancelled", "pending (item modified)"])
        self.assertEqual([result["status"] for result in second_sequence], ["pending (item modified)", "cancelled"])

    def test_registered_non_retail_execution_does_not_reset_retail_state(self) -> None:
        cancel_pending_order("RET-1002", "ordered by mistake")
        calculator_result = self._run_registered_tool("calculator", {"expression": "2 + 2"})

        self.assertEqual(calculator_result["result"], 4)
        self.assertEqual(get_order_details("RET-1002")["status"], "cancelled")

    def test_retail_tools_and_v1_tools_are_registered(self) -> None:
        registered_tools = set(mcp._tool_manager._tools)
        self.assertTrue(RETAIL_TOOL_NAMES.issubset(registered_tools))
        self.assertTrue(ENTERPRISE_V1_TOOL_NAMES.issubset(registered_tools))
        self.assertNotIn("reset_retail_state", registered_tools)
        self.assertEqual(len(mcp._tool_manager._tools), len(registered_tools))

    def test_retail_tool_signatures(self) -> None:
        expected_parameters = {
            "find_user_id_by_email": ["email"],
            "find_user_id_by_name_zip": ["first_name", "last_name", "zip"],
            "get_user_details": ["user_id"],
            "get_order_details": ["order_id"],
            "get_product_details": ["product_id"],
            "cancel_pending_order": ["order_id", "reason"],
            "modify_pending_order_items": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
            "modify_pending_order_address": ["order_id", "address1", "address2", "city", "state", "country", "zip"],
            "modify_user_address": ["user_id", "address1", "address2", "city", "state", "country", "zip"],
            "return_delivered_order_items": ["order_id", "item_ids", "payment_method_id"],
            "exchange_delivered_order_items": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
            "transfer_to_human_agents": ["summary"],
        }
        functions = {
            name: globals()[name]
            for name in RETAIL_TOOL_NAMES
        }
        for name, parameters in expected_parameters.items():
            self.assertEqual(list(inspect.signature(functions[name]).parameters), parameters)


if __name__ == "__main__":
    unittest.main()
