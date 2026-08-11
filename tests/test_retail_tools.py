from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import unittest

from mcp_server.retail_state import (
    RETAIL_FIXTURE_PATH,
    reset_retail_state,
    snapshot_retail_state,
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
from mcp_server.server import mcp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = (
    PROJECT_ROOT / "mcp_server" / "fixtures" / "tau2_retail_provenance.json"
)
EXPANSION_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "enterprise"
    / "enterprise_tau2_single_step.json"
)
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


class RetailToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = json.loads(EXPANSION_PATH.read_text(encoding="utf-8"))
        cls.example = {}
        for row in cls.rows:
            cls.example.setdefault(row["expected_tool"], row)

    def setUp(self) -> None:
        reset_retail_state()

    def _call_example(self, name: str) -> object:
        return mcp._tool_manager._tools[name].fn(
            **self.example[name]["expected_args"]
        )

    def test_fixture_hash_and_cardinality(self) -> None:
        provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        fixture_hash = hashlib.sha256(RETAIL_FIXTURE_PATH.read_bytes()).hexdigest()
        self.assertEqual(fixture_hash, provenance["derived_fixture_sha256"])
        self.assertEqual(
            provenance["source_revision"],
            "363133ada1936491fb5bcec33cd62c3518a99f65",
        )
        state = snapshot_retail_state()
        self.assertEqual(len(state["products"]), 50)
        self.assertEqual(len(state["users"]), 500)
        self.assertEqual(len(state["orders"]), 1000)

    def test_reset_restores_exact_state(self) -> None:
        initial = snapshot_retail_state()
        self._call_example("modify_user_address")
        self.assertNotEqual(snapshot_retail_state(), initial)
        reset_retail_state()
        self.assertEqual(snapshot_retail_state(), initial)

    def test_native_ids_are_case_sensitive(self) -> None:
        user_id = self.example["get_user_details"]["expected_args"]["user_id"]
        self.assertEqual(get_user_details(user_id)["user_id"], user_id)
        with self.assertRaises(ValueError):
            get_user_details(user_id.upper())

    def test_identity_lookups(self) -> None:
        email_args = self.example["find_user_id_by_email"]["expected_args"]
        name_args = self.example["find_user_id_by_name_zip"]["expected_args"]
        self.assertTrue(find_user_id_by_email(**email_args))
        self.assertTrue(find_user_id_by_name_zip(**name_args))

    def test_native_read_shapes(self) -> None:
        user = self._call_example("get_user_details")
        order = self._call_example("get_order_details")
        product = self._call_example("get_product_details")
        self.assertIn("orders", user)
        self.assertIsInstance(user["payment_methods"], dict)
        self.assertIn("fulfillments", order)
        self.assertIsInstance(product["variants"], dict)

    def test_order_lookup_accepts_an_unprefixed_tau2_order_id(self) -> None:
        order_id = self.example["get_order_details"]["expected_args"]["order_id"]
        self.assertTrue(order_id.startswith("#W"))
        result = get_order_details(order_id[1:])
        self.assertEqual(result["order_id"], order_id)

    def test_cancel_pending_order(self) -> None:
        result = self._call_example("cancel_pending_order")
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(
            any(
                payment["transaction_type"] == "refund"
                for payment in result["payment_history"]
            )
        )

    def test_modify_pending_order_items(self) -> None:
        args = self.example["modify_pending_order_items"]["expected_args"]
        result = modify_pending_order_items(**args)
        self.assertEqual(result["status"], "pending (item modified)")
        self.assertTrue(
            all(
                item_id in [item["item_id"] for item in result["items"]]
                for item_id in args["new_item_ids"]
            )
        )

    def test_address_mutations(self) -> None:
        order_args = self.example["modify_pending_order_address"]["expected_args"]
        user_args = self.example["modify_user_address"]["expected_args"]
        order = modify_pending_order_address(**order_args)
        user = modify_user_address(**user_args)
        self.assertEqual(order["address"]["address1"], order_args["address1"])
        self.assertEqual(user["address"]["address1"], user_args["address1"])

    def test_return_and_exchange(self) -> None:
        returned = self._call_example("return_delivered_order_items")
        self.assertEqual(returned["status"], "return requested")
        reset_retail_state()
        exchanged = self._call_example("exchange_delivered_order_items")
        self.assertEqual(exchanged["status"], "exchange requested")

    def test_transfer_matches_tau2_contract(self) -> None:
        self.assertEqual(
            transfer_to_human_agents("Customer needs policy assistance."),
            "Transfer successful",
        )

    def test_registered_tools_execute_native_examples(self) -> None:
        for name in RETAIL_TOOL_NAMES:
            reset_retail_state()
            result = asyncio.run(
                mcp._tool_manager._tools[name].run(
                    self.example[name]["expected_args"]
                )
            )
            self.assertIsNotNone(result, name)


if __name__ == "__main__":
    unittest.main()
