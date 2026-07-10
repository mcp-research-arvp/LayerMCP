from __future__ import annotations

import asyncio
from collections import Counter
import inspect
import json
from pathlib import Path
import unittest

from evaluation.evaluate import load_benchmark
from mcp_server.retail_state import reset_retail_state
from mcp_server.server import mcp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "enterprise" / "tool_routing_enterprise_v2_controlled.json"
PUBLIC_ADAPTED_PATH = PROJECT_ROOT / "benchmark" / "enterprise" / "tool_routing_enterprise_public_adapted.json"
TAU3_TASKS_PATH = PROJECT_ROOT / "data" / "raw" / "tau3_retail" / "benchmark" / "tasks.json"

FROZEN_RETAIL_TOOLS = {
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

REQUIRED_FIELDS = {
    "id",
    "domain",
    "task_type",
    "difficulty",
    "source",
    "query",
    "available_tools",
    "expected_tool",
    "expected_args",
    "expected_answer",
    "perturbation_type",
    "notes",
}

PROVENANCE_FIELDS = {
    "source_dataset",
    "source_domain",
    "source_task_id",
    "source_action",
    "provenance_type",
}


def _run_registered_tool(name: str, arguments: dict) -> object:
    reset_retail_state()
    return asyncio.run(mcp._tool_manager._tools[name].run(arguments))


class EnterpriseV2ControlledBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with BENCHMARK_PATH.open("r", encoding="utf-8") as handle:
            cls.raw_samples = json.load(handle)
        cls.samples = load_benchmark(BENCHMARK_PATH)

    def test_sample_count_and_balance(self) -> None:
        self.assertEqual(len(self.raw_samples), 48)
        counts = Counter(sample["expected_tool"] for sample in self.raw_samples)
        self.assertEqual(set(counts), FROZEN_RETAIL_TOOLS)
        self.assertTrue(all(count == 4 for count in counts.values()))

    def test_schema_and_tool_menu(self) -> None:
        seen_ids: set[str] = set()
        for sample in self.raw_samples:
            self.assertTrue(REQUIRED_FIELDS.issubset(sample))
            self.assertNotIn(sample["id"], seen_ids)
            seen_ids.add(sample["id"])
            self.assertEqual(sample["domain"], "enterprise_automation")
            self.assertEqual(sample["task_type"], "single_tool_routing")
            self.assertEqual(sample["source"], "controlled_synthetic")
            self.assertEqual(set(sample["available_tools"]), FROZEN_RETAIL_TOOLS)
            self.assertIn(sample["expected_tool"], sample["available_tools"])

    def test_expected_args_match_registered_tool_signatures(self) -> None:
        for sample in self.samples:
            tool = mcp._tool_manager._tools[sample.expected_tool]
            inspect.signature(tool.fn).bind(**sample.expected_args)

    def test_all_samples_execute_through_registered_tools(self) -> None:
        for sample in self.samples:
            result = _run_registered_tool(sample.expected_tool, sample.expected_args)
            self.assertIsNotNone(result)


class EnterprisePublicAdaptedBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with PUBLIC_ADAPTED_PATH.open("r", encoding="utf-8") as handle:
            cls.raw_samples = json.load(handle)
        cls.samples = load_benchmark(PUBLIC_ADAPTED_PATH)
        with TAU3_TASKS_PATH.open("r", encoding="utf-8") as handle:
            cls.valid_source_task_ids = {str(task["id"]) for task in json.load(handle)}

    def test_schema_provenance_and_tool_menu(self) -> None:
        seen_ids: set[str] = set()
        for sample in self.raw_samples:
            self.assertTrue(REQUIRED_FIELDS.issubset(sample))
            self.assertTrue(PROVENANCE_FIELDS.issubset(sample))
            self.assertNotIn(sample["id"], seen_ids)
            seen_ids.add(sample["id"])
            self.assertEqual(sample["domain"], "enterprise_automation")
            self.assertEqual(sample["task_type"], "single_tool_routing")
            self.assertEqual(sample["source"], "public_adapted")
            self.assertEqual(sample["source_dataset"], "tau3_retail")
            self.assertEqual(sample["source_domain"], "retail")
            self.assertEqual(sample["provenance_type"], "public_adapted")
            self.assertIsInstance(sample["source_action"], str)
            self.assertGreater(len(sample["source_action"].strip()), 0)
            self.assertIn(str(sample["source_task_id"]), self.valid_source_task_ids)
            self.assertEqual(set(sample["available_tools"]), FROZEN_RETAIL_TOOLS)
            self.assertIn(sample["expected_tool"], FROZEN_RETAIL_TOOLS)

    def test_public_adapted_coverage(self) -> None:
        counts = Counter(sample["expected_tool"] for sample in self.raw_samples)
        self.assertEqual(set(counts), FROZEN_RETAIL_TOOLS)
        self.assertTrue(all(count >= 1 for count in counts.values()))

    def test_expected_args_match_registered_tool_signatures(self) -> None:
        for sample in self.samples:
            tool = mcp._tool_manager._tools[sample.expected_tool]
            inspect.signature(tool.fn).bind(**sample.expected_args)

    def test_all_samples_execute_through_registered_tools(self) -> None:
        for sample in self.samples:
            result = _run_registered_tool(sample.expected_tool, sample.expected_args)
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
