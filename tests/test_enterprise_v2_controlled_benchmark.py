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
TAU2_EXPANSION_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "enterprise"
    / "tool_routing_enterprise_tau2_public_adapted.json"
)
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

PUBLIC_EXPECTED_ANSWER_PILOT = {
    "enterprise_public_adapted_find_user_id_by_email_001": "find_user_id_by_email",
    "enterprise_public_adapted_find_user_id_by_name_zip_001": "find_user_id_by_name_zip",
    "enterprise_public_adapted_get_user_details_001": "get_user_details",
    "enterprise_public_adapted_get_order_details_001": "get_order_details",
    "enterprise_public_adapted_get_product_details_001": "get_product_details",
    "enterprise_public_adapted_cancel_pending_order_001": "cancel_pending_order",
    "enterprise_public_adapted_modify_pending_order_items_001": "modify_pending_order_items",
    "enterprise_public_adapted_modify_pending_order_address_001": "modify_pending_order_address",
    "enterprise_public_adapted_modify_user_address_001": "modify_user_address",
    "enterprise_public_adapted_return_delivered_order_items_001": "return_delivered_order_items",
    "enterprise_public_adapted_exchange_delivered_order_items_001": "exchange_delivered_order_items",
    "enterprise_public_adapted_transfer_to_human_agents_001": "transfer_to_human_agents",
}
TAU2_EXPANSION_COUNTS = {
    "find_user_id_by_email": 10,
    "find_user_id_by_name_zip": 12,
    "get_user_details": 12,
    "get_order_details": 15,
    "get_product_details": 12,
    "cancel_pending_order": 10,
    "modify_pending_order_items": 10,
    "modify_pending_order_address": 10,
    "modify_user_address": 8,
    "return_delivered_order_items": 12,
    "exchange_delivered_order_items": 12,
    "transfer_to_human_agents": 4,
}
TAU2_PROVENANCE_FIELDS = {
    "source_dataset",
    "source_revision",
    "source_split",
    "source_task_id",
    "source_action_id",
    "source_action",
    "source_expected_args",
    "source_hash",
    "source_license",
    "transformation_notes",
    "entity_mapping_notes",
}

REQUIRED_FIELDS = {
    "id",
    "domain",
    "task_type",
    "difficulty",
    "source",
    "query",
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


def _structured_result_value(value: object) -> object:
    if isinstance(value, dict):
        return value
    return {"result": value}


def _contains_expected_answer(actual: object, expected: object) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_expected_answer(actual[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _contains_expected_answer(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected)
            )
        )
    return actual == expected


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

    def test_schema_and_registry_compatibility(self) -> None:
        seen_ids: set[str] = set()
        for sample in self.raw_samples:
            self.assertTrue(REQUIRED_FIELDS.issubset(sample))
            self.assertNotIn(sample["id"], seen_ids)
            seen_ids.add(sample["id"])
            self.assertEqual(sample["domain"], "enterprise_automation")
            self.assertEqual(sample["task_type"], "single_tool_routing")
            self.assertEqual(sample["source"], "controlled_synthetic")
            self.assertNotIn("available_tools", sample)
            self.assertIn(sample["expected_tool"], FROZEN_RETAIL_TOOLS)

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

    def test_schema_provenance_and_registry_compatibility(self) -> None:
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
            self.assertNotIn("available_tools", sample)
            self.assertIn(sample["expected_tool"], FROZEN_RETAIL_TOOLS)

    def test_source_task_ids_exist_in_optional_raw_tau_dataset(self) -> None:
        if not TAU3_TASKS_PATH.exists():
            self.skipTest(f"Optional raw tau source is not present: {TAU3_TASKS_PATH}")

        with TAU3_TASKS_PATH.open("r", encoding="utf-8") as handle:
            valid_source_task_ids = {
                str(task["id"]) for task in json.load(handle)
            }
        for sample in self.raw_samples:
            self.assertIn(str(sample["source_task_id"]), valid_source_task_ids)

    def test_public_adapted_coverage(self) -> None:
        counts = Counter(sample["expected_tool"] for sample in self.raw_samples)
        self.assertEqual(set(counts), FROZEN_RETAIL_TOOLS)
        self.assertTrue(all(count >= 1 for count in counts.values()))

    def test_expected_answer_pilot_has_one_row_per_retail_tool(self) -> None:
        populated = {
            sample["id"]: sample
            for sample in self.raw_samples
            if sample["expected_answer"] is not None
        }
        self.assertEqual(set(populated), set(PUBLIC_EXPECTED_ANSWER_PILOT))
        self.assertEqual(
            {sample["expected_tool"] for sample in populated.values()},
            FROZEN_RETAIL_TOOLS,
        )
        for sample_id, expected_tool in PUBLIC_EXPECTED_ANSWER_PILOT.items():
            self.assertEqual(populated[sample_id]["expected_tool"], expected_tool)

    def test_expected_answer_pilot_matches_deterministic_gold_execution(self) -> None:
        samples_by_id = {sample["id"]: sample for sample in self.raw_samples}
        for sample_id in PUBLIC_EXPECTED_ANSWER_PILOT:
            sample = samples_by_id[sample_id]
            first = _structured_result_value(
                _run_registered_tool(sample["expected_tool"], sample["expected_args"])
            )
            second = _structured_result_value(
                _run_registered_tool(sample["expected_tool"], sample["expected_args"])
            )
            self.assertEqual(first, second, sample_id)
            self.assertTrue(
                _contains_expected_answer(first, sample["expected_answer"]),
                sample_id,
            )

    def test_expected_args_match_registered_tool_signatures(self) -> None:
        for sample in self.samples:
            tool = mcp._tool_manager._tools[sample.expected_tool]
            inspect.signature(tool.fn).bind(**sample.expected_args)

    def test_all_samples_execute_through_registered_tools(self) -> None:
        for sample in self.samples:
            result = _run_registered_tool(sample.expected_tool, sample.expected_args)
            self.assertIsNotNone(result)


class EnterpriseTau2PublicAdaptedExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with TAU2_EXPANSION_PATH.open("r", encoding="utf-8") as handle:
            cls.raw_samples = json.load(handle)
        cls.samples = load_benchmark(TAU2_EXPANSION_PATH)

    def test_count_balance_schema_and_provenance(self) -> None:
        self.assertEqual(len(self.raw_samples), 127)
        self.assertEqual(len(self.samples), 127)
        self.assertEqual(
            Counter(sample["expected_tool"] for sample in self.raw_samples),
            TAU2_EXPANSION_COUNTS,
        )
        ids = [sample["id"] for sample in self.raw_samples]
        self.assertEqual(len(ids), len(set(ids)))
        for sample in self.raw_samples:
            self.assertTrue(REQUIRED_FIELDS.issubset(sample))
            self.assertTrue(TAU2_PROVENANCE_FIELDS.issubset(sample))
            self.assertEqual(sample["source"], "public_adapted")
            self.assertEqual(sample["source_dataset"], "tau2_bench_retail")
            self.assertIn(sample["source_split"], {"train", "test"})
            self.assertEqual(len(sample["source_hash"]), 64)
            self.assertEqual(sample["source_action"], sample["expected_tool"])
            self.assertNotIn("available_tools", sample)
            self.assertIsNotNone(sample["expected_answer"])
            self.assertGreater(len(sample["entity_mapping_notes"]), 0)

    def test_args_match_signatures_and_gold_results_are_deterministic(self) -> None:
        for sample in self.raw_samples:
            tool = mcp._tool_manager._tools[sample["expected_tool"]]
            inspect.signature(tool.fn).bind(**sample["expected_args"])
            first = _structured_result_value(
                _run_registered_tool(sample["expected_tool"], sample["expected_args"])
            )
            second = _structured_result_value(
                _run_registered_tool(sample["expected_tool"], sample["expected_args"])
            )
            self.assertEqual(first, second, sample["id"])
            self.assertTrue(
                _contains_expected_answer(first, sample["expected_answer"]),
                sample["id"],
            )


if __name__ == "__main__":
    unittest.main()
