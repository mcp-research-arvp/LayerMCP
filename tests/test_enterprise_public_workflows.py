import json
from collections import Counter
from pathlib import Path
import unittest

from evaluation.evaluate import (
    BenchmarkStep,
    DEFAULT_WORKFLOW_EXECUTION_MODE,
    REFERENCE_PREFIX_REPLAY_MODE,
    SERVER_PATH,
    _call_tool_with_workflow_isolation,
    _extract_structured_tool_result,
)
from mcp_server import retail_tools
from mcp_server.retail_state import reset_retail_state


BENCHMARK_PATH = Path("benchmark/enterprise/enterprise_public_workflows.json")

SUPPORTED_TOOLS = {
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


def _rows():
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def _mcp_expected_answer(value):
    return value if isinstance(value, dict) else {"result": value}


class EnterprisePublicWorkflowTests(unittest.TestCase):
    def test_enterprise_public_workflow_benchmark_shape(self):
        rows = _rows()

        self.assertEqual(len(rows), 69)
        self.assertEqual(
            Counter(row["source_split"] for row in rows),
            {"train": 45, "test": 24},
        )

        for row in rows:
            self.assertEqual(row["domain"], "enterprise_automation")
            self.assertEqual(row["task_type"], "multi_step_tool_routing")
            self.assertEqual(row["source"], "public_enterprise_workflow")
            self.assertEqual(row["source_dataset"], "tau2_bench_retail")
            self.assertEqual(
                row["query_origin"], "tau2_user_scenario_instruction_fields"
            )
            self.assertEqual(
                row["tool_sequence_origin"],
                "tau2_evaluation_criteria_actions",
            )
            self.assertEqual(row["source_action_role"], "reference_trajectory")
            self.assertEqual(row["benchmark_mode"], "grounded_tool_execution")
            self.assertEqual(
                row["workflow_execution_mode"],
                REFERENCE_PREFIX_REPLAY_MODE,
            )
            self.assertEqual(
                row["fixture_hash"],
                "660c72ef6d1ef6a9ad1a886c5835c5794075de452c6a0848a44b4377d5815262",
            )
            self.assertEqual(
                row["perturbation_type"],
                "source_workflow_format_adaptation",
            )
            self.assertTrue(row["query"])
            self.assertIn("Reason for call:", row["query"])

            steps = row["expected_steps"]
            self.assertGreaterEqual(len(steps), 2)
            for index, step in enumerate(steps):
                self.assertEqual(step["id"], f"step_{index:02d}")
                self.assertIn(step["expected_tool"], SUPPORTED_TOOLS)
                self.assertEqual(step["source_action"], step["expected_tool"])
                self.assertEqual(step["source_action_role"], "reference_trajectory")
                self.assertIsInstance(step["expected_args"], dict)
                self.assertEqual(
                    step["source_expected_args"], step["expected_args"]
                )
                self.assertIn("expected_answer", step)
                self.assertEqual(step["depends_on"], [])
                self.assertTrue(
                    step["query"].startswith(
                        "Perform the next grounded retail workflow operation: "
                    )
                )
                self.assertTrue(step["prompt_context"].strip())
                self.assertIn(
                    f"Source action index: {index}", step["prompt_context"]
                )
                self.assertIn(
                    "Authoritative source-action facts", step["prompt_context"]
                )
                self.assertIn("teacher-forced", step["prompt_context"])
                self.assertIn(
                    "earlier or later workflow steps", step["prompt_context"]
                )
                self.assertNotIn("expected_tool", step["prompt_context"])
                self.assertNotIn("expected_args", step["prompt_context"])

    def test_reference_trajectories_replay_against_retail_fixture(self):
        functions = {
            name: getattr(retail_tools, name) for name in SUPPORTED_TOOLS
        }
        for row in _rows():
            reset_retail_state()
            for step in row["expected_steps"]:
                actual = functions[step["expected_tool"]](**step["expected_args"])
                self.assertEqual(
                    _mcp_expected_answer(actual),
                    step["expected_answer"],
                )


class EnterpriseWorkflowMcpIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_shapes_primitives_and_replays_reference_state(self):
        primitive_result = await _call_tool_with_workflow_isolation(
            None,
            SERVER_PATH,
            "find_user_id_by_email",
            {"email": "mia.garcia2723@example.com"},
            (),
            DEFAULT_WORKFLOW_EXECUTION_MODE,
        )
        self.assertEqual(
            _extract_structured_tool_result(primitive_result).value,
            {"result": "mia_garcia_4516"},
        )

        address_args = {
            "order_id": "#W8665881",
            "address1": "123 Elm Street",
            "address2": "Suite 641",
            "city": "Austin",
            "state": "TX",
            "country": "USA",
            "zip": "78712",
        }
        prior_step = BenchmarkStep(
            id="step_00",
            query="Modify the pending order address.",
            expected_tool="modify_pending_order_address",
            expected_args=address_args,
            expected_answer=None,
            depends_on=(),
            source_program=None,
        )
        result = await _call_tool_with_workflow_isolation(
            None,
            SERVER_PATH,
            "get_order_details",
            {"order_id": "#W8665881"},
            (prior_step,),
            REFERENCE_PREFIX_REPLAY_MODE,
        )
        value = _extract_structured_tool_result(result).value
        self.assertEqual(value["address"], {
            key: address_args[key]
            for key in (
                "address1",
                "address2",
                "city",
                "state",
                "country",
                "zip",
            )
        })

    async def test_non_retail_workflow_does_not_replay_prefix_on_wrong_route(self):
        coding_prior_step = BenchmarkStep(
            id="step_00",
            query="Calculate an intermediate value.",
            expected_tool="calculator",
            expected_args={"expression": "2 + 2"},
            expected_answer={"result": 4},
            depends_on=(),
            source_program=None,
        )
        result = await _call_tool_with_workflow_isolation(
            None,
            SERVER_PATH,
            "find_user_id_by_email",
            {"email": "mia.garcia2723@example.com"},
            (coding_prior_step,),
            DEFAULT_WORKFLOW_EXECUTION_MODE,
        )
        self.assertEqual(
            _extract_structured_tool_result(result).value,
            {"result": "mia_garcia_4516"},
        )


if __name__ == "__main__":
    unittest.main()
