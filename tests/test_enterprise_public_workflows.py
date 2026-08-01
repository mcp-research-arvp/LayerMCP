import json
from collections import Counter
from pathlib import Path

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


def test_enterprise_public_workflow_benchmark_shape():
    rows = _rows()

    assert len(rows) == 69
    assert Counter(row["source_split"] for row in rows) == {
        "train": 45,
        "test": 24,
    }

    for row in rows:
        assert row["domain"] == "enterprise"
        assert row["task_type"] == "multi_step_tool_routing"
        assert row["source"] == "public_enterprise_workflow"
        assert row["source_dataset"] == "tau2_bench_retail"
        assert row["query_origin"] == "tau2_user_scenario_instruction_fields"
        assert row["tool_sequence_origin"] == "tau2_evaluation_criteria_actions"
        assert row["perturbation_type"] == "source_workflow_format_adaptation"

        assert row["query"]
        assert "Reason for call:" in row["query"]

        steps = row["expected_steps"]
        assert len(steps) >= 2

        for index, step in enumerate(steps):
            assert step["id"] == f"step_{index:02d}"
            assert step["expected_tool"] in SUPPORTED_TOOLS
            assert step["source_action"] == step["expected_tool"]
            assert isinstance(step["expected_args"], dict)
            assert step["source_expected_args"] == step["expected_args"]
            assert "expected_answer" in step
            assert step["depends_on"] == [f"step_{i:02d}" for i in range(index)]
            assert step["query"].startswith(
                "Perform the next grounded retail workflow operation: "
            )
            assert step["query"] != (
                "Select the next required retail tool call for this original "
                f"tau2 workflow at source action index {index}."
            )
            assert step["prompt_context"].strip()
            assert f"Source action index: {index}" in step["prompt_context"]
            assert "Authoritative source-action facts" in step["prompt_context"]
            assert "teacher-forced" in step["prompt_context"]
            assert "earlier or later workflow steps" in step["prompt_context"]
            assert "expected_tool" not in step["prompt_context"]
            assert "expected_args" not in step["prompt_context"]


def test_enterprise_public_workflows_replay_against_retail_fixture():
    rows = _rows()
    functions = {name: getattr(retail_tools, name) for name in SUPPORTED_TOOLS}

    for row in rows:
        reset_retail_state()

        for step in row["expected_steps"]:
            actual = functions[step["expected_tool"]](**step["expected_args"])
            assert actual == step["expected_answer"]
