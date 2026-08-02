"""Build source-faithful tau2 retail public workflow benchmark rows.

This builder keeps original tau2 retail user-scenario fields and converts only
fully supported multi-action workflows into LayerMCP expected_steps records.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_ROOT = (
    PROJECT_ROOT.parent
    / "raw_sources"
    / "tau2-bench"
    / "data"
    / "tau2"
    / "domains"
    / "retail"
)
OUTPUT_PATH = PROJECT_ROOT / "benchmark" / "enterprise" / "enterprise_public_workflows.json"

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

ACTION_DESCRIPTIONS = {
    "find_user_id_by_email": "identify the user from their email address",
    "find_user_id_by_name_zip": "identify the user from their name and ZIP code",
    "get_user_details": "retrieve the user's saved profile details",
    "get_order_details": "retrieve full order details",
    "get_product_details": "retrieve product details",
    "cancel_pending_order": "cancel a pending order",
    "modify_pending_order_items": "modify items in a pending order",
    "modify_pending_order_address": (
        "modify the shipping address for a pending order"
    ),
    "modify_user_address": "modify the user's saved address",
    "return_delivered_order_items": "return delivered order items",
    "exchange_delivered_order_items": "exchange delivered order items",
    "transfer_to_human_agents": "transfer the case to a human agent",
}

EXPECTED_TAU2_REVISION = "363133ada1936491fb5bcec33cd62c3518a99f65"


def _sha256_json(value: Any) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _split_map(splits: dict[str, list[Any]]) -> dict[str, str]:
    """Prefer train/test labels over base, because base contains all tasks."""

    out: dict[str, str] = {}

    for split in ("train", "test"):
        for task_id in splits.get(split, []):
            out[str(task_id)] = split

    for task_id in splits.get("base", []):
        out.setdefault(str(task_id), "base")

    return out


def _scenario_query(task: dict[str, Any]) -> str:
    instructions = (
        task.get("user_scenario", {})
        .get("instructions", {})
    )

    parts = []
    for label, key in [
        ("Task instructions", "task_instructions"),
        ("Domain", "domain"),
        ("Known info", "known_info"),
        ("Unknown info", "unknown_info"),
        ("Reason for call", "reason_for_call"),
    ]:
        value = instructions.get(key)
        if value:
            parts.append(f"{label}: {value}")

    if not parts:
        raise ValueError(f"Task {task.get('id')} has no user_scenario instruction text")

    return "\n".join(parts)


def _load_retail_runtime():
    from mcp_server.retail_state import reset_retail_state
    from mcp_server import retail_tools

    functions = {
        name: getattr(retail_tools, name)
        for name in sorted(SUPPORTED_TOOLS)
    }
    return functions, reset_retail_state


def _step_query(tool_name: str) -> str:
    return (
        "Perform the next grounded retail workflow operation: "
        f"{ACTION_DESCRIPTIONS[tool_name]}."
    )


def _step_prompt_context(
    action_index: int,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    facts = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    return (
        f"Source action index: {action_index}\n"
        f"Operation: {ACTION_DESCRIPTIONS[tool_name]}.\n"
        f"Authoritative source-action facts for this step: {facts}\n"
        "Treat these facts as authoritative for this current teacher-forced "
        "routing step. Select only the tool call needed for this operation. "
        "Do not perform or anticipate any earlier or later workflow steps."
    )


def _mcp_expected_answer(value: Any) -> Any:
    """Match FastMCP structuredContent for primitive function returns."""
    if isinstance(value, dict):
        return value
    return {"result": value}


def main() -> None:
    tau2_repo = RAW_ROOT.parents[3]
    actual_revision = _git_revision(tau2_repo)
    if actual_revision != EXPECTED_TAU2_REVISION:
        raise RuntimeError(
            f"tau2-bench revision mismatch: {actual_revision} != {EXPECTED_TAU2_REVISION}"
        )

    tasks = json.loads((RAW_ROOT / "tasks.json").read_text(encoding="utf-8"))
    splits = json.loads((RAW_ROOT / "split_tasks.json").read_text(encoding="utf-8"))
    split_by_id = _split_map(splits)

    functions, reset_state = _load_retail_runtime()

    rows = []
    skipped = Counter()
    failure_examples = []

    for task in tasks:
        task_id = str(task["id"])
        actions = task.get("evaluation_criteria", {}).get("actions", [])

        if len(actions) < 2:
            skipped["not_multi_action"] += 1
            continue

        unsupported = [
            action.get("name")
            for action in actions
            if action.get("name") not in SUPPORTED_TOOLS
        ]
        if unsupported:
            skipped["unsupported_action"] += 1
            continue

        reset_state()

        workflow_failed = False
        failure_reason = None
        steps = []
        for action_index, action in enumerate(actions):
            tool_name = action["name"]
            args = action.get("arguments") or {}
            try:
                expected_answer = functions[tool_name](**args)
            except Exception as exc:
                workflow_failed = True
                failure_reason = {
                    "task_id": task_id,
                    "action_index": action_index,
                    "tool": tool_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                skipped[f"execution_error:{tool_name}:{type(exc).__name__}"] += 1
                if len(failure_examples) < 10:
                    failure_examples.append(failure_reason)
                break

            action_id = action.get("action_id", f"{task_id}_{action_index}")
            step_id = f"step_{action_index:02d}"

            steps.append(
                {
                    "id": step_id,
                    "query": _step_query(tool_name),
                    "prompt_context": _step_prompt_context(
                        action_index,
                        tool_name,
                        args,
                    ),
                    "expected_tool": tool_name,
                    "expected_args": args,
                    "expected_answer": _mcp_expected_answer(expected_answer),
                    "depends_on": [],
                    "source_action_id": action_id,
                    "source_action_index": action_index,
                    "source_action": tool_name,
                    "source_action_role": "reference_trajectory",
                    "source_expected_args": args,
                    "source_action_info": action.get("info"),
                }
            )

        if workflow_failed:
            continue

        row = {
            "id": f"enterprise_public_tau2_workflow_{int(task_id):03d}",
            "domain": "enterprise_automation",
            "task_type": "multi_step_tool_routing",
            "source": "public_enterprise_workflow",
            "source_dataset": "tau2_bench_retail",
            "source_revision": EXPECTED_TAU2_REVISION,
            "source_split": split_by_id.get(task_id, "unknown"),
            "source_task_id": task_id,
            "source_hash": _sha256_json(task),
            "source_license": "MIT",
            "query_origin": "tau2_user_scenario_instruction_fields",
            "tool_sequence_origin": "tau2_evaluation_criteria_actions",
            "source_action_role": "reference_trajectory",
            "perturbation_type": "source_workflow_format_adaptation",
            "difficulty": "public_workflow",
            "query": _scenario_query(task),
            "expected_steps": steps,
            "source_user_scenario": task.get("user_scenario"),
            "source_description": task.get("description"),
            "source_reward_basis": task.get("evaluation_criteria", {}).get("reward_basis"),
            "source_communicate_info": task.get("evaluation_criteria", {}).get("communicate_info"),
            "source_nl_assertions": task.get("evaluation_criteria", {}).get("nl_assertions"),
            "notes": (
                "Original tau2 retail user-scenario fields are preserved. "
                "Only fully supported multi-action workflows are included. "
                "Expected steps preserve one tau2 reference trajectory; they are not "
                "a uniquely correct plan or a direct measure of tau2 task success. "
                "Expected step outputs are deterministic MCP-shaped LayerMCP "
                "retail-tool results from the pinned retail fixture."
            ),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No enterprise public workflow rows were built")

    OUTPUT_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {OUTPUT_PATH}")
    print(f"rows: {len(rows)}")
    print("skipped:", dict(skipped))
    print("failure_examples:", failure_examples)
    print("step-count distribution:", dict(Counter(len(r["expected_steps"]) for r in rows)))
    print("split distribution:", dict(Counter(r["source_split"] for r in rows)))


if __name__ == "__main__":
    main()
