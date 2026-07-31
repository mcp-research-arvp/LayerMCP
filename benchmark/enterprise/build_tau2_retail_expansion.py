#!/usr/bin/env python3
"""Build deduplicated tau2-native retail single-step benchmark rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "benchmark"
    / "enterprise"
    / "tool_routing_enterprise_tau2_public_adapted.json"
)
PROVENANCE_PATH = (
    PROJECT_ROOT / "mcp_server" / "fixtures" / "tau2_retail_provenance.json"
)
TAU2_REVISION = "363133ada1936491fb5bcec33cd62c3518a99f65"
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
EXCLUDED_ACTIONS = {"get_item_details", "modify_pending_order_payment"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_revision(path: Path) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != TAU2_REVISION:
        raise RuntimeError(f"{path} is at {actual}, expected {TAU2_REVISION}")


def _retail_functions() -> tuple[
    dict[str, Callable[..., object]], Callable[[], dict[str, Any]]
]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from mcp_server.retail_state import reset_retail_state
    from mcp_server import retail_tools

    return {
        name: getattr(retail_tools, name) for name in sorted(SUPPORTED_TOOLS)
    }, reset_retail_state


def _query(tool: str, args: dict[str, Any]) -> str:
    if tool == "find_user_id_by_email":
        return f"Find the retail user ID associated with {args['email']}."
    if tool == "find_user_id_by_name_zip":
        return (
            f"Find the retail user ID for {args['first_name']} "
            f"{args['last_name']} in ZIP {args['zip']}."
        )
    if tool == "get_user_details":
        return f"Retrieve the retail customer record for {args['user_id']}."
    if tool == "get_order_details":
        return f"Retrieve the current details for retail order {args['order_id']}."
    if tool == "get_product_details":
        return f"Retrieve inventory and variant details for product {args['product_id']}."
    if tool == "cancel_pending_order":
        return f"Cancel pending order {args['order_id']} because it is {args['reason']}."
    if tool == "modify_pending_order_items":
        return (
            f"In pending order {args['order_id']}, replace "
            f"{', '.join(args['item_ids'])} with {', '.join(args['new_item_ids'])} "
            f"using payment method {args['payment_method_id']}."
        )
    if tool == "modify_pending_order_address":
        return (
            f"Change pending order {args['order_id']}'s delivery address to "
            f"{args['address1']}, {args['address2']}, {args['city']}, "
            f"{args['state']}, {args['country']} {args['zip']}."
        )
    if tool == "modify_user_address":
        return (
            f"Change {args['user_id']}'s saved address to {args['address1']}, "
            f"{args['address2']}, {args['city']}, {args['state']}, "
            f"{args['country']} {args['zip']}."
        )
    if tool == "return_delivered_order_items":
        return (
            f"Return {', '.join(args['item_ids'])} from delivered order "
            f"{args['order_id']} to payment method {args['payment_method_id']}."
        )
    if tool == "exchange_delivered_order_items":
        return (
            f"Exchange {', '.join(args['item_ids'])} in delivered order "
            f"{args['order_id']} for {', '.join(args['new_item_ids'])}, using "
            f"payment method {args['payment_method_id']}."
        )
    if tool == "transfer_to_human_agents":
        return f'Escalate this retail request with the summary: "{args["summary"]}"'
    raise KeyError(tool)


def _structured(value: object) -> object:
    return value if isinstance(value, dict) else {"result": value}


def _expected_subset(tool: str, value: object) -> object:
    result = _structured(value)
    if tool.startswith("find_user_id_") or tool == "transfer_to_human_agents":
        return result
    assert isinstance(result, dict)
    fields = {
        "get_user_details": ("user_id", "email", "orders"),
        "get_order_details": ("order_id", "user_id", "status", "items"),
        "get_product_details": ("product_id", "name", "variants"),
        "cancel_pending_order": (
            "order_id",
            "status",
            "cancel_reason",
            "payment_history",
        ),
        "modify_pending_order_items": (
            "order_id",
            "status",
            "items",
            "payment_history",
        ),
        "modify_pending_order_address": ("order_id", "status", "address"),
        "modify_user_address": ("user_id", "address"),
        "return_delivered_order_items": (
            "order_id",
            "status",
            "return_items",
            "return_payment_method_id",
        ),
        "exchange_delivered_order_items": (
            "order_id",
            "status",
            "exchange_items",
            "exchange_new_items",
            "exchange_payment_method_id",
            "exchange_price_difference",
        ),
    }[tool]
    return {field: result[field] for field in fields}


def build(raw_root: Path) -> list[dict[str, Any]]:
    tau2_root = raw_root / "tau2-bench"
    _require_revision(tau2_root)
    retail_root = tau2_root / "data" / "tau2" / "domains" / "retail"
    tasks = json.loads((retail_root / "tasks.json").read_text(encoding="utf-8"))
    splits = json.loads(
        (retail_root / "split_tasks.json").read_text(encoding="utf-8")
    )
    split_by_id = {
        str(task_id): split
        for split in ("train", "test")
        for task_id in splits[split]
    }
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    fixture_hash = provenance["derived_fixture_sha256"]
    functions, reset_state = _retail_functions()

    rows: list[dict[str, Any]] = []
    seen_calls: set[tuple[str, str]] = set()
    rejected: Counter[str] = Counter()
    per_tool_index: Counter[str] = Counter()
    for task in tasks:
        task_id = str(task["id"])
        for action_index, action in enumerate(
            task.get("evaluation_criteria", {}).get("actions", [])
        ):
            tool = action["name"]
            if tool in EXCLUDED_ACTIONS or tool not in SUPPORTED_TOOLS:
                rejected[tool] += 1
                continue
            args = action["arguments"]
            call_key = (tool, _canonical_json(args))
            if call_key in seen_calls:
                continue
            try:
                inspect.signature(functions[tool]).bind(**args)
                reset_state()
                first = functions[tool](**args)
                reset_state()
                second = functions[tool](**args)
            except (TypeError, ValueError, KeyError):
                rejected[f"{tool}:not_standalone_executable"] += 1
                continue
            if first != second:
                raise RuntimeError(
                    f"Nondeterministic result for task {task_id}, action {action_index}"
                )

            seen_calls.add(call_key)
            per_tool_index[tool] += 1
            source_action_id = action.get(
                "action_id", f"{task_id}_{action_index}"
            )
            rows.append(
                {
                    "id": (
                        f"enterprise_tau2_{tool}_"
                        f"{per_tool_index[tool]:03d}"
                    ),
                    "domain": "enterprise_automation",
                    "task_type": "single_tool_routing",
                    "difficulty": "medium",
                    "source": "public_adapted",
                    "query": _query(tool, args),
                    "expected_tool": tool,
                    "expected_args": args,
                    "expected_answer": _expected_subset(tool, first),
                    "perturbation_type": "public_task_action_adaptation",
                    "notes": (
                        "One deduplicated tau2 retail gold action represented as "
                        "an independently executable single-tool query."
                    ),
                    "source_dataset": "tau2_bench_retail",
                    "source_revision": TAU2_REVISION,
                    "source_split": split_by_id[task_id],
                    "source_task_id": task_id,
                    "source_action_index": action_index,
                    "source_action_id": source_action_id,
                    "source_action": tool,
                    "source_expected_args": args,
                    "source_hash": _sha256_json(task),
                    "fixture_hash": fixture_hash,
                    "source_license": "MIT",
                    "transformation_notes": (
                        "The gold action name and native tau2 arguments are "
                        "unchanged. The surrounding workflow was rewritten as a "
                        "concise standalone request; only calls executable from "
                        "a fresh pinned tau2 retail state were retained."
                    ),
                    "entity_mapping_notes": (
                        "No entity remapping was applied; expected_args contain "
                        "native tau2 identifiers and values."
                    ),
                }
            )
    reset_state()
    if not rows:
        raise RuntimeError("No tau2-native rows were built")
    if len(rows) != len(seen_calls):
        raise RuntimeError("Generated rows are not unique by tool and arguments")
    print("Excluded:", dict(sorted(rejected.items())))
    print("Counts:", dict(sorted(Counter(r["expected_tool"] for r in rows).items())))
    return rows


def main() -> None:
    default_raw = Path(os.environ.get("SCRATCH", "")) / "layermcp" / "raw_sources"
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=default_raw)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = build(args.raw_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
