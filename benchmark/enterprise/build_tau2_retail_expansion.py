#!/usr/bin/env python3
"""Build the pinned tau2 retail single-step public-adapted expansion."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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
TAU2_REVISION = "363133ada1936491fb5bcec33cd62c3518a99f65"
TARGET_COUNTS = {
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


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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

    return {
        "find_user_id_by_email": find_user_id_by_email,
        "find_user_id_by_name_zip": find_user_id_by_name_zip,
        "get_user_details": get_user_details,
        "get_order_details": get_order_details,
        "get_product_details": get_product_details,
        "cancel_pending_order": cancel_pending_order,
        "modify_pending_order_items": modify_pending_order_items,
        "modify_pending_order_address": modify_pending_order_address,
        "modify_user_address": modify_user_address,
        "return_delivered_order_items": return_delivered_order_items,
        "exchange_delivered_order_items": exchange_delivered_order_items,
        "transfer_to_human_agents": transfer_to_human_agents,
    }, reset_retail_state


def _profiles(tool: str) -> list[dict[str, Any]]:
    profiles: dict[str, list[dict[str, Any]]] = {
        "find_user_id_by_email": [
            {"email": "yusuf.rossi@example.com"},
            {"email": "mei.kovacs@example.com"},
        ],
        "find_user_id_by_name_zip": [
            {
                "first_name": "Yusuf",
                "last_name": "Rossi",
                "zip": "19122",
            },
            {
                "first_name": "Mei",
                "last_name": "Kovacs",
                "zip": "28236",
            },
        ],
        "get_user_details": [
            {"user_id": "USER-YUSUF"},
            {"user_id": "USER-MEI"},
        ],
        "get_order_details": [
            {"order_id": "RET-1001"},
            {"order_id": "RET-1002"},
            {"order_id": "RET-1004"},
            {"order_id": "RET-2001"},
            {"order_id": "RET-2002"},
        ],
        "get_product_details": [
            {"product_id": "PROD-KEYBOARD"},
            {"product_id": "PROD-BOTTLE"},
            {"product_id": "PROD-LAMP"},
        ],
        "cancel_pending_order": [
            {"order_id": "RET-1001", "reason": "no longer needed"},
            {"order_id": "RET-1002", "reason": "ordered by mistake"},
        ],
        "modify_pending_order_items": [
            {
                "order_id": "RET-1001",
                "item_ids": ["ITEM-KB-LINEAR"],
                "new_item_ids": ["ITEM-KB-CLICKY"],
                "payment_method_id": "GIFT-YUSUF",
            },
            {
                "order_id": "RET-1001",
                "item_ids": ["ITEM-BOTTLE-500"],
                "new_item_ids": ["ITEM-BOTTLE-1000"],
                "payment_method_id": "CARD-YUSUF",
            },
            {
                "order_id": "RET-1002",
                "item_ids": ["ITEM-LAMP-USB"],
                "new_item_ids": ["ITEM-LAMP-BATTERY"],
                "payment_method_id": "GIFT-YUSUF",
            },
        ],
        "modify_pending_order_address": [
            {
                "order_id": "RET-1001",
                "address1": "101 Highway",
                "address2": "",
                "city": "New York",
                "state": "NY",
                "country": "US",
                "zip": "10001",
            },
            {
                "order_id": "RET-1002",
                "address1": "123 Elm Street",
                "address2": "Suite 641",
                "city": "Austin",
                "state": "TX",
                "country": "US",
                "zip": "78712",
            },
        ],
        "modify_user_address": [
            {
                "user_id": "USER-YUSUF",
                "address1": "101 Highway",
                "address2": "",
                "city": "New York",
                "state": "NY",
                "country": "US",
                "zip": "10001",
            },
            {
                "user_id": "USER-MEI",
                "address1": "157 Oak Street",
                "address2": "Suite 258",
                "city": "Phoenix",
                "state": "AZ",
                "country": "US",
                "zip": "85033",
            },
        ],
        "return_delivered_order_items": [
            {
                "order_id": "RET-2001",
                "item_ids": ["ITEM-BOTTLE-500"],
                "payment_method_id": "CARD-MEI",
            },
            {
                "order_id": "RET-2001",
                "item_ids": ["ITEM-KB-LINEAR"],
                "payment_method_id": "CARD-MEI",
            },
            {
                "order_id": "RET-2001",
                "item_ids": ["ITEM-KB-LINEAR", "ITEM-BOTTLE-500"],
                "payment_method_id": "GIFT-MEI",
            },
        ],
        "exchange_delivered_order_items": [
            {
                "order_id": "RET-2001",
                "item_ids": ["ITEM-KB-LINEAR"],
                "new_item_ids": ["ITEM-KB-CLICKY"],
                "payment_method_id": "CARD-MEI",
            },
            {
                "order_id": "RET-2001",
                "item_ids": ["ITEM-BOTTLE-500"],
                "new_item_ids": ["ITEM-BOTTLE-1000"],
                "payment_method_id": "CARD-MEI",
            },
        ],
        "transfer_to_human_agents": [],
    }
    return profiles[tool]


def _local_args(
    tool: str,
    index: int,
    source_task: dict[str, Any],
) -> dict[str, Any]:
    if tool == "transfer_to_human_agents":
        reason = source_task["user_scenario"]["instructions"]["reason_for_call"]
        return {
            "summary": (
                f"tau2 retail task {source_task['id']} requires human assistance: "
                f"{reason}"
            )
        }
    profiles = _profiles(tool)
    return dict(profiles[index % len(profiles)])


def _query(tool: str, args: dict[str, Any]) -> str:
    if tool == "find_user_id_by_email":
        return f"Resolve retail customer email {args['email']} to the user ID."
    if tool == "find_user_id_by_name_zip":
        return (
            f"Find the retail user ID for {args['first_name']} {args['last_name']} "
            f"in ZIP {args['zip']}."
        )
    if tool == "get_user_details":
        return f"Retrieve the complete retail customer record for {args['user_id']}."
    if tool == "get_order_details":
        return f"Retrieve retail order {args['order_id']} and its current status and contents."
    if tool == "get_product_details":
        return f"Inspect the available variants for retail product {args['product_id']}."
    if tool == "cancel_pending_order":
        return (
            f"Cancel pending order {args['order_id']} because it is "
            f"{args['reason']}."
        )
    if tool == "modify_pending_order_items":
        return (
            f"In pending order {args['order_id']}, replace "
            f"{', '.join(args['item_ids'])} with {', '.join(args['new_item_ids'])} "
            f"using {args['payment_method_id']} for any price difference."
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
            f"{args['order_id']} and refund {args['payment_method_id']}."
        )
    if tool == "exchange_delivered_order_items":
        return (
            f"Exchange {', '.join(args['item_ids'])} in delivered order "
            f"{args['order_id']} for {', '.join(args['new_item_ids'])}, using "
            f"{args['payment_method_id']} for the difference."
        )
    if tool == "transfer_to_human_agents":
        return f'Escalate this retail request with the exact summary: "{args["summary"]}"'
    raise KeyError(tool)


def _structured(value: object) -> object:
    return value if isinstance(value, dict) else {"result": value}


def _expected_subset(tool: str, value: object) -> object:
    result = _structured(value)
    if tool.startswith("find_user_id_"):
        return result
    assert isinstance(result, dict)
    fields = {
        "get_user_details": ("user_id", "email", "order_ids"),
        "get_order_details": ("order_id", "user_id", "status", "total"),
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
            "modified_items",
            "modification_payment_method_id",
            "modification_price_difference",
            "total",
        ),
        "modify_pending_order_address": (
            "order_id",
            "status",
            "address_modified",
            "address",
        ),
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
        "transfer_to_human_agents": (
            "transfer_requested",
            "summary",
            "status",
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
        task_id: split
        for split in ("train", "test")
        for task_id in splits[split]
    }
    selected: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        tool: [] for tool in TARGET_COUNTS
    }
    for task in tasks:
        for action in task["evaluation_criteria"]["actions"]:
            tool = action["name"]
            if tool in selected and len(selected[tool]) < TARGET_COUNTS[tool]:
                selected[tool].append((task, action))
    for tool, target in TARGET_COUNTS.items():
        if len(selected[tool]) != target:
            raise RuntimeError(f"Found only {len(selected[tool])}/{target} {tool} actions")

    functions, reset_state = _retail_functions()
    rows: list[dict[str, Any]] = []
    for tool, pairs in selected.items():
        for index, (task, action) in enumerate(pairs):
            args = _local_args(tool, index, task)
            reset_state()
            first = functions[tool](**args)
            reset_state()
            second = functions[tool](**args)
            if first != second:
                raise RuntimeError(f"Nondeterministic result for {task['id']} {action['action_id']}")
            source_args = action["arguments"]
            rows.append(
                {
                    "id": f"enterprise_tau2_{tool}_{index + 1:03d}",
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
                        "One tau2 retail gold action adapted to the bounded "
                        "LayerMCP retail fixture."
                    ),
                    "source_dataset": "tau2_bench_retail",
                    "source_revision": TAU2_REVISION,
                    "source_split": split_by_id[str(task["id"])],
                    "source_task_id": str(task["id"]),
                    "source_action_id": action["action_id"],
                    "source_action": tool,
                    "source_expected_args": source_args,
                    "source_hash": _sha256_json(task),
                    "source_license": "MIT",
                    "transformation_notes": (
                        "The source action name and intent are preserved. The "
                        "standalone query is a concise LayerMCP adaptation of the "
                        "action within its original dialogue workflow."
                    ),
                    "entity_mapping_notes": (
                        "tau2 entity identifiers are not present in the bounded "
                        "LayerMCP retail fixture. Source arguments "
                        f"{json.dumps(source_args, sort_keys=True)} were mapped to "
                        f"local arguments {json.dumps(args, sort_keys=True)}."
                    ),
                }
            )
    reset_state()
    if len(rows) != 127:
        raise RuntimeError(f"Expected 127 rows, built {len(rows)}")
    if Counter(row["expected_tool"] for row in rows) != Counter(TARGET_COUNTS):
        raise RuntimeError("Unexpected expected-tool distribution")
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
