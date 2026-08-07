"""Rescore saved evaluator records without rerunning a model router.

The saved JSONL records contain the model's selected tool and arguments.  This
utility applies the current final-outcome matcher to those recorded calls.  The
optional retail replay is deliberately limited to the read-only order lookup
whose accepted ID spelling changed in PR #29.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from evaluation.evaluate import _build_aggregate_metrics, _final_outcome_record_fields
from evaluation.evaluate import _score_final_outcome
from mcp_server.retail_state import reset_retail_state
from mcp_server.retail_tools import get_order_details


def _is_replayable_retail_order_lookup(record: dict[str, Any]) -> bool:
    """Whether a saved call is exactly the order-ID normalization case."""
    if bool(record.get("execution_success", False)):
        return False
    if record.get("expected_tool") != "get_order_details":
        return False
    if record.get("called_tool") != "get_order_details":
        return False

    selected_args = record.get("selected_args")
    if not isinstance(selected_args, dict):
        return False
    if set(selected_args) != {"order_id"}:
        return False
    order_id = selected_args.get("order_id")
    return isinstance(order_id, str) and re.fullmatch(r"W\d+", order_id) is not None


def _replay_retail_order_lookup(record: dict[str, Any]) -> dict[str, Any] | None:
    """Replay only the saved, failed unprefixed-TAU2-order-ID lookup case."""
    if not _is_replayable_retail_order_lookup(record):
        return None

    reset_retail_state()
    try:
        return get_order_details(**record["selected_args"])
    except (TypeError, ValueError):
        # An invented ID can share the old unprefixed spelling. It was not fixed
        # by normalization, so retain its recorded failure instead of aborting.
        return None


def rescore_records(
    records: list[dict[str, Any]], *, replay_retail_order_lookups: bool = False
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return rescored records and IDs whose outcome changed from false to true."""
    rescored: list[dict[str, Any]] = []
    corrected_sample_ids: list[str] = []

    for original in records:
        record = deepcopy(original)
        tool_result_value = record.get("tool_result_value")
        execution_success = bool(record.get("execution_success", False))
        extraction_diagnostic = None

        if replay_retail_order_lookups:
            replayed = _replay_retail_order_lookup(record)
            if replayed is not None:
                tool_result_value = replayed
                execution_success = True
                record["tool_result_value"] = replayed
                record["tool_error"] = None

        called_tool = record.get("called_tool")
        no_tool_call = called_tool is None
        score = _score_final_outcome(
            expected_answer=record.get("expected_answer"),
            tool_result_value=tool_result_value,
            result_extraction_diagnostic=extraction_diagnostic,
            domain=str(record.get("domain", "unspecified")),
            call_predicted_tools=True,
            no_tool_call=no_tool_call,
            execution_success=execution_success,
            expected_tool=record.get("expected_tool"),
            called_tool=called_tool,
        )
        new_fields = _final_outcome_record_fields(score)
        if record.get("final_outcome_correct") is False and score.correct is True:
            corrected_sample_ids.append(str(record.get("sample_id", "<unknown>")))
        record["execution_success"] = execution_success
        record.update(new_fields)
        rescored.append(record)

    return rescored, corrected_sample_ids


def _is_false_to_true_correction(
    original: dict[str, Any], rescored: dict[str, Any]
) -> bool:
    return (
        original.get("final_outcome_correct") is False
        and rescored.get("final_outcome_correct") is True
    )


def change_breakdown(
    original_records: list[dict[str, Any]], rescored_records: list[dict[str, Any]]
) -> dict[str, int]:
    """Count false-to-true corrections attributable to each PR #29 change."""
    finance_alias_changes = 0
    retail_order_id_changes = 0

    for original, rescored in zip(original_records, rescored_records, strict=True):
        if not _is_false_to_true_correction(original, rescored):
            continue

        expected_answer = original.get("expected_answer")
        actual_result = original.get("tool_result_value")
        if (
            original.get("expected_tool") == "finance_query_table"
            and original.get("called_tool") == "finance_query_table"
            and isinstance(expected_answer, dict)
            and isinstance(actual_result, dict)
            and isinstance(expected_answer.get("columns"), list)
            and isinstance(actual_result.get("columns"), list)
            and expected_answer["columns"] != actual_result["columns"]
        ):
            finance_alias_changes += 1
        if _is_replayable_retail_order_lookup(original):
            retail_order_id_changes += 1

    return {
        "finance_alias_changes": finance_alias_changes,
        "retail_order_id_changes": retail_order_id_changes,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object.")
            records.append(value)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rescore saved LayerMCP evaluator JSONL records."
    )
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replay-retail-order-lookups",
        action="store_true",
        help=(
            "Replay recorded get_order_details calls against the local retail "
            "fixture before rescoring."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    records = _load_records(args.samples)
    rescored, corrected_sample_ids = rescore_records(
        records,
        replay_retail_order_lookups=args.replay_retail_order_lookups,
    )
    _write_jsonl(args.output, rescored)

    input_sha256 = hashlib.sha256(args.samples.read_bytes()).hexdigest()
    report = {
        "input_samples": str(args.samples),
        "input_sha256": input_sha256,
        "output_samples": str(args.output),
        "replay_retail_order_lookups": args.replay_retail_order_lookups,
        "false_to_true_correction_ids": corrected_sample_ids,
        "false_to_true_correction_count": len(corrected_sample_ids),
        "change_breakdown": change_breakdown(records, rescored),
        **_build_aggregate_metrics(rescored),
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
