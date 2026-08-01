from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

FINRETRIEVAL_FIXTURE_ID = "finretrieval-replay-v1"
FINRETRIEVAL_FIXTURE_VERSION = "finretrieval_replay_fixture_v1"
FINRETRIEVAL_SOURCE_REVISION = "86a111357cffa181b3ba0a6b5ce94625d4511176"
FINRETRIEVAL_REPLAY_TOOL_NAMES = frozenset(
    {
        "finance_discover_companies",
        "finance_discover_company_series",
        "finance_get_company_fundamentals",
        "finance_search_web_archive",
    }
)

_MAX_REPLAY_RECORDS = 5_000
_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "finance"
    / "fixtures"
    / "finretrieval_replay.json"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _record_key(tool: str, args: dict[str, Any]) -> str:
    return _canonical_json({"tool": tool, "args": args})


def _load_fixture() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Unable to load the FinRetrieval replay fixture.") from exc
    if not isinstance(fixture, dict):
        raise RuntimeError("The FinRetrieval replay fixture must be an object.")
    if fixture.get("fixture_id") != FINRETRIEVAL_FIXTURE_ID:
        raise RuntimeError("The FinRetrieval replay fixture has an unexpected ID.")
    if fixture.get("fixture_version") != FINRETRIEVAL_FIXTURE_VERSION:
        raise RuntimeError("The FinRetrieval replay fixture has an unexpected version.")

    manifest = fixture.get("manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("The FinRetrieval replay fixture requires a manifest.")
    required_manifest = {
        "source_dataset": "FinRetrieval",
        "source_repository": "daloopa/finretrieval",
        "source_revision": FINRETRIEVAL_SOURCE_REVISION,
        "source_license": "MIT",
        "workflow_count": 498,
        "excluded_no_correct_trace_indices": [253, 455],
        "synthetic": False,
        "network_access": False,
    }
    for field, expected in required_manifest.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"The FinRetrieval replay manifest has unexpected {field}."
            )

    records = fixture.get("records")
    if (
        not isinstance(records, list)
        or not records
        or len(records) > _MAX_REPLAY_RECORDS
    ):
        raise RuntimeError("The FinRetrieval replay fixture has invalid records.")
    if manifest.get("replay_record_count") != len(records):
        raise RuntimeError(
            "The FinRetrieval replay manifest record count is inconsistent."
        )

    index: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("A FinRetrieval replay record is not an object.")
        tool = record.get("tool")
        args = record.get("args")
        result = record.get("result")
        source_calls = record.get("source_calls")
        if tool not in FINRETRIEVAL_REPLAY_TOOL_NAMES:
            raise RuntimeError("A FinRetrieval replay record has an unknown tool.")
        if not isinstance(args, dict) or not isinstance(result, dict):
            raise RuntimeError(
                "A FinRetrieval replay record has invalid arguments or result."
            )
        if not isinstance(source_calls, list) or not source_calls:
            raise RuntimeError(
                "A FinRetrieval replay record requires source-call provenance."
            )
        key = _record_key(tool, args)
        if key in index:
            raise RuntimeError(
                "The FinRetrieval replay fixture repeats normalized arguments."
            )
        index[key] = record
    return fixture, index


_FINRETRIEVAL_FIXTURE, _FINRETRIEVAL_INDEX = _load_fixture()


def replay_finretrieval_call(
    tool: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Return one deterministic recorded result for normalized tool arguments."""
    if tool not in FINRETRIEVAL_REPLAY_TOOL_NAMES:
        raise ValueError("Unknown FinRetrieval replay tool.")
    key = _record_key(tool, args)
    record = _FINRETRIEVAL_INDEX.get(key)
    if record is None:
        raise ValueError(
            "No allowlisted FinRetrieval replay exists for those arguments."
        )
    return {
        "tool": tool,
        "arguments": deepcopy(args),
        "result": deepcopy(record["result"]),
        "offline_replay": True,
        "network_access": False,
        "source": "finretrieval_recorded_fixture",
        "provenance": {
            "fixture_id": FINRETRIEVAL_FIXTURE_ID,
            "fixture_version": FINRETRIEVAL_FIXTURE_VERSION,
            "source_dataset": "FinRetrieval",
            "source_repository": "daloopa/finretrieval",
            "source_revision": FINRETRIEVAL_SOURCE_REVISION,
            "source_license": "MIT",
            "classification": "selected_correct_model_trajectory_replay",
            "source_calls": deepcopy(record["source_calls"]),
        },
    }


def snapshot_finretrieval_state() -> dict[str, Any]:
    """Return detached replay inventory metadata for diagnostics and tests."""
    manifest = deepcopy(_FINRETRIEVAL_FIXTURE["manifest"])
    return {
        "fixture_id": FINRETRIEVAL_FIXTURE_ID,
        "fixture_version": FINRETRIEVAL_FIXTURE_VERSION,
        "replay_tools": sorted(FINRETRIEVAL_REPLAY_TOOL_NAMES),
        "record_count": len(_FINRETRIEVAL_INDEX),
        "manifest": manifest,
    }
