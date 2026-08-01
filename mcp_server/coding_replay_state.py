from __future__ import annotations

import json
import re
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

CODING_REPLAY_TOOL_NAMES = frozenset(
    {
        "code_replay_sweagent_shell",
        "code_replay_sweagent_file_view",
        "code_replay_sweagent_file_search",
        "code_replay_sweagent_file_edit",
        "code_replay_sweagent_submit",
    }
)

CODING_REPLAY_FIXTURE_ID = "nebius-sweagent-benchmark-replay-v1"
CODING_REPLAY_FIXTURE_VERSION = "coding_nebius_sweagent_benchmark_replay_v1"
CODING_REPLAY_RECORD_COUNT = 139

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODING_REPLAY_FIXTURE_PATH = (
    _PROJECT_ROOT
    / "benchmark"
    / "coding"
    / "fixtures"
    / "coding_nebius_sweagent_benchmark_replay.json"
)
CODING_REPLAY_FIXTURE_PATHS = (CODING_REPLAY_FIXTURE_PATH,)

_MAX_FIXTURE_BYTES = 2 * 1024 * 1024
_MAX_REPLAY_RECORDS = 1_000
_MAX_IDENTIFIER_LENGTH = 512
_MAX_STEP_INDEX = 1_000_000
_MAX_ARGUMENT_BYTES = 512 * 1024
_MAX_OBSERVATION_CHARS = 256 * 1024
_MAX_OBSERVATION_BYTES = 512 * 1024
_MAX_ORIGINAL_OBSERVATION_CHARS = 1_000_000_000
_MAX_SOURCE_BYTES = 256 * 1024
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

_REPLAY_LOCK = threading.RLock()
_FIXTURE_INVENTORY: tuple[dict[str, Any], ...] | None = None
_REPLAY_INDEX: dict[str, dict[str, Any]] | None = None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _canonical_json(value: Any, field: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} must contain finite JSON values.") from exc


def _bounded_json_object(
    value: object,
    field: str,
    maximum_bytes: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} must be an object.")
    serialized = _canonical_json(value, field)
    if len(serialized.encode("utf-8")) > maximum_bytes:
        raise RuntimeError(f"{field} exceeds its bounded serialized size.")
    return value


def _bounded_identifier(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_IDENTIFIER_LENGTH
        or "\x00" in value
    ):
        raise RuntimeError(
            f"{field} must be a non-empty NUL-free string of at most "
            f"{_MAX_IDENTIFIER_LENGTH} characters."
        )
    return value


def _validate_observation(
    value: object,
    *,
    fixture_path: Path,
    record_id: str,
) -> dict[str, Any]:
    field = f"{fixture_path.name} record {record_id!r} observation"
    value = _bounded_json_object(value, field, _MAX_OBSERVATION_BYTES)

    required = {
        "available",
        "excerpt",
        "full_sha256",
        "original_chars",
        "truncated",
    }
    missing = sorted(required - set(value))
    if missing:
        raise RuntimeError(f"{field} is missing: {', '.join(missing)}.")

    available = value["available"]
    if not isinstance(available, bool):
        raise RuntimeError(f"{field}.available must be a boolean.")

    excerpt = value["excerpt"]
    if (
        not isinstance(excerpt, str)
        or len(excerpt) > _MAX_OBSERVATION_CHARS
        or "\x00" in excerpt
    ):
        raise RuntimeError(
            f"{field}.excerpt must be a bounded NUL-free string."
        )

    full_sha256 = value["full_sha256"]
    if not isinstance(full_sha256, str) or not _SHA256_PATTERN.fullmatch(
        full_sha256
    ):
        raise RuntimeError(f"{field}.full_sha256 must be a lowercase SHA-256.")

    original_chars = value["original_chars"]
    if (
        isinstance(original_chars, bool)
        or not isinstance(original_chars, int)
        or original_chars < len(excerpt)
        or original_chars > _MAX_ORIGINAL_OBSERVATION_CHARS
    ):
        raise RuntimeError(
            f"{field}.original_chars must be an integer no smaller than "
            "the excerpt length."
        )

    truncated = value["truncated"]
    if not isinstance(truncated, bool):
        raise RuntimeError(f"{field}.truncated must be a boolean.")
    if truncated and original_chars == len(excerpt):
        raise RuntimeError(
            f"{field} cannot be marked truncated without omitted characters."
        )
    if not truncated and original_chars != len(excerpt):
        raise RuntimeError(
            f"{field} character count is inconsistent with truncated=false."
        )
    if not available and (
        excerpt
        or original_chars != 0
        or truncated
        or full_sha256 != _EMPTY_SHA256
    ):
        raise RuntimeError(
            f"{field} unavailable observations must use the empty observation "
            "sentinel."
        )

    return value


def _load_fixture(
    fixture_path: Path,
    replay_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    try:
        file_size = fixture_path.stat().st_size
    except OSError as exc:
        raise RuntimeError(
            f"Unable to inspect coding replay fixture: {fixture_path}."
        ) from exc
    if file_size < 2 or file_size > _MAX_FIXTURE_BYTES:
        raise RuntimeError(
            f"Coding replay fixture {fixture_path.name} has an invalid file size."
        )

    try:
        fixture = json.loads(
            fixture_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (
        OSError,
        RecursionError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"Unable to load coding replay fixture: {fixture_path.name}."
        ) from exc
    if not isinstance(fixture, dict):
        raise RuntimeError(
            f"Coding replay fixture {fixture_path.name} must be an object."
        )

    fixture_id = _bounded_identifier(
        fixture.get("fixture_id"),
        f"{fixture_path.name} fixture_id",
    )
    fixture_version = _bounded_identifier(
        fixture.get("fixture_version"),
        f"{fixture_path.name} fixture_version",
    )
    if fixture_id != CODING_REPLAY_FIXTURE_ID:
        raise RuntimeError(
            f"Coding replay fixture {fixture_path.name} has unexpected fixture_id."
        )
    if fixture_version != CODING_REPLAY_FIXTURE_VERSION:
        raise RuntimeError(
            f"Coding replay fixture {fixture_path.name} has unexpected "
            "fixture_version."
        )
    manifest = _bounded_json_object(
        fixture.get("manifest"),
        f"{fixture_path.name} manifest",
        _MAX_MANIFEST_BYTES,
    )
    records = fixture.get("records")
    if not isinstance(records, list):
        raise RuntimeError(
            f"Coding replay fixture {fixture_path.name} records must be a list."
        )
    if not records:
        raise RuntimeError(
            f"Coding replay fixture {fixture_path.name} must contain records."
        )
    if len(records) != CODING_REPLAY_RECORD_COUNT:
        raise RuntimeError(
            f"Coding replay fixture {fixture_path.name} has {len(records)} "
            f"records; expected {CODING_REPLAY_RECORD_COUNT}."
        )
    if len(replay_index) + len(records) > _MAX_REPLAY_RECORDS:
        raise RuntimeError(
            f"Coding replay fixtures may contain at most "
            f"{_MAX_REPLAY_RECORDS} records in total."
        )

    for record_number, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"{fixture_path.name} record {record_number} must be an object."
            )
        record_id = _bounded_identifier(
            record.get("record_id"),
            f"{fixture_path.name} record {record_number} record_id",
        )
        if record_id in replay_index:
            raise RuntimeError(
                f"Coding replay record IDs must be globally unique: {record_id!r}."
            )
        trajectory_id = _bounded_identifier(
            record.get("trajectory_id"),
            f"{fixture_path.name} record {record_id!r} trajectory_id",
        )
        step_index = record.get("step_index")
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or step_index < 0
            or step_index > _MAX_STEP_INDEX
        ):
            raise RuntimeError(
                f"{fixture_path.name} record {record_id!r} step_index must "
                f"be between 0 and {_MAX_STEP_INDEX}."
            )
        tool = record.get("tool")
        if tool not in CODING_REPLAY_TOOL_NAMES:
            raise RuntimeError(
                f"{fixture_path.name} record {record_id!r} has an unknown tool."
            )
        args = _bounded_json_object(
            record.get("args"),
            f"{fixture_path.name} record {record_id!r} args",
            _MAX_ARGUMENT_BYTES,
        )
        observation = _validate_observation(
            record.get("observation"),
            fixture_path=fixture_path,
            record_id=record_id,
        )
        source = _bounded_json_object(
            record.get("source"),
            f"{fixture_path.name} record {record_id!r} source",
            _MAX_SOURCE_BYTES,
        )

        replay_index[record_id] = {
            "record_id": record_id,
            "trajectory_id": trajectory_id,
            "step_index": step_index,
            "tool": tool,
            "args": deepcopy(args),
            "observation": deepcopy(observation),
            "source": deepcopy(source),
            "fixture_id": fixture_id,
            "fixture_version": fixture_version,
        }

    return {
        "fixture_id": fixture_id,
        "fixture_version": fixture_version,
        "record_count": len(records),
        "manifest": deepcopy(manifest),
    }


def _ensure_replay_state() -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, dict[str, Any]],
]:
    """Lazily load and validate the checked-in benchmark replay fixture."""
    global _FIXTURE_INVENTORY, _REPLAY_INDEX
    with _REPLAY_LOCK:
        if _FIXTURE_INVENTORY is None or _REPLAY_INDEX is None:
            replay_index: dict[str, dict[str, Any]] = {}
            fixture = _load_fixture(
                CODING_REPLAY_FIXTURE_PATH,
                replay_index,
            )
            _FIXTURE_INVENTORY = (fixture,)
            _REPLAY_INDEX = replay_index
        return _FIXTURE_INVENTORY, _REPLAY_INDEX


def _reset_replay_state_for_tests() -> None:
    """Clear lazy runtime caches; intended only for isolated tests."""
    global _FIXTURE_INVENTORY, _REPLAY_INDEX
    with _REPLAY_LOCK:
        _FIXTURE_INVENTORY = None
        _REPLAY_INDEX = None


def replay_coding_call(
    tool: str,
    record_id: str,
    trajectory_id: str,
    step_index: int,
) -> dict[str, Any]:
    """Return one allowlisted coding observation without executing its action."""
    if tool not in CODING_REPLAY_TOOL_NAMES:
        raise ValueError("Unknown coding replay tool.")
    if (
        not isinstance(record_id, str)
        or not record_id
        or len(record_id) > _MAX_IDENTIFIER_LENGTH
        or "\x00" in record_id
    ):
        raise ValueError("record_id must be a bounded non-empty NUL-free string.")
    if (
        not isinstance(trajectory_id, str)
        or not trajectory_id
        or len(trajectory_id) > _MAX_IDENTIFIER_LENGTH
        or "\x00" in trajectory_id
    ):
        raise ValueError(
            "trajectory_id must be a bounded non-empty NUL-free string."
        )
    if (
        isinstance(step_index, bool)
        or not isinstance(step_index, int)
        or step_index < 0
        or step_index > _MAX_STEP_INDEX
    ):
        raise ValueError("step_index must be a non-negative integer.")
    with _REPLAY_LOCK:
        _, replay_index = _ensure_replay_state()
        record = replay_index.get(record_id)
        if record is None:
            raise ValueError("No allowlisted coding replay exists for record_id.")
        if record["tool"] != tool:
            raise ValueError("record_id is not allowlisted for this replay tool.")
        if record["trajectory_id"] != trajectory_id:
            raise ValueError("trajectory_id does not match the replay record.")
        if record["step_index"] != step_index:
            raise ValueError("step_index does not match the replay record.")
        return {
            "record_id": record["record_id"],
            "tool": record["tool"],
            "arguments": deepcopy(record["args"]),
            "observation": deepcopy(record["observation"]),
            "offline_replay": True,
            "network_access": False,
            "process_executed": False,
            "mutation_applied": False,
            "provenance": {
                "fixture_id": record["fixture_id"],
                "fixture_version": record["fixture_version"],
                "source": deepcopy(record["source"]),
            },
        }


def snapshot_coding_replay_state() -> dict[str, Any]:
    """Return detached metadata for the checked-in benchmark replay fixture."""
    with _REPLAY_LOCK:
        fixtures, replay_index = _ensure_replay_state()
        return {
            "record_count": len(replay_index),
            "replay_tools": sorted(CODING_REPLAY_TOOL_NAMES),
            "fixtures": deepcopy(list(fixtures)),
        }
