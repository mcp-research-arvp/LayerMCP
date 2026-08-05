"""Build the deterministic Nebius coding-trajectory benchmark expansion.

The importer consumes the pinned SWE-agent and SWE-rebench/OpenHands parquet
releases, selects two fixed 500-workflow candidate batches, retains only
workflows containing at most five calls, and writes the resulting benchmarks
plus one benchmark-scoped, inert SWE-agent replay fixture. DuckDB is an
importer-only dependency.
Recorded commands, edits, tests, and submissions are never executed by the
LayerMCP runtime. Benchmark calls contain only replay coordinates; exact source
arguments for retained calls remain in the checked-in fixture and are resolved
after coordinate validation. Full 500-workflow source fixtures are not emitted.

SWE-agent is not a native function-calling dataset. Its model discussion is
discarded and the final fenced action in each ``role=ai`` message is parsed
mechanically. OpenHands ``think`` calls and assistant prose are discarded so
that model chain-of-thought is not redistributed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


SWEAGENT_DATASET = "nebius/SWE-agent-trajectories"
SWEAGENT_REVISION = "68195a1450865274106246d0d0296a1d6807b88e"
SWEAGENT_TOTAL = 80_036
SWEAGENT_SUCCESSFUL = 13_389
SWEAGENT_UNIQUE_SUCCESSFUL = 838
SWEAGENT_SOURCE_FILES = (
    (
        "data/train-00000-of-00012.parquet",
        "5a395e8c7bb8ddc4b8f4d268506b3a0e2cf9b5ec3922600117322fe788067a13",
    ),
    (
        "data/train-00001-of-00012.parquet",
        "fca106cce0f09891c2fadc032fb304da9ae5a7c31d2a39eb7ec70a7bdd4a9882",
    ),
    (
        "data/train-00002-of-00012.parquet",
        "fc28c2ab014c6c90d72026dda9cf8753e37b4b7b128c05c4232980cdcc99f3f7",
    ),
    (
        "data/train-00003-of-00012.parquet",
        "b6a4e13118de1792b383c077bf6023881a9ff5d4c171a75e767ffb5ad7c037c5",
    ),
    (
        "data/train-00004-of-00012.parquet",
        "d1a8d8d3bafbfd589d32a42d4b0a321f107d9aab170e3dc2f749b0f252a958c5",
    ),
    (
        "data/train-00005-of-00012.parquet",
        "c507212cd721512240a1bb87a8554bb7b429a9c03649c1473eb3d9e4673e6aca",
    ),
    (
        "data/train-00006-of-00012.parquet",
        "fcad81ec5704cb2dc8502a70a1a6cd86a7032227d8de7a613d904636ee53337c",
    ),
    (
        "data/train-00007-of-00012.parquet",
        "65547ce464ae1bbff550eacfcfec251ee1cb9b439744389a66bb97a8d5aa1cdb",
    ),
    (
        "data/train-00008-of-00012.parquet",
        "bfbd9a58fc9494b49ebd73d5f5e6836a72ae557cbc7ee59d2b2e3f91e3d44027",
    ),
    (
        "data/train-00009-of-00012.parquet",
        "8b4c45d8811f0fbbbc5fc46157312ddccb010c439d9e18289ae3ca3e5387ff09",
    ),
    (
        "data/train-00010-of-00012.parquet",
        "419e02daa099343e73d45d298b64e574ee012398ebbe19e5798526e5f1336b12",
    ),
    (
        "data/train-00011-of-00012.parquet",
        "7c3ccb843bab5457e29a82d9a36a3d59b8e1778c1d97324643c5f85a0d3f9492",
    ),
)

OPENHANDS_DATASET = "nebius/SWE-rebench-openhands-trajectories"
OPENHANDS_REVISION = "35455389ab51bf5e2306bfd436ef72d0f98bf882"
OPENHANDS_TOTAL = 67_074
OPENHANDS_SUCCESSFUL = 32_161
OPENHANDS_UNIQUE_SUCCESSFUL = 3_792
OPENHANDS_CLEAN_SUBMITTED = 31_020
OPENHANDS_CLEAN_SUBMITTED_UNIQUE = 3_735
OPENHANDS_SOURCE_FILE = (
    "trajectories.parquet",
    "14048dd1fcd22ce094b6e85f8a38f223a9ef1327031aaaad052804870212efa1",
)
OPENHANDS_TOOLS_SHA256 = (
    "d13c473da05d5cd18f6d2fb23146015585f4159fb64cb819fdc9dd8948caae9d"
)

SOURCE_SELECTION_WORKFLOWS_PER_SOURCE = 500
MAX_WORKFLOW_STEPS = 5
EXPECTED_RETAINED_WORKFLOW_COUNTS = {"sweagent": 33, "openhands": 0}
EXPECTED_RETAINED_CALL_COUNTS = {"sweagent": 139, "openhands": 0}
MAX_QUERY_CHARS = 128_000
MAX_ACTION_CHARS = 256 * 1024
MAX_ARGUMENT_BYTES = 512 * 1024
MAX_OBSERVATION_EXCERPT_CHARS = 1_024

SWEAGENT_FIXTURE_ID = "nebius-sweagent-benchmark-replay-v1"
SWEAGENT_FIXTURE_VERSION = "coding_nebius_sweagent_benchmark_replay_v1"

SWEAGENT_BENCHMARK_FILENAME = (
    "coding_nebius_sweagent_replay_multistep.json"
)
OPENHANDS_BENCHMARK_FILENAME = (
    "coding_nebius_swerebench_openhands_replay_multistep.json"
)
SWEAGENT_FIXTURE_FILENAME = "coding_nebius_sweagent_benchmark_replay.json"

_FENCED_ACTION_PATTERN = re.compile(
    r"```[^\n`]*\n(.*?)```",
    flags=re.DOTALL,
)
_OPENHANDS_ISSUE_PATTERN = re.compile(
    r"<issue_description>\s*(.*?)\s*</issue_description>",
    flags=re.DOTALL,
)
_PEM_PATTERN = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
    r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
    flags=re.DOTALL,
)
_TOKEN_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_value(value: Any) -> str:
    if isinstance(value, str):
        return _sha256_text(value)
    return _sha256_text(_canonical_json(value))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned_url(dataset: str, revision: str, source_path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{dataset}/resolve/"
        f"{revision}/{source_path}"
    )


def _validated_source(
    supplied: str,
    *,
    dataset: str,
    revision: str,
    source_path: str,
    expected_sha256: str,
) -> str:
    expected_url = _pinned_url(dataset, revision, source_path)
    if supplied.startswith(("https://", "http://")):
        if supplied != expected_url:
            raise ValueError(
                f"Remote source must be the exact pinned URL {expected_url!r}."
            )
        return supplied

    path = Path(supplied).resolve()
    if not path.is_file():
        raise ValueError(f"Source parquet does not exist: {path}.")
    actual = _file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{path.name} has SHA-256 {actual}; expected {expected_sha256}."
        )
    return str(path)


def _validated_tools_file(supplied: str | None) -> None:
    if supplied is None:
        return
    path = Path(supplied).resolve()
    actual = _file_sha256(path)
    if actual != OPENHANDS_TOOLS_SHA256:
        raise ValueError(
            f"{path.name} has SHA-256 {actual}; expected "
            f"{OPENHANDS_TOOLS_SHA256}."
        )


def _duckdb_connection() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - importer environment only
        raise RuntimeError(
            "This importer requires DuckDB in the import environment. "
            "LayerMCP runtime does not require DuckDB."
        ) from exc
    return duckdb.connect()


def _source_file_label(filename: str, expected_paths: Iterable[str]) -> str:
    matching = [
        source_path
        for source_path in expected_paths
        if filename.endswith(source_path)
        or filename.endswith(Path(source_path).name)
    ]
    if len(matching) != 1:
        raise ValueError(f"Unable to identify pinned source file {filename!r}.")
    return matching[0]


def _load_sweagent_candidate_rows(
    connection: Any,
    sources: list[str],
) -> list[dict[str, Any]]:
    counts = connection.execute(
        """
        SELECT count(*) AS total,
               count(*) FILTER (WHERE target) AS successful,
               count(DISTINCT instance_id) FILTER (WHERE target)
                   AS unique_successful
          FROM read_parquet(?)
        """,
        [sources],
    ).fetchone()
    expected_counts = (
        SWEAGENT_TOTAL,
        SWEAGENT_SUCCESSFUL,
        SWEAGENT_UNIQUE_SUCCESSFUL,
    )
    if tuple(counts) != expected_counts:
        raise ValueError(
            f"Unexpected SWE-agent source counts {tuple(counts)!r}; "
            f"expected {expected_counts!r}."
        )

    coordinates = connection.execute(
        """
        SELECT filename, file_row_number, instance_id, model_name, exit_status
          FROM read_parquet(
              ?,
              filename = true,
              file_row_number = true
          )
         WHERE target
         ORDER BY filename, file_row_number
         LIMIT 10000
        """,
        [sources],
    ).fetchall()
    if len(coordinates) != 10_000:
        raise ValueError("SWE-agent candidate coordinate count is inconsistent.")

    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE selected_sweagent_rows (
            filename VARCHAR,
            file_row_number BIGINT
        )
        """
    )
    connection.executemany(
        "INSERT INTO selected_sweagent_rows VALUES (?, ?)",
        [(row[0], row[1]) for row in coordinates],
    )
    cursor = connection.execute(
        """
        SELECT source.filename, source.file_row_number, source.instance_id,
               source.model_name, source.exit_status, source.trajectory
          FROM read_parquet(
                   ?,
                   filename = true,
                   file_row_number = true
               ) AS source
          JOIN selected_sweagent_rows AS selected
            ON source.filename = selected.filename
           AND source.file_row_number = selected.file_row_number
         ORDER BY source.filename, source.file_row_number
        """,
        [sources],
    )
    columns = [item[0] for item in cursor.description]
    return [
        dict(zip(columns, values, strict=True)) for values in cursor.fetchall()
    ]


def _load_openhands_candidate_rows(
    connection: Any,
    source: str,
) -> list[dict[str, Any]]:
    counts = connection.execute(
        """
        SELECT count(*) AS total,
               count(*) FILTER (WHERE resolved = 1) AS successful,
               count(DISTINCT instance_id) FILTER (WHERE resolved = 1)
                   AS unique_successful,
               count(*) FILTER (
                   WHERE resolved = 1 AND exit_status = 'submit'
               ) AS clean_submitted,
               count(DISTINCT instance_id) FILTER (
                   WHERE resolved = 1 AND exit_status = 'submit'
               ) AS clean_submitted_unique
          FROM read_parquet(?)
        """,
        [source],
    ).fetchone()
    expected_counts = (
        OPENHANDS_TOTAL,
        OPENHANDS_SUCCESSFUL,
        OPENHANDS_UNIQUE_SUCCESSFUL,
        OPENHANDS_CLEAN_SUBMITTED,
        OPENHANDS_CLEAN_SUBMITTED_UNIQUE,
    )
    if tuple(counts) != expected_counts:
        raise ValueError(
            f"Unexpected OpenHands source counts {tuple(counts)!r}; "
            f"expected {expected_counts!r}."
        )

    coordinates = connection.execute(
        """
        SELECT filename, file_row_number, trajectory_id, instance_id, repo,
               exit_status
          FROM (
                SELECT filename, file_row_number, trajectory_id, instance_id,
                       repo, exit_status,
                       row_number() OVER (
                           PARTITION BY instance_id
                           ORDER BY file_row_number, trajectory_id
                       ) AS source_rank
                  FROM read_parquet(
                      ?,
                      filename = true,
                      file_row_number = true
                  )
                 WHERE resolved = 1 AND exit_status = 'submit'
          )
         WHERE source_rank = 1
         ORDER BY file_row_number, trajectory_id
         LIMIT 1200
        """,
        [source],
    ).fetchall()
    if len(coordinates) != 1_200:
        raise ValueError("OpenHands candidate coordinate count is inconsistent.")

    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE selected_openhands_rows (
            filename VARCHAR,
            file_row_number BIGINT
        )
        """
    )
    connection.executemany(
        "INSERT INTO selected_openhands_rows VALUES (?, ?)",
        [(row[0], row[1]) for row in coordinates],
    )
    cursor = connection.execute(
        """
        SELECT source.filename, source.file_row_number, source.trajectory_id,
               source.instance_id, source.repo, source.exit_status,
               source.trajectory
          FROM read_parquet(
                   ?,
                   filename = true,
                   file_row_number = true
               ) AS source
          JOIN selected_openhands_rows AS selected
            ON source.filename = selected.filename
           AND source.file_row_number = selected.file_row_number
         ORDER BY source.file_row_number, source.trajectory_id
        """,
        [source],
    )
    columns = [item[0] for item in cursor.description]
    return [
        dict(zip(columns, values, strict=True)) for values in cursor.fetchall()
    ]


def _required_message_text(message: dict[str, Any], field: str) -> str:
    value = message.get("text")
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text.")
    return value


def _extract_sweagent_issue(trajectory: list[dict[str, Any]]) -> tuple[str, str]:
    initial_user = next(
        (
            message
            for message in trajectory
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        None,
    )
    if not isinstance(initial_user, dict):
        raise ValueError("SWE-agent trajectory has no initial user prompt.")
    full_prompt = _required_message_text(initial_user, "initial user prompt")

    issue_marker = "ISSUE:\n"
    instructions_marker = "\n\nINSTRUCTIONS:"
    if issue_marker not in full_prompt:
        raise ValueError("SWE-agent prompt has no ISSUE marker.")
    issue = full_prompt.split(issue_marker, 1)[1]
    if instructions_marker in issue:
        issue = issue.split(instructions_marker, 1)[0]
    issue = issue.strip()
    if not issue or len(issue) > MAX_QUERY_CHARS or "\x00" in issue:
        raise ValueError("SWE-agent issue text is empty or exceeds bounds.")
    return issue, full_prompt


def _parse_sweagent_action(text: str) -> str:
    matches = _FENCED_ACTION_PATTERN.findall(text)
    if not matches:
        raise ValueError("SWE-agent AI message has no fenced action.")
    command = matches[-1].strip()
    if not command or len(command) > MAX_ACTION_CHARS or "\x00" in command:
        raise ValueError("SWE-agent action is empty or exceeds bounds.")
    return command


def _sweagent_tool(command: str) -> tuple[str, str]:
    source_tool = command.lstrip().split(maxsplit=1)[0]
    if source_tool in {"open", "goto", "scroll_down", "scroll_up"}:
        return "code_replay_sweagent_file_view", source_tool
    if source_tool in {"search_dir", "search_file", "find_file"}:
        return "code_replay_sweagent_file_search", source_tool
    if source_tool in {"create", "edit"}:
        return "code_replay_sweagent_file_edit", source_tool
    if source_tool == "submit":
        return "code_replay_sweagent_submit", source_tool
    return "code_replay_sweagent_shell", source_tool


def _redact_sensitive_text(text: str) -> tuple[str, bool]:
    redacted = _PEM_PATTERN.sub("[REDACTED PRIVATE KEY]", text)
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub("[REDACTED TOKEN]", redacted)
    if "\x00" in redacted:
        redacted = redacted.replace("\x00", "\N{REPLACEMENT CHARACTER}")
    return redacted, redacted != text


def _contains_high_confidence_secret(value: Any) -> bool:
    text = value if isinstance(value, str) else _canonical_json(value)
    if _PEM_PATTERN.search(text):
        return True
    return any(pattern.search(text) for pattern in _TOKEN_PATTERNS)


def _compact_observation(value: str | None) -> tuple[dict[str, Any], bool]:
    if value is None:
        return (
            {
                "available": False,
                "excerpt": "",
                "full_sha256": _sha256_text(""),
                "original_chars": 0,
                "truncated": False,
            },
            False,
        )

    redacted, transformed = _redact_sensitive_text(value)
    if len(redacted) <= MAX_OBSERVATION_EXCERPT_CHARS and not transformed:
        excerpt = redacted
        truncated = False
    else:
        head_chars = 704
        tail_chars = 288
        marker = "\n...[bounded replay excerpt]...\n"
        excerpt = redacted[:head_chars] + marker + redacted[-tail_chars:]
        if len(excerpt) >= len(value):
            excerpt = excerpt[: max(0, len(value) - 1)]
        truncated = True

    return (
        {
            "available": True,
            "excerpt": excerpt,
            "full_sha256": _sha256_text(value),
            "original_chars": len(value),
            "truncated": truncated,
        },
        transformed,
    )


def _sweagent_calls(
    trajectory: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    calls: list[dict[str, Any]] = []
    for message_index, message in enumerate(trajectory):
        if not isinstance(message, dict) or message.get("role") != "ai":
            continue
        ai_text = _required_message_text(
            message,
            f"SWE-agent message {message_index}",
        )
        command = _parse_sweagent_action(ai_text)
        tool, source_tool = _sweagent_tool(command)

        observation_text: str | None = None
        if message_index + 1 < len(trajectory):
            next_message = trajectory[message_index + 1]
            if (
                isinstance(next_message, dict)
                and next_message.get("role") == "user"
            ):
                observation_text = _required_message_text(
                    next_message,
                    f"SWE-agent observation {message_index + 1}",
                )
        if observation_text is None and source_tool != "submit":
            raise ValueError(
                "A non-terminal SWE-agent action has no following observation."
            )
        if _contains_high_confidence_secret(command):
            raise ValueError("SWE-agent action contains a possible secret.")
        observation, redacted = _compact_observation(observation_text)
        calls.append(
            {
                "source_message_index": message_index,
                "source_tool_call_index": None,
                "source_tool": source_tool,
                "normalized_tool": tool,
                "source_args": {"command": command},
                "observation": observation,
                "observation_redacted": redacted,
            }
        )

    if len(calls) < 2:
        raise ValueError("SWE-agent workflow requires at least two actions.")
    return calls, 0


def _as_trajectory(value: Any, field: str) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(f"{field} must be a list of message objects.")
    return value


def _extract_openhands_issue(
    trajectory: list[dict[str, Any]],
) -> tuple[str, str]:
    initial_user = next(
        (
            message
            for message in trajectory
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ),
        None,
    )
    if initial_user is None:
        raise ValueError("OpenHands trajectory has no initial user prompt.")
    full_prompt = str(initial_user["content"])
    match = _OPENHANDS_ISSUE_PATTERN.search(full_prompt)
    if match is None:
        raise ValueError("OpenHands prompt has no issue_description element.")
    issue = match.group(1).strip()
    if not issue or len(issue) > MAX_QUERY_CHARS or "\x00" in issue:
        raise ValueError("OpenHands issue text is empty or exceeds bounds.")
    return issue, full_prompt


def _openhand_result_messages(
    trajectory: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_call_id: dict[str, dict[str, Any]] = {}
    for message_index, message in enumerate(trajectory):
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id or call_id in by_call_id:
            raise ValueError("OpenHands tool result IDs must be unique strings.")
        result = deepcopy(message)
        result["_source_message_index"] = message_index
        by_call_id[call_id] = result
    return by_call_id


def _normalize_openhands_args(
    source_tool: str,
    raw_args: Any,
) -> tuple[str, dict[str, Any]]:
    if isinstance(raw_args, str):
        raw_args = json.loads(raw_args)
    if not isinstance(raw_args, dict):
        raise ValueError("OpenHands function arguments must be an object.")
    if len(_canonical_json(raw_args).encode("utf-8")) > MAX_ARGUMENT_BYTES:
        raise ValueError("OpenHands arguments exceed the replay bound.")
    if _contains_high_confidence_secret(raw_args):
        raise ValueError("OpenHands arguments contain a possible secret.")

    specifications = {
        "execute_bash": (
            "code_replay_openhands_execute_bash",
            {"command"},
            {"command", "is_input", "timeout"},
        ),
        "finish": (
            "code_replay_openhands_finish",
            {"message"},
            {"message"},
        ),
        "task_tracker": (
            "code_replay_openhands_task_tracker",
            {"command"},
            {"command", "task_list"},
        ),
        "str_replace_editor": (
            "code_replay_openhands_str_replace_editor",
            {"command", "path"},
            {
                "command",
                "path",
                "file_text",
                "insert_line",
                "new_str",
                "old_str",
                "view_range",
            },
        ),
    }
    if source_tool not in specifications:
        raise ValueError(f"Unsupported OpenHands source tool {source_tool!r}.")
    normalized_tool, required, allowed = specifications[source_tool]
    if not required.issubset(raw_args) or not set(raw_args).issubset(allowed):
        raise ValueError(f"Invalid {source_tool} argument fields.")

    normalized = deepcopy(raw_args)
    for key, value in list(normalized.items()):
        if value is None:
            normalized.pop(key)
    return normalized_tool, normalized


def _openhands_calls(
    trajectory: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    results_by_id = _openhand_result_messages(trajectory)
    calls: list[dict[str, Any]] = []
    omitted_think_calls = 0

    for message_index, message in enumerate(trajectory):
        if message.get("role") != "assistant":
            continue
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            continue
        if not isinstance(raw_calls, list):
            raise ValueError("OpenHands assistant tool_calls must be a list.")

        operational = []
        for tool_call_index, tool_call in enumerate(raw_calls):
            if not isinstance(tool_call, dict):
                raise ValueError("OpenHands tool call must be an object.")
            function = tool_call.get("function")
            if not isinstance(function, dict):
                raise ValueError("OpenHands tool call requires a function.")
            source_tool = function.get("name")
            if source_tool == "think":
                omitted_think_calls += 1
                continue
            operational.append((tool_call_index, tool_call, function))

        if len(operational) > 1:
            raise ValueError(
                "Parallel OpenHands operational calls are excluded from the "
                "linear replay benchmark."
            )
        for tool_call_index, tool_call, function in operational:
            call_id = tool_call.get("id")
            source_tool = function.get("name")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("OpenHands call ID must be a non-empty string.")
            if not isinstance(source_tool, str):
                raise ValueError("OpenHands source tool name must be text.")
            normalized_tool, source_args = _normalize_openhands_args(
                source_tool,
                function.get("arguments"),
            )

            result_message = results_by_id.get(call_id)
            if result_message is None:
                if source_tool != "finish":
                    raise ValueError(
                        "A non-terminal OpenHands call has no matched result."
                    )
                observation_text = None
                result_message_index = None
            else:
                if result_message.get("name") != source_tool:
                    raise ValueError("OpenHands call/result tool names differ.")
                content = result_message.get("content")
                if not isinstance(content, str):
                    raise ValueError("OpenHands tool observation must be text.")
                observation_text = content
                result_message_index = result_message["_source_message_index"]

            observation, redacted = _compact_observation(observation_text)
            calls.append(
                {
                    "source_message_index": message_index,
                    "source_tool_call_index": tool_call_index,
                    "source_result_message_index": result_message_index,
                    "source_tool_call_id": call_id,
                    "source_tool": source_tool,
                    "normalized_tool": normalized_tool,
                    "source_args": source_args,
                    "observation": observation,
                    "observation_redacted": redacted,
                }
            )

    if len(calls) < 2:
        raise ValueError("OpenHands workflow requires at least two retained calls.")
    return calls, omitted_think_calls


def _replay_step_query(
    *,
    source_label: str,
    record_id: str,
    trajectory_id: str,
    step_index: int,
    source_tool: str,
) -> str:
    return (
        f"Replay the recorded {source_label} call as inert offline data. "
        f"record_id={record_id}; trajectory_id={trajectory_id}; "
        f"step_index={step_index}; source_tool={source_tool}"
    )


def _build_steps_and_records(
    *,
    dataset_prefix: str,
    source_label: str,
    trajectory_id: str,
    instance_id: str,
    calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for step_index, call in enumerate(calls):
        record_id = (
            f"coding-{dataset_prefix}-{trajectory_id}-call-{step_index:03d}"
        )
        source_args = deepcopy(call["source_args"])
        expected_args = {
            "record_id": record_id,
            "trajectory_id": trajectory_id,
            "step_index": step_index,
        }
        source_output_sha = call["observation"]["full_sha256"]
        source_input_sha = _sha256_value(source_args)
        step_id = f"call_{step_index + 1:03d}"
        source_metadata = {
            "source_instance_id": instance_id,
            "source_message_index": call["source_message_index"],
            "source_tool": call["source_tool"],
            "source_input_canonical_sha256": source_input_sha,
            "source_output_canonical_sha256": source_output_sha,
            "source_observation_redacted": call["observation_redacted"],
        }
        if call.get("source_tool_call_index") is not None:
            source_metadata["source_tool_call_index"] = call[
                "source_tool_call_index"
            ]
            source_metadata["source_tool_call_id"] = call[
                "source_tool_call_id"
            ]
            source_metadata["source_result_message_index"] = call[
                "source_result_message_index"
            ]

        steps.append(
            {
                "id": step_id,
                "query": _replay_step_query(
                    source_label=source_label,
                    record_id=record_id,
                    trajectory_id=trajectory_id,
                    step_index=step_index,
                    source_tool=call["source_tool"],
                ),
                "expected_tool": call["normalized_tool"],
                "expected_args": expected_args,
                "expected_answer": {
                    "record_id": record_id,
                    "tool": call["normalized_tool"],
                    "offline_replay": True,
                    "network_access": False,
                    "process_executed": False,
                    "mutation_applied": False,
                },
                "depends_on": (
                    [] if step_index == 0 else [f"call_{step_index:03d}"]
                ),
                **source_metadata,
            }
        )
        records.append(
            {
                "record_id": record_id,
                "trajectory_id": trajectory_id,
                "step_index": step_index,
                "tool": call["normalized_tool"],
                "args": source_args,
                "observation": call["observation"],
                "source": source_metadata,
            }
        )
    return steps, records


def _sweagent_trajectory_id(source_path: str, row_number: int) -> str:
    match = re.search(r"train-(\d{5})-of-00012", source_path)
    if match is None:
        raise ValueError(f"Unexpected SWE-agent source path {source_path!r}.")
    return f"sweagent-{int(match.group(1)):02d}-{row_number:06d}"


def _retained_benchmark_workflows(
    workflows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the source workflows within the public five-call ceiling."""
    return [
        workflow
        for workflow in workflows
        if len(workflow["expected_steps"]) <= MAX_WORKFLOW_STEPS
    ]


def _validate_retained_counts(
    source: str,
    workflows: list[dict[str, Any]],
) -> None:
    workflow_count = len(workflows)
    call_count = sum(len(workflow["expected_steps"]) for workflow in workflows)
    expected = (
        EXPECTED_RETAINED_WORKFLOW_COUNTS[source],
        EXPECTED_RETAINED_CALL_COUNTS[source],
    )
    if (workflow_count, call_count) != expected:
        raise ValueError(
            f"Unexpected retained {source} benchmark counts "
            f"{(workflow_count, call_count)!r}; expected {expected!r}."
        )


def _build_sweagent(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    workflows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    selected_instances: set[str] = set()
    excluded = Counter()
    model_distribution = Counter()
    tool_distribution = Counter()
    expected_paths = [item[0] for item in SWEAGENT_SOURCE_FILES]

    for row in rows:
        if len(workflows) >= SOURCE_SELECTION_WORKFLOWS_PER_SOURCE:
            break
        instance_id = str(row["instance_id"])
        if instance_id in selected_instances:
            continue
        try:
            trajectory = _as_trajectory(
                row["trajectory"],
                "SWE-agent trajectory",
            )
            query, full_prompt = _extract_sweagent_issue(trajectory)
            calls, _ = _sweagent_calls(trajectory)
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            excluded[type(exc).__name__ + ":" + str(exc)] += 1
            continue

        source_path = _source_file_label(row["filename"], expected_paths)
        source_row_number = int(row["file_row_number"])
        trajectory_id = _sweagent_trajectory_id(
            source_path,
            source_row_number,
        )
        trajectory_sha = _sha256_value(trajectory)
        steps, trajectory_records = _build_steps_and_records(
            dataset_prefix="sweagent",
            source_label="SWE-agent action",
            trajectory_id=trajectory_id,
            instance_id=instance_id,
            calls=calls,
        )
        workflow_number = len(workflows) + 1
        workflows.append(
            {
                "id": (
                    "coding_multistep_nebius_sweagent_"
                    f"{workflow_number:03d}"
                ),
                "domain": "coding",
                "task_type": "multi_step_tool_routing",
                "benchmark_mode": "offline_trace_replay",
                "difficulty": "hard",
                "source": "public_coding_successful_model_trajectory",
                "query": query,
                "expected_steps": steps,
                "source_dataset": SWEAGENT_DATASET,
                "source_revision": SWEAGENT_REVISION,
                "source_trajectory_id": trajectory_id,
                "source_instance_id": instance_id,
                "source_model_name": row["model_name"],
                "source_exit_status": row["exit_status"],
                "source_success": True,
                "source_file": source_path,
                "source_file_row_number": source_row_number,
                "source_trajectory_sha256": trajectory_sha,
                "source_full_prompt_sha256": _sha256_text(full_prompt),
                "source_issue_sha256": _sha256_text(query),
                "source_original_action_count": len(calls),
                "source_selected_call_count": len(steps),
                "query_origin": "extracted_public_issue_prompt",
                "tool_sequence_origin": (
                    "successful_released_sweagent_trajectory_actions"
                ),
                "source_license": "CC-BY-4.0",
                "fixture_id": SWEAGENT_FIXTURE_ID,
                "fixture_version": SWEAGENT_FIXTURE_VERSION,
                "provenance_type": (
                    "public_issue_and_successful_upstream_trajectory_adaptation"
                ),
                "perturbation_type": "released_successful_trajectory",
                "notes": (
                    "Issue text is mechanically extracted from the released "
                    "prompt. Model discussion is omitted; fenced actions and "
                    "ordered observations are retained as inert offline replay."
                ),
            }
        )
        records.extend(trajectory_records)
        selected_instances.add(instance_id)
        model_distribution[str(row["model_name"])] += 1
        tool_distribution.update(step["expected_tool"] for step in steps)

    if len(workflows) != SOURCE_SELECTION_WORKFLOWS_PER_SOURCE:
        raise ValueError(
            f"Only {len(workflows)} eligible SWE-agent workflows were found."
        )
    if len(selected_instances) != SOURCE_SELECTION_WORKFLOWS_PER_SOURCE:
        raise ValueError("SWE-agent selected issue IDs are not unique.")

    fixture = {
        "fixture_id": SWEAGENT_FIXTURE_ID,
        "fixture_version": SWEAGENT_FIXTURE_VERSION,
        "description": (
            "Bounded inert replay for the retained Nebius SWE-agent benchmark "
            "workflows. No recorded action is executed."
        ),
        "manifest": {
            "source_dataset": SWEAGENT_DATASET,
            "source_repository": (
                f"https://huggingface.co/datasets/{SWEAGENT_DATASET}"
            ),
            "source_revision": SWEAGENT_REVISION,
            "source_config": "default",
            "source_split": "train",
            "source_license": "CC-BY-4.0",
            "source_total_trajectory_count": SWEAGENT_TOTAL,
            "source_successful_trajectory_count": SWEAGENT_SUCCESSFUL,
            "source_unique_successful_instance_count": (
                SWEAGENT_UNIQUE_SUCCESSFUL
            ),
            "source_success_field": "target",
            "source_success_value": True,
            "source_files": [
                {"path": path, "sha256": sha256}
                for path, sha256 in SWEAGENT_SOURCE_FILES
            ],
            "selection_rule": (
                "Pinned physical source order over target=true rows; retain "
                "the first eligible trajectory per instance_id and stop at "
                "500 issues. Eligibility requires the complete AI action "
                "sequence to parse, every non-submit action to have a "
                "following user observation, bounded actions without "
                "high-confidence secrets, and at least two calls."
            ),
            "workflow_count": len(workflows),
            "selected_call_count": len(records),
            "replay_record_count": len(records),
            "selected_model_distribution": dict(sorted(model_distribution.items())),
            "selected_tool_distribution": dict(sorted(tool_distribution.items())),
            "excluded_candidate_reasons": dict(sorted(excluded.items())),
            "omitted_reasoning_tools": ["discussion"],
            "synthetic": False,
            "network_access": False,
            "process_execution": False,
            "mutation_applied": False,
            "teacher_forced_routing": True,
            "underlying_repository_license_notice": (
                "The dataset card requires respecting each underlying "
                "repository license and applicable model-output licenses."
            ),
        },
        "records": records,
    }
    retained_workflows = _retained_benchmark_workflows(workflows)
    _validate_retained_counts("sweagent", retained_workflows)
    retained_record_ids = [
        step["expected_args"]["record_id"]
        for workflow in retained_workflows
        for step in workflow["expected_steps"]
    ]
    if (
        len(retained_record_ids) != EXPECTED_RETAINED_CALL_COUNTS["sweagent"]
        or len(set(retained_record_ids)) != len(retained_record_ids)
    ):
        raise ValueError("The SWE-agent benchmark record references drifted.")
    retained_record_id_set = set(retained_record_ids)
    retained_records = [
        record
        for record in records
        if record["record_id"] in retained_record_id_set
    ]
    if [record["record_id"] for record in retained_records] != retained_record_ids:
        raise ValueError("The SWE-agent benchmark fixture records drifted.")
    retained_model_distribution = Counter(
        workflow["source_model_name"] for workflow in retained_workflows
    )
    retained_tool_distribution = Counter(
        step["expected_tool"]
        for workflow in retained_workflows
        for step in workflow["expected_steps"]
    )
    fixture["description"] = (
        "Bounded inert replay for the 33 retained Nebius SWE-agent benchmark "
        "workflows and their 139 calls. No recorded action is executed."
    )
    fixture["manifest"].update(
        {
            "selection_rule": (
                "Select the first 500 eligible successful workflows in pinned "
                "source order, then retain only the 33 workflows containing at "
                "most five calls. This benchmark-scoped fixture contains exactly "
                "the records referenced by those retained workflow steps."
            ),
            "source_candidate_workflow_count": (
                SOURCE_SELECTION_WORKFLOWS_PER_SOURCE
            ),
            "source_candidate_call_count": len(records),
            "benchmark_max_workflow_steps": MAX_WORKFLOW_STEPS,
            "workflow_count": len(retained_workflows),
            "selected_call_count": len(retained_records),
            "replay_record_count": len(retained_records),
            "selected_model_distribution": dict(
                sorted(retained_model_distribution.items())
            ),
            "selected_tool_distribution": dict(
                sorted(retained_tool_distribution.items())
            ),
            "record_scope": "exact_benchmark_expected_step_record_ids",
        }
    )
    fixture["records"] = retained_records
    return retained_workflows, fixture, selected_instances


def _build_openhands(
    rows: list[dict[str, Any]],
    *,
    excluded_instance_ids: set[str],
) -> list[dict[str, Any]]:
    workflows: list[dict[str, Any]] = []
    selected_instances: set[str] = set()

    for row in rows:
        if len(workflows) >= SOURCE_SELECTION_WORKFLOWS_PER_SOURCE:
            break
        instance_id = str(row["instance_id"])
        if instance_id in excluded_instance_ids:
            continue
        try:
            trajectory = _as_trajectory(
                row["trajectory"],
                "OpenHands trajectory",
            )
            query, full_prompt = _extract_openhands_issue(trajectory)
            calls, omitted_think = _openhands_calls(trajectory)
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        trajectory_id = str(row["trajectory_id"])
        trajectory_sha = _sha256_value(trajectory)
        steps, _ = _build_steps_and_records(
            dataset_prefix="openhands",
            source_label="OpenHands function",
            trajectory_id=trajectory_id,
            instance_id=instance_id,
            calls=calls,
        )
        workflow_number = len(workflows) + 1
        workflows.append(
            {
                "id": (
                    "coding_multistep_nebius_openhands_"
                    f"{workflow_number:03d}"
                ),
                "domain": "coding",
                "task_type": "multi_step_tool_routing",
                "difficulty": "hard",
                "source": "public_coding_resolved_function_call_trajectory",
                "query": query,
                "expected_steps": steps,
                "source_dataset": OPENHANDS_DATASET,
                "source_revision": OPENHANDS_REVISION,
                "source_trajectory_id": trajectory_id,
                "source_instance_id": instance_id,
                "source_repository_name": row["repo"],
                "source_exit_status": row["exit_status"],
                "source_success": True,
                "source_file": OPENHANDS_SOURCE_FILE[0],
                "source_file_row_number": int(row["file_row_number"]),
                "source_trajectory_sha256": trajectory_sha,
                "source_full_prompt_sha256": _sha256_text(full_prompt),
                "source_issue_sha256": _sha256_text(query),
                "source_original_operational_call_count": len(calls),
                "source_omitted_think_call_count": omitted_think,
                "source_selected_call_count": len(steps),
                "query_origin": "extracted_public_issue_prompt",
                "tool_sequence_origin": (
                    "resolved_released_openhands_function_call_trajectory"
                ),
                "source_license": "CC-BY-4.0",
                "provenance_type": (
                    "public_issue_and_resolved_upstream_trajectory_adaptation"
                ),
                "perturbation_type": "released_resolved_trajectory",
                "notes": (
                    "Issue text and operational function calls come from the "
                    "released resolved trajectory. Assistant prose and think "
                    "calls are omitted; calls are replayed inertly offline."
                ),
            }
        )
        selected_instances.add(instance_id)

    if len(workflows) != SOURCE_SELECTION_WORKFLOWS_PER_SOURCE:
        raise ValueError(
            f"Only {len(workflows)} eligible OpenHands workflows were found."
        )
    if len(selected_instances) != SOURCE_SELECTION_WORKFLOWS_PER_SOURCE:
        raise ValueError("OpenHands selected issue IDs are not unique.")
    if selected_instances & excluded_instance_ids:
        raise ValueError("The two selected sources repeat an issue ID.")

    retained_workflows = _retained_benchmark_workflows(workflows)
    _validate_retained_counts("openhands", retained_workflows)
    return retained_workflows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build(
    *,
    sweagent_sources: list[str],
    openhands_source: str,
    openhands_tools: str | None,
    output_root: Path,
) -> tuple[Path, Path, Path]:
    if len(sweagent_sources) != len(SWEAGENT_SOURCE_FILES):
        raise ValueError("Exactly 12 SWE-agent parquet sources are required.")
    validated_sweagent_sources = [
        _validated_source(
            supplied,
            dataset=SWEAGENT_DATASET,
            revision=SWEAGENT_REVISION,
            source_path=source_path,
            expected_sha256=sha256,
        )
        for supplied, (source_path, sha256) in zip(
            sweagent_sources,
            SWEAGENT_SOURCE_FILES,
            strict=True,
        )
    ]
    validated_openhands_source = _validated_source(
        openhands_source,
        dataset=OPENHANDS_DATASET,
        revision=OPENHANDS_REVISION,
        source_path=OPENHANDS_SOURCE_FILE[0],
        expected_sha256=OPENHANDS_SOURCE_FILE[1],
    )
    _validated_tools_file(openhands_tools)

    connection = _duckdb_connection()
    try:
        sweagent_rows = _load_sweagent_candidate_rows(
            connection,
            validated_sweagent_sources,
        )
        openhands_rows = _load_openhands_candidate_rows(
            connection,
            validated_openhands_source,
        )
    finally:
        connection.close()

    sweagent_workflows, sweagent_fixture, selected_sweagent_instances = (
        _build_sweagent(sweagent_rows)
    )
    openhands_workflows = _build_openhands(
        openhands_rows,
        excluded_instance_ids=selected_sweagent_instances,
    )

    sweagent_benchmark_path = output_root / SWEAGENT_BENCHMARK_FILENAME
    openhands_benchmark_path = output_root / OPENHANDS_BENCHMARK_FILENAME
    sweagent_fixture_path = (
        output_root / "fixtures" / SWEAGENT_FIXTURE_FILENAME
    )
    _write_json(sweagent_benchmark_path, sweagent_workflows)
    _write_json(openhands_benchmark_path, openhands_workflows)
    _write_json(sweagent_fixture_path, sweagent_fixture)
    return (
        sweagent_benchmark_path,
        openhands_benchmark_path,
        sweagent_fixture_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweagent-sources",
        nargs=12,
        required=True,
        metavar="PARQUET",
    )
    parser.add_argument("--openhands-source", required=True)
    parser.add_argument(
        "--openhands-tools",
        help="Optional local pinned tools.json to hash-check.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    paths = build(
        sweagent_sources=args.sweagent_sources,
        openhands_source=args.openhands_source,
        openhands_tools=args.openhands_tools,
        output_root=args.output_root,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
