from __future__ import annotations

from typing import Any

from mcp_server.coding_replay_state import (
    CODING_REPLAY_TOOL_NAMES,
    replay_coding_call,
)


def _replay_coordinate(
    tool: str,
    record_id: str,
    trajectory_id: str,
    step_index: int,
) -> dict[str, Any]:
    if tool not in CODING_REPLAY_TOOL_NAMES:
        raise ValueError("Unknown coding replay tool.")
    return replay_coding_call(
        tool,
        record_id,
        trajectory_id,
        step_index,
    )


def code_replay_sweagent_shell(
    record_id: str,
    trajectory_id: str,
    step_index: int,
) -> dict[str, Any]:
    """Replay one recorded SWE-agent shell action by its allowlisted coordinates."""
    return _replay_coordinate(
        "code_replay_sweagent_shell",
        record_id,
        trajectory_id,
        step_index,
    )


def code_replay_sweagent_file_view(
    record_id: str,
    trajectory_id: str,
    step_index: int,
) -> dict[str, Any]:
    """Replay one recorded SWE-agent file-view action by its coordinates."""
    return _replay_coordinate(
        "code_replay_sweagent_file_view",
        record_id,
        trajectory_id,
        step_index,
    )


def code_replay_sweagent_file_search(
    record_id: str,
    trajectory_id: str,
    step_index: int,
) -> dict[str, Any]:
    """Replay one recorded SWE-agent file-search action by its coordinates."""
    return _replay_coordinate(
        "code_replay_sweagent_file_search",
        record_id,
        trajectory_id,
        step_index,
    )


def code_replay_sweagent_file_edit(
    record_id: str,
    trajectory_id: str,
    step_index: int,
) -> dict[str, Any]:
    """Replay one recorded SWE-agent file-edit action by its coordinates."""
    return _replay_coordinate(
        "code_replay_sweagent_file_edit",
        record_id,
        trajectory_id,
        step_index,
    )


def code_replay_sweagent_submit(
    record_id: str,
    trajectory_id: str,
    step_index: int,
) -> dict[str, Any]:
    """Replay one recorded SWE-agent submission by its allowlisted coordinates."""
    return _replay_coordinate(
        "code_replay_sweagent_submit",
        record_id,
        trajectory_id,
        step_index,
    )
