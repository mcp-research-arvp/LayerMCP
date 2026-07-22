from __future__ import annotations

from collections.abc import Mapping, Sequence


def format_tool_catalog(
    available_tools: Sequence[str],
    live_descriptions: Mapping[str, str] | None = None,
) -> str:
    """Format tools using MCP descriptions when an MCP session supplied them."""
    descriptions = live_descriptions or {}
    return "\n".join(
        f"- {tool}: {' '.join(descriptions[tool].split())}"
        if descriptions.get(tool)
        else f"- {tool}"
        for tool in available_tools
    )
