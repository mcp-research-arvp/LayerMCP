"""Data models for tasks, suites, and per-task results."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A single tool-selection scenario.

    `tools` holds OpenAI-format tool specs (including distractors). The model
    must pick `expected_tool` with `expected_arguments`. If `expected_tool` is
    None, the correct behaviour is to answer directly and call no tool — this
    is how we measure over-triggering / hallucinated calls.
    """

    id: str
    domain: str
    query: str
    system: Optional[str] = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    expected_tool: Optional[str] = None
    expected_arguments: Optional[dict[str, Any]] = None
    argument_match: Literal["exact", "subset"] = "subset"
    notes: Optional[str] = None

    @property
    def available_tool_names(self) -> set[str]:
        names: set[str] = set()
        for tool in self.tools:
            name = tool.get("function", {}).get("name")
            if name:
                names.add(name)
        return names


class Suite(BaseModel):
    name: str
    tasks: list[Task]


class TaskResult(BaseModel):
    """Recorded outcome for one task. Correctness scoring is intentionally
    out of scope here — this just captures what the model did plus telemetry."""

    task_id: str
    domain: str
    expected_tool: Optional[str] = None

    # what the model did
    chosen_tool: Optional[str] = None
    chosen_arguments: Optional[dict[str, Any]] = None
    made_call: bool = False

    # economy / performance
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    had_reasoning: bool = False
    reasoning_chars: int = 0
    latency_s: float = 0.0
    tokens_per_second: Optional[float] = None

    # debugging
    raw_content: Optional[str] = None
    error: Optional[str] = None
