"""The evaluation loop: send each task to the model, record its response.

No correctness scoring here — we capture the model's tool choice and telemetry
(tokens, latency) and write them out. Scoring is added separately later.
"""

from __future__ import annotations

import json

from .client import LLMClient
from .schema import Suite, Task, TaskResult

DEFAULT_SYSTEM = (
    "You are a helpful assistant with access to tools. "
    "Call a tool only when it is needed to answer the user; "
    "otherwise answer directly."
)


def build_messages(task: Task, no_think: bool) -> list[dict[str, str]]:
    system = task.system or DEFAULT_SYSTEM
    if no_think:
        # Qwen3 soft switch to disable chain-of-thought.
        system = f"{system} /no_think"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": task.query},
    ]


def record_response(task: Task, response: dict) -> TaskResult:
    """Pull the first tool call and telemetry out of a raw API response."""
    result = TaskResult(task_id=task.id, domain=task.domain, expected_tool=task.expected_tool)

    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError) as exc:
        result.error = f"malformed response: {exc}"
        return result

    usage = response.get("usage", {})
    result.prompt_tokens = usage.get("prompt_tokens", 0)
    result.completion_tokens = usage.get("completion_tokens", 0)
    result.total_tokens = usage.get("total_tokens", 0)
    result.latency_s = response.get("_latency_s", 0.0)
    result.tokens_per_second = response.get("timings", {}).get("predicted_per_second")

    reasoning = message.get("reasoning_content")
    if reasoning:
        result.had_reasoning = True
        result.reasoning_chars = len(reasoning)
    result.raw_content = message.get("content") or None

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        fn = tool_calls[0].get("function", {})
        result.chosen_tool = fn.get("name")
        result.made_call = result.chosen_tool is not None
        raw_args = fn.get("arguments", "")
        if isinstance(raw_args, dict):
            result.chosen_arguments = raw_args
        elif raw_args:
            try:
                result.chosen_arguments = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                result.error = f"bad argument JSON: {exc}"
        else:
            result.chosen_arguments = {}

    return result


def run_suite(
    client: LLMClient,
    suite: Suite,
    max_tokens: int = 512,
    temperature: float = 0.0,
    no_think: bool = False,
    limit: int | None = None,
    verbose: bool = True,
) -> list[TaskResult]:
    tasks = suite.tasks[:limit] if limit else suite.tasks
    results: list[TaskResult] = []

    for i, task in enumerate(tasks, 1):
        try:
            response = client.chat(
                build_messages(task, no_think),
                tools=task.tools or None,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = record_response(task, response)
        except Exception as exc:  # noqa: BLE001 - record any failure as a task error
            result = TaskResult(
                task_id=task.id,
                domain=task.domain,
                expected_tool=task.expected_tool,
                error=str(exc),
            )
        results.append(result)

        if verbose:
            print(
                f"[{i:>2}/{len(tasks)}] {task.id:18s} "
                f"chose={str(result.chosen_tool):18s} expected={str(task.expected_tool):18s} "
                f"{result.latency_s:5.1f}s {result.completion_tokens:>4}tok"
                + (f"  ! {result.error}" if result.error else "")
            )

    return results
