"""Build the deterministic FinRetrieval multi-call benchmark adaptation.

This importer is intentionally offline. It consumes the three parquet files from
the pinned FinRetrieval release and writes a compact, executable replay fixture
plus one benchmark workflow for every question that has a correct, no-error,
multi-call trajectory of at most five calls. The replay fixture deliberately
retains the complete selected source-trace inventory for provenance.

DuckDB is an importer-only dependency; it is not required by LayerMCP at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.finance.grounding import (  # noqa: E402
    recorded_call_prompt_context,
)

SOURCE_REVISION = "86a111357cffa181b3ba0a6b5ce94625d4511176"
QUESTIONS_SHA256 = "4f5a4b20d5163390502fd84a21c87581578341c97edbf2726177c7412b88c4a9"
SCORES_SHA256 = "29eb5238e92153ce88bd5b68063d9e8aca4d4d74fa5107a9cc79e6da78fdc0b9"
TOOL_TRACES_SHA256 = "96d15a4d9bc9f9effaa0b95edb87f52445207a2417d6339b18e6e79df920595c"
EXCLUDED_NO_CORRECT_TRACE = [253, 455]
EXPECTED_SELECTED_WORKFLOWS = 498
MAX_BENCHMARK_WORKFLOW_STEPS = 5
EXCLUDED_OVER_STEP_LIMIT = [
    12,
    29,
    38,
    75,
    108,
    122,
    321,
    322,
    359,
    377,
    466,
    480,
    486,
]
EXPECTED_BENCHMARK_WORKFLOWS = 485
EXPECTED_BENCHMARK_CALLS = 1_490

BENCHMARK_FILENAME = "tool_routing_finance_finretrieval_multistep.json"
FIXTURE_FILENAME = "finretrieval_replay.json"
FIXTURE_ID = "finretrieval-replay-v1"

_DALOOPA_TOOL_MAP = {
    "discover_companies": "finance_discover_companies",
    "mcp__daloopa__discover_companies": "finance_discover_companies",
    "discover_company_series": "finance_discover_company_series",
    "mcp__daloopa__discover_company_series": "finance_discover_company_series",
    "get_company_fundamentals": "finance_get_company_fundamentals",
    "mcp__daloopa__get_company_fundamentals": ("finance_get_company_fundamentals"),
}
_WEB_TOOL_NAMES = {"WebSearch", "google_search", "google_search_agent"}
_SUPPORTED_SOURCE_TOOLS = frozenset(_DALOOPA_TOOL_MAP) | _WEB_TOOL_NAMES


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_value(value: Any) -> str:
    payload = value if isinstance(value, str) else _canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_source_hash(path: Path, expected: str) -> None:
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{path.name} has SHA-256 {actual}; expected pinned hash {expected}."
        )


def _load_parquet_rows(
    questions_path: Path,
    scores_path: Path,
    traces_path: Path,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - importer environment only
        raise RuntimeError(
            "This importer requires DuckDB. Install it in a temporary import "
            "environment; LayerMCP runtime does not require it."
        ) from exc

    connection = duckdb.connect()
    question_cursor = connection.execute(
        "SELECT * FROM read_parquet(?) ORDER BY index",
        [str(questions_path)],
    )
    question_columns = [item[0] for item in question_cursor.description]
    questions = {
        int(values[0]): dict(zip(question_columns, values, strict=True))
        for values in question_cursor.fetchall()
    }

    trace_cursor = connection.execute(
        """
        SELECT t.index, t.configuration, t.tool_calls, t.num_tool_calls,
               t.total_duration_ms, s.is_correct, s.expected_value,
               s.expected_unit, s.expected_currency
          FROM read_parquet(?) AS t
          JOIN read_parquet(?) AS s
            USING (index, configuration)
         ORDER BY t.index, t.configuration
        """,
        [str(traces_path), str(scores_path)],
    )
    columns = [item[0] for item in trace_cursor.description]
    traces = [
        dict(zip(columns, values, strict=True)) for values in trace_cursor.fetchall()
    ]
    return questions, traces


def _parse_calls(raw_calls: str, declared_count: int) -> list[dict[str, Any]]:
    calls = json.loads(raw_calls)
    if not isinstance(calls, list) or len(calls) != declared_count:
        raise ValueError("FinRetrieval trace call count does not match its payload.")
    if not all(isinstance(call, dict) for call in calls):
        raise ValueError("FinRetrieval tool_calls must contain JSON objects.")
    return calls


def _has_semantic_error(call: dict[str, Any]) -> bool:
    output = call.get("output")
    if output is None:
        return False
    serialized = output if isinstance(output, str) else _canonical_json(output)
    return "Output validation error:" in serialized


def _select_traces(traces: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    candidates_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        if not trace["is_correct"] or trace["num_tool_calls"] < 2:
            continue
        calls = _parse_calls(trace["tool_calls"], trace["num_tool_calls"])
        if any(call.get("is_error") for call in calls):
            continue
        if any(_has_semantic_error(call) for call in calls):
            continue
        if any(call.get("name") not in _SUPPORTED_SOURCE_TOOLS for call in calls):
            continue
        try:
            for call in calls:
                _normalized_call(call)
        except (KeyError, TypeError, ValueError):
            continue
        candidate = deepcopy(trace)
        candidate["calls"] = calls
        candidates_by_index[int(trace["index"])].append(candidate)

    selected: dict[int, dict[str, Any]] = {}
    for index, candidates in candidates_by_index.items():
        candidates.sort(
            key=lambda item: (
                any(call["name"] not in _DALOOPA_TOOL_MAP for call in item["calls"]),
                item["num_tool_calls"],
                sum(call["name"] not in _DALOOPA_TOOL_MAP for call in item["calls"]),
                item["configuration"],
            )
        )
        selected[index] = candidates[0]

    missing = sorted(set(range(500)) - set(selected))
    if missing != EXCLUDED_NO_CORRECT_TRACE:
        raise ValueError(
            "Unexpected FinRetrieval correct-trace coverage; missing indices "
            f"{missing!r}."
        )
    if len(selected) != EXPECTED_SELECTED_WORKFLOWS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_SELECTED_WORKFLOWS} selected workflows, got "
            f"{len(selected)}."
        )
    return selected


def _required_list(
    value: Any,
    field: str,
    *,
    allow_empty: bool = False,
) -> list[Any]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{field} must be {qualifier}.")
    return deepcopy(value)


def _normalized_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    source_name = call["name"]
    raw_args = call.get("input")
    if not isinstance(raw_args, dict):
        raise ValueError(f"{source_name} input must be an object.")

    normalized_tool = _DALOOPA_TOOL_MAP.get(source_name)
    if normalized_tool == "finance_discover_companies":
        return normalized_tool, {
            "keywords": _required_list(raw_args.get("keywords"), "keywords")
        }
    if normalized_tool == "finance_discover_company_series":
        return normalized_tool, {
            "company_id": int(raw_args["company_id"]),
            "keywords": _required_list(raw_args.get("keywords"), "keywords"),
            "periods": _required_list(
                raw_args.get("periods", []),
                "periods",
                allow_empty=True,
            ),
        }
    if normalized_tool == "finance_get_company_fundamentals":
        return normalized_tool, {
            "company_id": int(raw_args["company_id"]),
            "series_ids": [
                int(item)
                for item in _required_list(
                    raw_args.get("series_ids"),
                    "series_ids",
                )
            ],
            "periods": _required_list(
                raw_args.get("periods", []),
                "periods",
                allow_empty=True,
            ),
        }

    if source_name == "google_search":
        return "finance_search_web_archive", {
            "action": "search",
            "query": str(raw_args["query"]),
        }
    if source_name == "google_search_agent":
        return "finance_search_web_archive", {
            "action": "search",
            "query": str(raw_args["request"]),
        }
    if source_name == "WebSearch":
        source_action = raw_args.get("type")
        action_map = {
            "search": "search",
            "open_page": "open",
            "find_in_page": "find",
        }
        if source_action not in action_map:
            raise ValueError(f"Unsupported WebSearch action: {source_action!r}.")
        normalized_args: dict[str, Any] = {"action": action_map[source_action]}
        for field in ("query", "url", "pattern"):
            value = raw_args.get(field)
            if value is not None:
                normalized_args[field] = str(value)
        required_fields = {
            "search": ("query",),
            "open": ("url",),
            "find": ("url", "pattern"),
        }
        missing = [
            field
            for field in required_fields[normalized_args["action"]]
            if field not in normalized_args
        ]
        if missing:
            raise ValueError(
                f"WebSearch {source_action!r} lacks required fields: {missing!r}."
            )
        return "finance_search_web_archive", normalized_args

    raise ValueError(f"Unsupported FinRetrieval source tool: {source_name!r}.")


def _unwrap_output(value: Any) -> Any:
    current = deepcopy(value)
    for _ in range(5):
        if isinstance(current, str):
            try:
                current = json.loads(current)
            except json.JSONDecodeError:
                return current
            continue
        if isinstance(current, dict) and current.get("structuredContent") is not None:
            current = current["structuredContent"]
            continue
        if (
            isinstance(current, dict)
            and isinstance(current.get("content"), list)
            and current["content"]
            and isinstance(current["content"][0], dict)
            and isinstance(current["content"][0].get("text"), str)
        ):
            current = current["content"][0]["text"]
            continue
        if (
            isinstance(current, dict)
            and current.get("type") == "text"
            and isinstance(current.get("text"), str)
        ):
            current = current["text"]
            continue
        break
    return current


def _list_payload(value: Any) -> list[Any]:
    unwrapped = _unwrap_output(value)
    if isinstance(unwrapped, dict) and isinstance(unwrapped.get("result"), list):
        return unwrapped["result"]
    if isinstance(unwrapped, list):
        return unwrapped
    if isinstance(unwrapped, dict):
        flattened: list[Any] = []
        for item in unwrapped.values():
            if isinstance(item, list):
                flattened.extend(item)
        return flattened
    return []


def _future_ids(
    calls: list[dict[str, Any]],
    step_index: int,
    *,
    field: str,
) -> set[int]:
    values: set[int] = set()
    for call in calls[step_index + 1 :]:
        raw_args = call.get("input")
        if not isinstance(raw_args, dict):
            continue
        raw_value = raw_args.get(field)
        if isinstance(raw_value, list):
            values.update(
                int(item)
                for item in raw_value
                if isinstance(item, int) and not isinstance(item, bool)
            )
        elif isinstance(raw_value, int) and not isinstance(raw_value, bool):
            values.add(raw_value)
    return values


def _compact_result(
    call: dict[str, Any],
    calls: list[dict[str, Any]],
    step_index: int,
    normalized_tool: str,
) -> dict[str, Any]:
    output = call.get("output")
    if output is None:
        return {"recorded_output_available": False}

    if normalized_tool == "finance_discover_companies":
        company_ids = _future_ids(calls, step_index, field="company_id")
        items = [
            item
            for item in _list_payload(output)
            if isinstance(item, dict)
            and (not company_ids or item.get("company_id") in company_ids)
        ]
        return {
            "recorded_output_available": True,
            "companies": items[:25],
            "count": len(items[:25]),
        }

    if normalized_tool == "finance_discover_company_series":
        series_ids = _future_ids(calls, step_index, field="series_ids")
        all_items = [item for item in _list_payload(output) if isinstance(item, dict)]
        matching = [item for item in all_items if item.get("id") in series_ids]
        items = matching or all_items[:25]
        return {
            "recorded_output_available": True,
            "series": items[:50],
            "count": len(items[:50]),
        }

    if normalized_tool == "finance_get_company_fundamentals":
        items = [item for item in _list_payload(output) if isinstance(item, dict)]
        return {
            "recorded_output_available": True,
            "fundamentals": items[:100],
            "count": len(items[:100]),
        }

    unwrapped = _unwrap_output(output)
    text = (
        unwrapped
        if isinstance(unwrapped, str)
        else json.dumps(unwrapped, ensure_ascii=False, sort_keys=True)
    )
    return {
        "recorded_output_available": True,
        "excerpt": text[:4_000],
        "truncated": len(text) > 4_000,
    }


def _step_query(tool: str, args: dict[str, Any]) -> str:
    if tool == "finance_discover_companies":
        return "Identify the matching company record using the recorded keywords."
    if tool == "finance_discover_company_series":
        return (
            "Find the relevant financial series for the selected company and periods."
        )
    if tool == "finance_get_company_fundamentals":
        return (
            "Retrieve the recorded fundamental values for the selected company "
            "series and periods."
        )
    action = args["action"]
    return {
        "search": "Search the frozen research-web archive for the next evidence.",
        "open": "Open the selected page from the frozen research-web archive.",
        "find": "Find the recorded evidence pattern in the archived page.",
    }[action]


def _fixture_record_key(tool: str, args: dict[str, Any]) -> str:
    return _canonical_json({"tool": tool, "args": args})


def _merge_result_alternatives(results: list[dict[str, Any]]) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for result in results:
        unique.setdefault(_canonical_json(result), result)
    if len(unique) == 1:
        return deepcopy(next(iter(unique.values())))
    return {
        "recorded_output_available": any(
            item.get("recorded_output_available", False) for item in unique.values()
        ),
        "alternatives": [deepcopy(unique[key]) for key in sorted(unique)],
    }


def build(
    *,
    questions_path: Path,
    scores_path: Path,
    traces_path: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    _require_source_hash(questions_path, QUESTIONS_SHA256)
    _require_source_hash(scores_path, SCORES_SHA256)
    _require_source_hash(traces_path, TOOL_TRACES_SHA256)
    questions, traces = _load_parquet_rows(
        questions_path,
        scores_path,
        traces_path,
    )
    if sorted(questions) != list(range(500)):
        raise ValueError("FinRetrieval questions must have indices 0 through 499.")
    selected = _select_traces(traces)

    replay_buckets: dict[str, dict[str, Any]] = {}
    workflows: list[dict[str, Any]] = []
    selected_call_count = 0
    benchmark_call_count = 0
    excluded_over_step_limit: list[int] = []
    web_backed_indices: list[int] = []

    for index in sorted(selected):
        question = questions[index]
        trace = selected[index]
        calls = trace["calls"]
        steps: list[dict[str, Any]] = []
        uses_web = False

        for step_index, call in enumerate(calls):
            normalized_tool, normalized_args = _normalized_call(call)
            uses_web = uses_web or normalized_tool == "finance_search_web_archive"
            result = _compact_result(
                call,
                calls,
                step_index,
                normalized_tool,
            )
            key = _fixture_record_key(normalized_tool, normalized_args)
            bucket = replay_buckets.setdefault(
                key,
                {
                    "tool": normalized_tool,
                    "args": deepcopy(normalized_args),
                    "results": [],
                    "source_calls": [],
                },
            )
            bucket["results"].append(result)
            bucket["source_calls"].append(
                {
                    "source_index": index,
                    "step_index": step_index,
                    "configuration": trace["configuration"],
                    "source_tool": call["name"],
                    "source_call_id": call.get("id"),
                    "source_output_canonical_sha256": _sha256_value(call.get("output")),
                }
            )

            step_id = f"call_{step_index + 1:03d}"
            steps.append(
                {
                    "id": step_id,
                    "query": _step_query(normalized_tool, normalized_args),
                    "prompt_context": recorded_call_prompt_context(
                        normalized_tool,
                        normalized_args,
                    ),
                    "expected_tool": normalized_tool,
                    "expected_args": normalized_args,
                    "expected_answer": {
                        "tool": normalized_tool,
                        "arguments": normalized_args,
                        "offline_replay": True,
                        "network_access": False,
                    },
                    "depends_on": (
                        [] if step_index == 0 else [f"call_{step_index:03d}"]
                    ),
                    "source_program": _canonical_json(
                        {
                            "name": call["name"],
                            "input": call.get("input", {}),
                        }
                    ),
                    "source_tool": call["name"],
                    "source_call_id": call.get("id"),
                    "source_output_canonical_sha256": _sha256_value(call.get("output")),
                    "source_is_error": False,
                    "source_duration_ms": call.get("duration_ms"),
                }
            )

        if uses_web:
            web_backed_indices.append(index)
        selected_call_count += len(steps)
        if len(steps) > MAX_BENCHMARK_WORKFLOW_STEPS:
            excluded_over_step_limit.append(index)
            continue
        benchmark_call_count += len(steps)
        workflows.append(
            {
                "id": f"finance_multistep_finretrieval_{index:03d}",
                "domain": "finance",
                "task_type": "multi_step_tool_routing",
                "difficulty": "hard",
                "source": "public_finance_model_trajectory",
                "query": question["question"],
                "expected_steps": steps,
                "expected_final_answer": question["answer"],
                "source_dataset": "FinRetrieval",
                "source_index": index,
                "source_configuration": trace["configuration"],
                "source_num_tool_calls": len(steps),
                "source_trace_correct": True,
                "source_expected_value": trace["expected_value"],
                "source_expected_unit": trace["expected_unit"],
                "source_expected_currency": trace["expected_currency"],
                "source_answer": question["answer"],
                "source_value": question["value"],
                "source_unit": question["unit"],
                "source_category": question["category"],
                "source_ticker": question["ticker"],
                "source_company": question["company"],
                "source_country": question["country"],
                "source_fiscal_period": question["fiscal_period"],
                "source_calendar_period": question["calendar_period"],
                "source_period_type": question["period_type"],
                "query_origin": "official_research_dataset_question",
                "tool_sequence_origin": "selected_correct_model_trajectory",
                "tool_normalization": (
                    "Daloopa aliases are mapped to four deterministic LayerMCP "
                    "finance tools; recorded web clients are mapped to one frozen "
                    "archive tool. Original names and inputs remain on each step."
                ),
                "fixture_id": FIXTURE_ID,
                "source_repository": "daloopa/finretrieval",
                "source_revision": SOURCE_REVISION,
                "source_split": "questions_and_selected_scored_tool_traces",
                "source_questions_sha256": QUESTIONS_SHA256,
                "source_scores_sha256": SCORES_SHA256,
                "source_tool_traces_sha256": TOOL_TRACES_SHA256,
                "source_url": (
                    "https://huggingface.co/datasets/daloopa/finretrieval/"
                    f"tree/{SOURCE_REVISION}"
                ),
                "source_paper_url": "https://arxiv.org/abs/2603.04403",
                "source_license": "MIT",
                "provenance_type": (
                    "research_dataset_correct_model_trajectory_adaptation"
                ),
                "perturbation_type": "selected_research_trajectory",
                "notes": (
                    "Exact FinRetrieval question with one released correct, "
                    "no-error multi-call model trajectory. Tool aliases and "
                    "outputs are mechanically normalized for deterministic "
                    "offline execution."
                ),
            }
        )

    if excluded_over_step_limit != EXCLUDED_OVER_STEP_LIMIT:
        raise ValueError(
            "Unexpected FinRetrieval workflows above the five-call benchmark "
            f"limit: {excluded_over_step_limit!r}."
        )
    if len(workflows) != EXPECTED_BENCHMARK_WORKFLOWS:
        raise ValueError(
            f"Expected {EXPECTED_BENCHMARK_WORKFLOWS} benchmark workflows, got "
            f"{len(workflows)}."
        )
    if benchmark_call_count != EXPECTED_BENCHMARK_CALLS:
        raise ValueError(
            f"Expected {EXPECTED_BENCHMARK_CALLS} benchmark calls, got "
            f"{benchmark_call_count}."
        )

    replay_records: list[dict[str, Any]] = []
    for key in sorted(replay_buckets):
        bucket = replay_buckets[key]
        replay_records.append(
            {
                "tool": bucket["tool"],
                "args": bucket["args"],
                "result": _merge_result_alternatives(bucket["results"]),
                "source_calls": sorted(
                    bucket["source_calls"],
                    key=lambda item: (
                        item["source_index"],
                        item["step_index"],
                        item["configuration"],
                    ),
                ),
            }
        )

    fixture = {
        "fixture_id": FIXTURE_ID,
        "fixture_version": "finretrieval_replay_fixture_v1",
        "description": (
            "Compact deterministic replay of selected correct FinRetrieval "
            "tool calls. No runtime network or Daloopa access is performed."
        ),
        "records": replay_records,
        "manifest": {
            "source_dataset": "FinRetrieval",
            "source_repository": "daloopa/finretrieval",
            "source_revision": SOURCE_REVISION,
            "source_license": "MIT",
            "source_paper_url": "https://arxiv.org/abs/2603.04403",
            "source_questions_sha256": QUESTIONS_SHA256,
            "source_scores_sha256": SCORES_SHA256,
            "source_tool_traces_sha256": TOOL_TRACES_SHA256,
            "selection_rule": (
                "correct score; at least two calls; no call marked is_error; no "
                "recorded Output validation error; supported Daloopa/web tools; "
                "prefer Daloopa-only; then fewest calls, fewest web calls, and "
                "lexicographic configuration"
            ),
            # This manifest describes the complete replay fixture, including
            # selected source traces omitted from the <=5-call benchmark.
            "workflow_count": len(selected),
            "selected_call_count": selected_call_count,
            "replay_record_count": len(replay_records),
            "excluded_no_correct_trace_indices": EXCLUDED_NO_CORRECT_TRACE,
            "web_backed_indices": web_backed_indices,
            "synthetic": False,
            "network_access": False,
        },
    }

    benchmark_path = output_root / BENCHMARK_FILENAME
    fixture_path = output_root / "fixtures" / FIXTURE_FILENAME
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(
        json.dumps(workflows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fixture_path.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return benchmark_path, fixture_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--tool-traces", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    benchmark_path, fixture_path = build(
        questions_path=args.questions,
        scores_path=args.scores,
        traces_path=args.tool_traces,
        output_root=args.output_root,
    )
    print(benchmark_path)
    print(fixture_path)


if __name__ == "__main__":
    main()
