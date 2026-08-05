from __future__ import annotations

import json
from pathlib import Path
import re
import unittest
from typing import Any

from benchmark.finance.apply_grounding import (
    apply_grounding,
    ground_benchmark_rows,
)
from benchmark.finance.grounding import (
    CALCULATOR_GROUNDING_KIND,
    NORMALIZED_CALL_GROUNDING_KIND,
    PROMPT_CONTEXT_CHAR_LIMIT,
    RECORDED_CALL_GROUNDING_KIND,
    TABLE_GROUNDING_KIND,
)
from evaluation.evaluate import (
    _multistep_query,
    _query_with_context,
    load_benchmark,
)
from mcp_server.finance_state import get_finance_fixture
from mcp_server.finretrieval_state import FINRETRIEVAL_REPLAY_TOOL_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark" / "finance"
BENCHMARK_PATHS = sorted(
    BENCHMARK_ROOT.glob("finance_*.json")
)
SOURCE_FILTER_COLUMNS = (
    "source_id",
    "source_table_uid",
    "source_row_index",
    "operation_index",
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(
        isinstance(row, dict) for row in value
    ):
        raise AssertionError(f"{path} must contain a list of objects.")
    return value


def _expected_sql_filter(sql: str, column: str) -> str | int | None:
    quoted = re.search(
        rf"\b{re.escape(column)}\s*=\s*'((?:''|[^'])*)'",
        sql,
        flags=re.IGNORECASE,
    )
    if quoted is not None:
        return quoted.group(1).replace("''", "'")
    numeric = re.search(
        rf"\b{re.escape(column)}\s*=\s*(-?\d+)\b",
        sql,
        flags=re.IGNORECASE,
    )
    if numeric is not None:
        return int(numeric.group(1))
    return None


class FinanceGroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = get_finance_fixture()["tables"]
        cls.rows_by_path = {
            path: _load_rows(path) for path in BENCHMARK_PATHS
        }

    def test_grounding_rewrite_is_deterministic_and_has_no_drift(self) -> None:
        self.assertEqual(apply_grounding(check=True), [])
        for path, rows in self.rows_by_path.items():
            with self.subTest(path=path.name):
                self.assertEqual(
                    ground_benchmark_rows(rows, self.tables),
                    rows,
                )

    def test_no_finance_workflow_exceeds_five_steps(self) -> None:
        for path, rows in self.rows_by_path.items():
            for row in rows:
                with self.subTest(path=path.name, row=row["id"]):
                    self.assertLessEqual(len(row.get("expected_steps", [])), 5)

    def test_every_finance_call_has_parseable_bounded_context(self) -> None:
        call_count = 0
        kind_counts: dict[str, int] = {}
        for path, rows in self.rows_by_path.items():
            for row in rows:
                for call in row.get("expected_steps") or [row]:
                    call_count += 1
                    with self.subTest(
                        path=path.name,
                        row=row["id"],
                        step=call.get("id"),
                    ):
                        context = call.get("prompt_context")
                        self.assertIsInstance(context, str)
                        assert isinstance(context, str)
                        self.assertLessEqual(
                            len(context),
                            PROMPT_CONTEXT_CHAR_LIMIT,
                        )
                        payload = json.loads(context)
                        self.assertIsInstance(payload, dict)
                        kind = payload["kind"]
                        kind_counts[kind] = kind_counts.get(kind, 0) + 1
                        self._assert_call_context(call, payload)

        self.assertEqual(call_count, 3_408)
        self.assertEqual(
            kind_counts,
            {
                TABLE_GROUNDING_KIND: 1_192,
                CALCULATOR_GROUNDING_KIND: 636,
                RECORDED_CALL_GROUNDING_KIND: 1_490,
                NORMALIZED_CALL_GROUNDING_KIND: 90,
            },
        )

    def test_context_is_present_in_the_actual_routed_prompt(self) -> None:
        for path, rows in self.rows_by_path.items():
            samples = load_benchmark(path)
            self.assertEqual(len(samples), len(rows))
            for row, sample in zip(rows, samples, strict=True):
                if row.get("expected_steps"):
                    self.assertEqual(
                        len(row["expected_steps"]),
                        len(sample.expected_steps),
                    )
                    for raw_step, step in zip(
                        row["expected_steps"],
                        sample.expected_steps,
                        strict=True,
                    ):
                        with self.subTest(
                            path=path.name,
                            row=row["id"],
                            step=raw_step["id"],
                        ):
                            routed = _multistep_query(sample, step, [])
                            self.assertIn(raw_step["prompt_context"], routed)
                else:
                    with self.subTest(path=path.name, row=row["id"]):
                        routed = _query_with_context(
                            sample.query,
                            sample.prompt_context,
                        )
                        self.assertIn(row["prompt_context"], routed)

    def _assert_call_context(
        self,
        call: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        expected_tool = call["expected_tool"]
        expected_args = call["expected_args"]
        if expected_tool == "finance_query_table":
            self.assertEqual(payload["kind"], TABLE_GROUNDING_KIND)
            self._assert_table_context(expected_args, payload)
            return
        if expected_tool == "calculator":
            self.assertEqual(payload["kind"], CALCULATOR_GROUNDING_KIND)
            self.assertEqual(payload["arguments"], expected_args)
            self.assertEqual(
                payload["arguments"]["expression"],
                expected_args["expression"],
            )
            return
        if expected_tool in FINRETRIEVAL_REPLAY_TOOL_NAMES:
            self.assertEqual(payload["kind"], RECORDED_CALL_GROUNDING_KIND)
            self.assertEqual(payload["tool"], expected_tool)
            self.assertEqual(payload["arguments"], expected_args)
            self.assertIn("trace data", payload["instruction"])
            return
        self.assertEqual(payload["kind"], NORMALIZED_CALL_GROUNDING_KIND)
        self.assertEqual(payload["arguments"], expected_args)

    def _assert_table_context(
        self,
        expected_args: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        dataset_id = expected_args["dataset_id"]
        sql = expected_args["sql"]
        table = self.tables[dataset_id]
        self.assertEqual(payload["dataset_id"], dataset_id)
        self.assertEqual(payload["table_name"], "data")
        self.assertEqual(payload["columns"], table["columns"])
        self.assertTrue(payload["columns"])

        expected_filter = {
            column: value
            for column in SOURCE_FILTER_COLUMNS
            if (value := _expected_sql_filter(sql, column)) is not None
        }
        self.assertEqual(payload["source_filter"], expected_filter)

        if dataset_id == "finqa-public-test-program-results-v1":
            self.assertNotIn("relevant_rows", payload)
            self.assertEqual(
                set(payload["source_filter"]),
                {"source_row_index", "operation_index"},
            )
            return

        relevant_rows = payload.get("relevant_rows")
        self.assertIsInstance(relevant_rows, list)
        assert isinstance(relevant_rows, list)
        self.assertTrue(relevant_rows)
        stable_filter = {
            key: value
            for key, value in expected_filter.items()
            if key != "operation_index"
        }
        for row in relevant_rows:
            self.assertTrue(
                all(row.get(key) == value for key, value in stable_filter.items())
            )


if __name__ == "__main__":
    unittest.main()
