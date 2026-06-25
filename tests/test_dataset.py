from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from evaluation.dataset import (
    BenchmarkValidationError,
    NO_TOOL_NAME,
    load_benchmark,
)


class DatasetTests(unittest.TestCase):
    def test_default_benchmark_file_is_valid(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        samples = load_benchmark(project_root / "benchmark" / "tool_routing.jsonl")

        self.assertGreaterEqual(len(samples), 10)
        self.assertTrue(any(not sample.expects_tool_call for sample in samples))

    def test_loads_jsonl_benchmark_sample(self) -> None:
        path = self.write_jsonl([self.sample_record()])

        samples = load_benchmark(path)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].id, "sample_001")
        self.assertEqual(samples[0].tools, ("calculator", "github_search"))
        self.assertEqual(samples[0].expected_arguments, {"expression": "2 + 2"})

    def test_supports_legacy_json_list_files(self) -> None:
        path = self.write_json([self.sample_record()])

        samples = load_benchmark(path)

        self.assertEqual(samples[0].expected_tool, "calculator")

    def test_rejects_missing_required_fields(self) -> None:
        record = self.sample_record()
        del record["domain"]

        with self.assertRaisesRegex(BenchmarkValidationError, "domain"):
            load_benchmark(self.write_jsonl([record]))

    def test_rejects_expected_tool_not_in_tools(self) -> None:
        record = self.sample_record(expected_tool="customer_lookup")

        with self.assertRaisesRegex(BenchmarkValidationError, "expected_tool"):
            load_benchmark(self.write_jsonl([record]))

    def test_rejects_no_tool_sample_with_expected_arguments(self) -> None:
        record = self.sample_record(
            expected_tool=NO_TOOL_NAME,
            expected_arguments={"expression": "2 + 2"},
        )

        with self.assertRaisesRegex(BenchmarkValidationError, "expected_arguments"):
            load_benchmark(self.write_jsonl([record]))

    def test_rejects_duplicate_sample_ids(self) -> None:
        first = self.sample_record()
        second = self.sample_record()

        with self.assertRaisesRegex(BenchmarkValidationError, "unique"):
            load_benchmark(self.write_jsonl([first, second]))

    def sample_record(self, **overrides: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": "sample_001",
            "domain": "math",
            "query": "What is 2 + 2?",
            "tools": ["calculator", "github_search"],
            "expected_tool": "calculator",
            "expected_arguments": {"expression": "2 + 2"},
            "expected_result": {"expression": "2 + 2", "result": 4},
            "difficulty": "easy",
        }
        record.update(overrides)
        return record

    def write_jsonl(self, records: list[dict[str, Any]]) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        path = Path(temp_dir.name) / "benchmark.jsonl"
        lines = [json.dumps(record, sort_keys=True) for record in records]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def write_json(self, records: list[dict[str, Any]]) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        path = Path(temp_dir.name) / "benchmark.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
