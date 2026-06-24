from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation.evaluate import _load_dataset, _summarize_tool_result


class EvaluateHelperTests(unittest.TestCase):
    def test_load_dataset_reads_json_list(self) -> None:
        path = self.create_temp_file('[{"query": "q", "expected_tool": "tool"}]')

        self.assertEqual(_load_dataset(path), [{"query": "q", "expected_tool": "tool"}])

    def test_load_dataset_rejects_non_list_json(self) -> None:
        path = self.create_temp_file('{"query": "q"}')

        with self.assertRaisesRegex(ValueError, "must be a JSON list"):
            _load_dataset(path)

    def test_summarize_structured_tool_result(self) -> None:
        result = SimpleNamespace(structuredContent={"answer": 42})

        self.assertEqual(_summarize_tool_result(result), json.dumps({"answer": 42}))

    def test_summarize_text_tool_result(self) -> None:
        result = SimpleNamespace(content=[SimpleNamespace(text="hello")])

        self.assertEqual(_summarize_tool_result(result), "hello")

    def create_temp_file(self, content: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        path = Path(temp_dir.name) / "dataset.json"
        path.write_text(content, encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
