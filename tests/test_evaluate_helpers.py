from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from evaluation.evaluate import _summarize_tool_result


class EvaluateHelperTests(unittest.TestCase):
    def test_summarize_structured_tool_result(self) -> None:
        result = SimpleNamespace(structuredContent={"answer": 42})

        self.assertEqual(_summarize_tool_result(result), json.dumps({"answer": 42}))

    def test_summarize_text_tool_result(self) -> None:
        result = SimpleNamespace(content=[SimpleNamespace(text="hello")])

        self.assertEqual(_summarize_tool_result(result), "hello")


if __name__ == "__main__":
    unittest.main()
