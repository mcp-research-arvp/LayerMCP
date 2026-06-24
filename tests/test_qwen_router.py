from __future__ import annotations

import unittest

from models.qwen_router import HALLUCINATED_TOOL, _extract_tool_name


class QwenRouterHelperTests(unittest.TestCase):
    def test_extracts_exact_tool_name(self) -> None:
        self.assertEqual(
            _extract_tool_name("calculator", ["calculator", "github_search"]),
            "calculator",
        )

    def test_extracts_tool_name_from_extra_text(self) -> None:
        self.assertEqual(
            _extract_tool_name("Use github_search.", ["calculator", "github_search"]),
            "github_search",
        )

    def test_unknown_output_maps_to_hallucinated_tool(self) -> None:
        self.assertEqual(
            _extract_tool_name("web_search", ["calculator", "github_search"]),
            HALLUCINATED_TOOL,
        )


if __name__ == "__main__":
    unittest.main()
