from __future__ import annotations

import inspect
import re
import unittest

from mcp_server.coding_state import (
    CODING_FIXTURE_VERSION,
    CODESEARCHNET_CODING_FIXTURE_VERSION,
    CODESEARCHNET_CODING_REPOSITORY_ID,
    CODESEARCHNET_SOURCE_REVISION,
    DEFAULT_CODING_REPOSITORY_ID,
    snapshot_coding_state,
)
from mcp_server.coding_tools import (
    CODING_TOOL_NAMES,
    code_list_files,
    code_read_file,
    code_search_text,
    git_diff,
    git_log,
    git_show,
    git_status,
)
from mcp_server.server import mcp


TOOL_FUNCTIONS = {
    "code_list_files": code_list_files,
    "code_read_file": code_read_file,
    "code_search_text": code_search_text,
    "git_log": git_log,
    "git_show": git_show,
    "git_diff": git_diff,
    "git_status": git_status,
}


class CodingToolTests(unittest.TestCase):
    def test_exact_tool_catalog_and_signatures_are_registered(self) -> None:
        self.assertEqual(CODING_TOOL_NAMES, frozenset(TOOL_FUNCTIONS))
        self.assertTrue(CODING_TOOL_NAMES <= set(mcp._tool_manager._tools))

        expected_parameters = {
            "code_list_files": ["repo_id", "path", "glob", "max_results"],
            "code_read_file": ["repo_id", "path", "start_line", "end_line"],
            "code_search_text": [
                "repo_id",
                "pattern",
                "path_glob",
                "case_sensitive",
                "max_results",
            ],
            "git_log": ["repo_id", "path", "max_count"],
            "git_show": ["repo_id", "revision", "path"],
            "git_diff": [
                "repo_id",
                "base",
                "head",
                "path",
                "context_lines",
            ],
            "git_status": [
                "repo_id",
                "path",
                "include_untracked",
                "max_entries",
            ],
        }
        for name, function in TOOL_FUNCTIONS.items():
            with self.subTest(tool=name):
                self.assertEqual(
                    list(inspect.signature(function).parameters),
                    expected_parameters[name],
                )

    def test_repository_inventory_is_deterministic(self) -> None:
        first_state = snapshot_coding_state()
        self.assertEqual(first_state, snapshot_coding_state())
        repositories = {
            repository["repo_id"]: repository
            for repository in first_state["repositories"]
        }
        self.assertEqual(
            set(repositories),
            {DEFAULT_CODING_REPOSITORY_ID, CODESEARCHNET_CODING_REPOSITORY_ID},
        )
        self.assertEqual(
            repositories[DEFAULT_CODING_REPOSITORY_ID]["fixture_version"],
            CODING_FIXTURE_VERSION,
        )
        self.assertTrue(
            all(
                re.fullmatch(r"[0-9a-f]{40,64}", repository["base_commit"])
                for repository in repositories.values()
            )
        )

        controlled = code_list_files(DEFAULT_CODING_REPOSITORY_ID)
        self.assertEqual(
            [entry["path"] for entry in controlled["files"]],
            [
                "README.md",
                "src/__init__.py",
                "src/auth.py",
                "src/payments.py",
                "tests/test_auth.py",
                "tests/test_payments.py",
            ],
        )
        self.assertEqual(controlled["count"], 6)
        self.assertFalse(controlled["truncated"])

        public = code_list_files(CODESEARCHNET_CODING_REPOSITORY_ID)
        self.assertEqual(
            [entry["path"] for entry in public["files"]],
            [
                "README.md",
                "resources/annotationStore_selected.jsonl",
                "resources/queries_selected.txt",
            ],
        )
        self.assertEqual(public["count"], 3)
        self.assertFalse(public["truncated"])

    def test_codesearchnet_fixture_has_pinned_provenance(self) -> None:
        state = snapshot_coding_state()
        repository = next(
            repository
            for repository in state["repositories"]
            if repository["repo_id"] == CODESEARCHNET_CODING_REPOSITORY_ID
        )
        self.assertEqual(
            repository["fixture_version"], CODESEARCHNET_CODING_FIXTURE_VERSION
        )
        provenance = repository["provenance"]
        self.assertEqual(
            provenance["source_dataset"],
            "CodeSearchNet Challenge human evaluation",
        )
        self.assertEqual(provenance["source_repository"], "github/CodeSearchNet")
        self.assertEqual(provenance["source_revision"], CODESEARCHNET_SOURCE_REVISION)
        self.assertEqual(provenance["source_license"], "MIT")
        self.assertEqual(
            provenance["query_origin"], "codesearchnet_published_query"
        )
        self.assertEqual(
            provenance["provenance_type"], "research_dataset_adaptation"
        )
        self.assertEqual(
            provenance["source_query_sha256"],
            "037509c717c2e164721f0fd3ea45cb05f36669551af643f53930a92b76b146cf",
        )
        self.assertEqual(
            provenance["source_annotation_sha256"],
            "0340af32b551ceadb74fec147f97642b7fedf3ff039e38fb86baff49ee899846",
        )

        queries = code_read_file(
            CODESEARCHNET_CODING_REPOSITORY_ID,
            "resources/queries_selected.txt",
        )
        self.assertEqual(queries["total_lines"], 15)
        self.assertTrue(queries["content"].startswith("k means clustering\n"))
        self.assertTrue(queries["content"].endswith("linear regression\n"))

    def test_codesearchnet_fixture_has_one_fixed_initial_commit(self) -> None:
        state = snapshot_coding_state()
        repository = next(
            repository
            for repository in state["repositories"]
            if repository["repo_id"] == CODESEARCHNET_CODING_REPOSITORY_ID
        )
        history = git_log(CODESEARCHNET_CODING_REPOSITORY_ID, max_count=10)
        self.assertEqual(history["count"], 1)
        self.assertEqual(history["commits"][0]["sha"], repository["base_commit"])
        self.assertEqual(history["commits"][0]["author"], "LayerMCP Fixture")
        self.assertEqual(
            history["commits"][0]["authored_at"],
            "2024-04-10T12:00:00+00:00",
        )
        self.assertEqual(
            history["commits"][0]["subject"],
            "Initialize CodeSearchNet public annotation fixture",
        )

    def test_expected_codesearchnet_search_is_exact_and_bounded(self) -> None:
        result = code_search_text(
            CODESEARCHNET_CODING_REPOSITORY_ID,
            "k means clustering",
            "resources/annotationStore_selected.jsonl",
            True,
            1,
        )
        self.assertEqual(result["repo_id"], CODESEARCHNET_CODING_REPOSITORY_ID)
        self.assertEqual(result["count"], 1)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["engine"], "ripgrep-fixed-string")
        self.assertEqual(result["source"], "coding-fixture")
        match = result["matches"][0]
        self.assertEqual(match["path"], "resources/annotationStore_selected.jsonl")
        self.assertEqual(match["line"], 1)
        self.assertEqual(match["column"], 148)
        self.assertEqual(match["columns"], [148, 391])
        self.assertIn('"source_annotation_index_zero_based":1635', match["text"])
        self.assertIn('"query":"k means clustering"', match["text"])

    def test_invalid_identifiers_paths_ranges_and_limits_are_rejected(self) -> None:
        invalid_calls = [
            lambda: code_list_files("unknown/repository"),
            lambda: code_list_files(DEFAULT_CODING_REPOSITORY_ID, "../outside"),
            lambda: code_list_files(DEFAULT_CODING_REPOSITORY_ID, glob="../*.py"),
            lambda: code_list_files(DEFAULT_CODING_REPOSITORY_ID, max_results=0),
            lambda: code_read_file(DEFAULT_CODING_REPOSITORY_ID, ".git/config"),
            lambda: code_read_file(DEFAULT_CODING_REPOSITORY_ID, "README.md", 0),
            lambda: code_read_file(DEFAULT_CODING_REPOSITORY_ID, "README.md", 3, 2),
            lambda: code_search_text(DEFAULT_CODING_REPOSITORY_ID, ""),
            lambda: code_search_text(
                DEFAULT_CODING_REPOSITORY_ID,
                "token",
                case_sensitive="yes",  # type: ignore[arg-type]
            ),
            lambda: code_search_text(
                DEFAULT_CODING_REPOSITORY_ID, "token", path_glob="../*"
            ),
            lambda: git_log(DEFAULT_CODING_REPOSITORY_ID, max_count=0),
            lambda: git_show(DEFAULT_CODING_REPOSITORY_ID, "HEAD^"),
            lambda: git_diff(
                DEFAULT_CODING_REPOSITORY_ID,
                "HEAD~1",
                "HEAD",
                context_lines=21,
            ),
            lambda: git_status(
                DEFAULT_CODING_REPOSITORY_ID,
                include_untracked="yes",  # type: ignore[arg-type]
            ),
            lambda: git_status(DEFAULT_CODING_REPOSITORY_ID, max_entries=0),
        ]
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
