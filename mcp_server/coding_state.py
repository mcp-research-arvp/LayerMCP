from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CODING_REPOSITORY_ID = "example/research-mcp"


@dataclass(frozen=True)
class CodingRepository:
    """Server-owned repository workspace used by the coding research tools."""

    repo_id: str
    path: Path
    base_commit: str


_CODING_LOCK = threading.RLock()
_WORKSPACE_DIRECTORY: tempfile.TemporaryDirectory[str] | None = None
_REPOSITORIES: dict[str, CodingRepository] = {}


_INITIAL_FILES = {
    "README.md": """# Example Research MCP

Deterministic repository fixture for LayerMCP coding-tool experiments.
""",
    "src/__init__.py": "",
    "src/auth.py": """DEMO_TOKEN = "offline-demo-token"


def authenticate(token: str) -> bool:
    return token == DEMO_TOKEN
""",
    "tests/test_auth.py": """from src.auth import authenticate


def test_authenticate_accepts_demo_token() -> None:
    assert authenticate("offline-demo-token")


def test_authenticate_rejects_unknown_token() -> None:
    assert not authenticate("unknown")
""",
}

_PAYMENTS_FILES = {
    "README.md": """# Example Research MCP

Deterministic repository fixture for LayerMCP coding-tool experiments.

The fixture contains authentication and invoice-total examples.
""",
    "src/payments.py": """from collections.abc import Iterable


def calculate_invoice_total(items: Iterable[dict[str, float]]) -> float:
    return round(sum(item["price"] for item in items), 2)
""",
    "tests/test_payments.py": """from src.payments import calculate_invoice_total


def test_calculate_invoice_total() -> None:
    items = [{"price": 12.50}, {"price": 7.25}]
    assert calculate_invoice_total(items) == 19.75
""",
}

_AUTH_NORMALIZATION_FILES = {
    "src/auth.py": """DEMO_TOKEN = "offline-demo-token"


def normalize_token(token: str) -> str:
    return token.strip()


def authenticate(token: str) -> bool:
    return normalize_token(token) == DEMO_TOKEN
""",
    "tests/test_auth.py": """from src.auth import authenticate


def test_authenticate_accepts_demo_token() -> None:
    assert authenticate("offline-demo-token")


def test_authenticate_normalizes_whitespace() -> None:
    assert authenticate("  offline-demo-token  ")


def test_authenticate_rejects_unknown_token() -> None:
    assert not authenticate("unknown")
""",
}


def _git_environment(home: Path, timestamp: str | None = None) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    if timestamp is not None:
        env.update(
            {
                "GIT_AUTHOR_DATE": timestamp,
                "GIT_COMMITTER_DATE": timestamp,
            }
        )
    return env


def _run_git(
    repository: Path,
    *arguments: str,
    home: Path,
    timestamp: str | None = None,
) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError(
            "Git is required to initialize the coding repository fixture."
        )

    completed = subprocess.run(
        [git, "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        env=_git_environment(home, timestamp),
        shell=False,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        error = (
            completed.stderr.strip() or completed.stdout.strip() or "unknown Git error"
        )
        raise RuntimeError(f"Unable to initialize coding fixture: {error[:500]}")
    return completed.stdout.strip()


def _write_snapshot(repository: Path, files: dict[str, str]) -> None:
    for relative_path, content in files.items():
        destination = repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")


def _commit_snapshot(
    repository: Path,
    home: Path,
    files: dict[str, str],
    message: str,
    timestamp: str,
) -> None:
    _write_snapshot(repository, files)
    _run_git(repository, "add", "--all", home=home)
    _run_git(
        repository,
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "--message",
        message,
        home=home,
        timestamp=timestamp,
    )


def _create_fixture(root: Path) -> CodingRepository:
    repository = root / "repository"
    git_home = root / "git-home"
    repository.mkdir(parents=True)
    git_home.mkdir(parents=True)

    _run_git(repository, "init", "--quiet", "--initial-branch=main", home=git_home)
    _run_git(repository, "config", "user.name", "LayerMCP Fixture", home=git_home)
    _run_git(
        repository,
        "config",
        "user.email",
        "fixture@layermcp.invalid",
        home=git_home,
    )
    _run_git(repository, "config", "core.autocrlf", "false", home=git_home)

    _commit_snapshot(
        repository,
        git_home,
        _INITIAL_FILES,
        "Initialize authentication fixture",
        "2024-01-10T12:00:00+00:00",
    )
    _commit_snapshot(
        repository,
        git_home,
        _PAYMENTS_FILES,
        "Add invoice total calculation",
        "2024-02-15T12:00:00+00:00",
    )
    _commit_snapshot(
        repository,
        git_home,
        _AUTH_NORMALIZATION_FILES,
        "Normalize authentication tokens",
        "2024-03-20T12:00:00+00:00",
    )

    base_commit = _run_git(repository, "rev-parse", "HEAD", home=git_home)
    return CodingRepository(
        repo_id=DEFAULT_CODING_REPOSITORY_ID,
        path=repository.resolve(),
        base_commit=base_commit,
    )


def _ensure_coding_state() -> None:
    global _WORKSPACE_DIRECTORY
    if _WORKSPACE_DIRECTORY is not None:
        return

    _WORKSPACE_DIRECTORY = tempfile.TemporaryDirectory(prefix="layermcp-coding-")
    workspace_root = Path(_WORKSPACE_DIRECTORY.name)
    repository = _create_fixture(workspace_root)
    _REPOSITORIES[repository.repo_id] = repository


def get_coding_lock() -> threading.RLock:
    return _CODING_LOCK


def get_coding_repository(
    repo_id: str = DEFAULT_CODING_REPOSITORY_ID,
) -> CodingRepository:
    if not isinstance(repo_id, str):
        raise ValueError("repo_id must be a string.")
    normalized = repo_id.strip()
    with _CODING_LOCK:
        _ensure_coding_state()
        repository = _REPOSITORIES.get(normalized)
        if repository is None:
            available = ", ".join(sorted(_REPOSITORIES))
            raise ValueError(f"repo_id must be one of: {available}")
        return repository


def snapshot_coding_state() -> dict[str, Any]:
    with _CODING_LOCK:
        _ensure_coding_state()
        return {
            "repositories": [
                {
                    "repo_id": repository.repo_id,
                    "base_commit": repository.base_commit,
                }
                for repository in sorted(
                    _REPOSITORIES.values(), key=lambda item: item.repo_id
                )
            ]
        }


def reset_coding_state() -> dict[str, Any]:
    global _WORKSPACE_DIRECTORY
    with _CODING_LOCK:
        if _WORKSPACE_DIRECTORY is not None:
            _WORKSPACE_DIRECTORY.cleanup()
        _WORKSPACE_DIRECTORY = None
        _REPOSITORIES.clear()
        _ensure_coding_state()
        return snapshot_coding_state()


def _cleanup_coding_state() -> None:
    global _WORKSPACE_DIRECTORY
    with _CODING_LOCK:
        if _WORKSPACE_DIRECTORY is not None:
            _WORKSPACE_DIRECTORY.cleanup()
        _WORKSPACE_DIRECTORY = None
        _REPOSITORIES.clear()


atexit.register(_cleanup_coding_state)
