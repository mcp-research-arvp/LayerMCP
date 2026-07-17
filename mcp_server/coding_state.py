from __future__ import annotations

import atexit
from copy import deepcopy
import json
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_CODING_REPOSITORY_ID = "example/research-mcp"
CODING_FIXTURE_VERSION = "coding_fixture_v1"
CODESEARCHNET_CODING_REPOSITORY_ID = "codesearchnet-public-v1"
CODESEARCHNET_CODING_FIXTURE_VERSION = "coding_codesearchnet_fixture_v1"
CODESEARCHNET_SOURCE_REVISION = "bb121a53a559e99a6849409355ee5c83803f2e87"


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CODESEARCHNET_FIXTURE_PATH = (
    _PROJECT_ROOT
    / "benchmark"
    / "coding"
    / "fixtures"
    / "codesearchnet_public_annotations.json"
)
_MAX_DECLARATIVE_FILES = 100
_MAX_DECLARATIVE_FILE_BYTES = 256 * 1024
_MAX_DECLARATIVE_TOTAL_BYTES = 1024 * 1024
_MAX_DECLARATIVE_PATH_LENGTH = 1024


@dataclass(frozen=True)
class CodingRepository:
    """Server-owned repository workspace used by the coding research tools."""

    repo_id: str
    path: Path
    base_commit: str
    fixture_version: str
    provenance: dict[str, Any]


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


def _validate_declarative_path(path: object) -> str:
    if not isinstance(path, str):
        raise RuntimeError("Coding fixture file paths must be strings.")
    if not path or len(path) > _MAX_DECLARATIVE_PATH_LENGTH:
        raise RuntimeError(
            "Coding fixture file paths must contain "
            f"1-{_MAX_DECLARATIVE_PATH_LENGTH} characters."
        )
    if "\x00" in path or "\\" in path:
        raise RuntimeError(
            "Coding fixture file paths must be NUL-free POSIX relative paths."
        )

    pure_path = PurePosixPath(path)
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or pure_path.as_posix() == "."
        or path != pure_path.as_posix()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise RuntimeError("Coding fixture file paths must be normalized and relative.")
    try:
        path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RuntimeError("Coding fixture file paths must be valid UTF-8.") from exc
    if any(part.rstrip(" .").casefold() == ".git" for part in pure_path.parts):
        raise RuntimeError("Coding fixture files must not create Git metadata paths.")
    return pure_path.as_posix()


def _load_codesearchnet_fixture() -> tuple[dict[str, str], dict[str, Any]]:
    try:
        fixture = json.loads(_CODESEARCHNET_FIXTURE_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Unable to load the declarative CodeSearchNet coding fixture."
        ) from exc

    if not isinstance(fixture, dict):
        raise RuntimeError("The CodeSearchNet coding fixture must be a JSON object.")
    if fixture.get("repo_id") != CODESEARCHNET_CODING_REPOSITORY_ID:
        raise RuntimeError(
            "The CodeSearchNet coding fixture has an unexpected repo_id."
        )
    if fixture.get("fixture_version") != CODESEARCHNET_CODING_FIXTURE_VERSION:
        raise RuntimeError(
            "The CodeSearchNet coding fixture has an unexpected fixture_version."
        )

    raw_files = fixture.get("files")
    if (
        not isinstance(raw_files, dict)
        or not raw_files
        or len(raw_files) > _MAX_DECLARATIVE_FILES
    ):
        raise RuntimeError(
            "The CodeSearchNet coding fixture must declare between 1 and "
            f"{_MAX_DECLARATIVE_FILES} files."
        )

    files: dict[str, str] = {}
    total_bytes = 0
    for raw_path, content in raw_files.items():
        path = _validate_declarative_path(raw_path)
        if not isinstance(content, str) or "\x00" in content:
            raise RuntimeError("Coding fixture files must contain NUL-free UTF-8 text.")
        try:
            content_size = len(content.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                "Coding fixture files must contain valid UTF-8 text."
            ) from exc
        if content_size > _MAX_DECLARATIVE_FILE_BYTES:
            raise RuntimeError(
                "A CodeSearchNet fixture file exceeds the bounded file size."
            )
        files[path] = content
        total_bytes += content_size

    if total_bytes > _MAX_DECLARATIVE_TOTAL_BYTES:
        raise RuntimeError("The CodeSearchNet fixture exceeds the bounded total size.")
    file_paths = set(files)
    for path in file_paths:
        parts = PurePosixPath(path).parts
        for end in range(1, len(parts)):
            parent = PurePosixPath(*parts[:end]).as_posix()
            if parent in file_paths:
                raise RuntimeError(
                    "Coding fixture paths must not use a file as a parent directory."
                )

    provenance = fixture.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("The CodeSearchNet coding fixture requires provenance.")
    required_provenance = {
        "source_revision": CODESEARCHNET_SOURCE_REVISION,
        "query_origin": "codesearchnet_published_query",
        "provenance_type": "research_dataset_adaptation",
    }
    for field, expected in required_provenance.items():
        if provenance.get(field) != expected:
            raise RuntimeError(
                f"The CodeSearchNet coding fixture has unexpected {field}."
            )

    return files, deepcopy(provenance)


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
        fixture_version=CODING_FIXTURE_VERSION,
        provenance={
            "provenance_type": "controlled_fixture",
            "synthetic": True,
        },
    )


def _create_codesearchnet_fixture(root: Path) -> CodingRepository:
    files, provenance = _load_codesearchnet_fixture()
    repository = root / "codesearchnet-public-repository"
    git_home = root / "codesearchnet-public-git-home"
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
        files,
        "Initialize CodeSearchNet public annotation fixture",
        "2024-04-10T12:00:00+00:00",
    )

    base_commit = _run_git(repository, "rev-parse", "HEAD", home=git_home)
    return CodingRepository(
        repo_id=CODESEARCHNET_CODING_REPOSITORY_ID,
        path=repository.resolve(),
        base_commit=base_commit,
        fixture_version=CODESEARCHNET_CODING_FIXTURE_VERSION,
        provenance=provenance,
    )


def _ensure_coding_state() -> None:
    global _WORKSPACE_DIRECTORY
    if _WORKSPACE_DIRECTORY is not None:
        return

    workspace_directory = tempfile.TemporaryDirectory(prefix="layermcp-coding-")
    workspace_root = Path(workspace_directory.name)
    try:
        repositories = (
            _create_fixture(workspace_root),
            _create_codesearchnet_fixture(workspace_root),
        )
    except Exception:
        workspace_directory.cleanup()
        raise

    _REPOSITORIES.clear()
    _REPOSITORIES.update(
        {repository.repo_id: repository for repository in repositories}
    )
    _WORKSPACE_DIRECTORY = workspace_directory


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
                    "fixture_version": repository.fixture_version,
                    "provenance": deepcopy(repository.provenance),
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
