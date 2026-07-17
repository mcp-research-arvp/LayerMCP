from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from mcp_server.coding_state import (
    CodingRepository,
    get_coding_lock,
    get_coding_repository,
)


_SOURCE = "coding-fixture"
CODING_TOOL_NAMES = frozenset(
    {
        "code_list_files",
        "code_read_file",
        "code_search_text",
        "git_log",
        "git_show",
        "git_diff",
        "git_status",
    }
)
_MAX_PATH_LENGTH = 1024
_MAX_GLOB_LENGTH = 512
_MAX_PATTERN_LENGTH = 4096
_MAX_LIST_RESULTS = 500
_MAX_SEARCH_RESULTS = 500
_MAX_STATUS_ENTRIES = 500
_MAX_READ_BYTES = 256 * 1024
_MAX_READ_LINES = 2000
_MAX_COMMAND_OUTPUT = 1024 * 1024
_MAX_DIFF_OUTPUT = 256 * 1024
_MAX_SEARCH_EXCERPT_BYTES = 2000
_MAX_SEARCH_RESPONSE_BYTES = 256 * 1024
_MAX_SEARCH_COLUMNS = 1000
_COMMAND_TIMEOUT_SECONDS = 15
_REVISION_PATTERN = re.compile(r"(?:HEAD(?:~[0-9]{1,3})?|[0-9a-fA-F]{7,40})\Z")
_BRANCH_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}\Z")


@dataclass(frozen=True)
class _ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    output_truncated: bool


class _ProcessTimeout(RuntimeError):
    pass


def _validate_limit(value: int, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer.")
    if value < 1 or value > maximum:
        raise ValueError(f"{field} must be between 1 and {maximum}.")
    return value


def _normalize_relative_path(path: str, field: str = "path") -> str:
    if not isinstance(path, str):
        raise ValueError(f"{field} must be a string.")
    if not path or len(path) > _MAX_PATH_LENGTH:
        raise ValueError(f"{field} must contain 1-{_MAX_PATH_LENGTH} characters.")
    if "\x00" in path:
        raise ValueError(f"{field} must not contain NUL characters.")
    if "\\" in path:
        raise ValueError(f"{field} must use POSIX-style '/' separators.")
    if path.startswith("//") or re.match(r"^[A-Za-z]:", path) or "://" in path:
        raise ValueError(f"{field} must be a repository-relative path.")

    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(f"{field} must stay inside the repository.")
    if any(_is_git_metadata_component(component) for component in pure_path.parts):
        raise ValueError(f"{field} must not access repository metadata.")

    normalized = pure_path.as_posix()
    if normalized in {"", "./"}:
        return "."
    return normalized


def _is_git_metadata_component(component: str) -> bool:
    """Account for case-insensitive and Windows trailing-dot path aliases."""
    return component.rstrip(" .").casefold() == ".git"


def _resolve_repository_path(
    repository: CodingRepository,
    path: str,
    *,
    field: str = "path",
    must_exist: bool = True,
    expect_file: bool = False,
    expect_directory: bool = False,
) -> tuple[Path, str]:
    normalized = _normalize_relative_path(path, field)
    candidate = repository.path if normalized == "." else repository.path / normalized

    current = repository.path
    if current.is_symlink():
        raise ValueError("The repository root must not be a symbolic link.")
    for component in PurePosixPath(normalized).parts:
        if component == ".":
            continue
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{field} must not traverse symbolic links.")

    try:
        candidate.resolve(strict=False).relative_to(repository.path)
    except ValueError as exc:
        raise ValueError(f"{field} must stay inside the repository.") from exc

    if must_exist and not candidate.exists():
        raise ValueError(f"{field} does not exist in the repository: {normalized}")
    if must_exist and expect_file and not candidate.is_file():
        raise ValueError(f"{field} must identify a regular file.")
    if must_exist and expect_directory and not candidate.is_dir():
        raise ValueError(f"{field} must identify a directory.")
    if candidate.exists() and not (candidate.is_file() or candidate.is_dir()):
        raise ValueError(f"{field} must identify a regular file or directory.")
    return candidate, normalized


def _validate_glob(pattern: str | None, field: str = "glob") -> str | None:
    if pattern is None:
        return None
    if not isinstance(pattern, str) or not pattern or len(pattern) > _MAX_GLOB_LENGTH:
        raise ValueError(f"{field} must contain 1-{_MAX_GLOB_LENGTH} characters.")
    if "\x00" in pattern or "\\" in pattern:
        raise ValueError(f"{field} must be a POSIX-style glob without NUL characters.")
    if pattern.startswith(("/", "!")) or ".." in PurePosixPath(pattern).parts:
        raise ValueError(f"{field} must stay inside the repository.")
    if any(
        _is_git_metadata_component(component)
        for component in PurePosixPath(pattern).parts
    ):
        raise ValueError(f"{field} must not include repository metadata.")
    return pattern


def _matches_glob(path: str, pattern: str | None) -> bool:
    if pattern is None or pattern in {"*", "**", "**/*"}:
        return True
    return PurePosixPath(path).match(pattern) or fnmatch.fnmatchcase(path, pattern)


def _iter_regular_files(
    repository: CodingRepository,
    base_path: Path | None = None,
) -> Iterable[tuple[Path, str]]:
    base = base_path or repository.path
    for directory, directory_names, file_names in os.walk(base, followlinks=False):
        directory_path = Path(directory)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not _is_git_metadata_component(name)
            and not (directory_path / name).is_symlink()
        )
        for file_name in sorted(file_names):
            candidate = directory_path / file_name
            if candidate.is_symlink() or not candidate.is_file():
                continue
            relative = candidate.relative_to(repository.path).as_posix()
            if any(
                _is_git_metadata_component(component)
                for component in PurePosixPath(relative).parts
            ):
                continue
            yield candidate, relative


def _command_environment(repository: CodingRepository | None = None) -> dict[str, str]:
    home = (
        repository.path.parent / "git-home"
        if repository
        else Path(tempfile.gettempdir())
    )
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows fallback
            process.kill()
    except ProcessLookupError:
        return


def _run_process(
    arguments: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    input_text: str | None = None,
    timeout_seconds: int = _COMMAND_TIMEOUT_SECONDS,
    max_output_bytes: int = _MAX_COMMAND_OUTPUT,
) -> _ProcessResult:
    process = subprocess.Popen(
        arguments,
        cwd=cwd,
        env=environment,
        shell=False,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(
            None if input_text is None else input_text.encode("utf-8"),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        _terminate_process(process)
        process.communicate()
        raise _ProcessTimeout("The command exceeded its time limit.") from exc

    truncated = len(stdout) > max_output_bytes or len(stderr) > max_output_bytes
    stdout = stdout[:max_output_bytes]
    stderr = stderr[:max_output_bytes]
    return _ProcessResult(
        returncode=process.returncode,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        output_truncated=truncated,
    )


def _sanitize_output(text: str, repository: CodingRepository) -> str:
    return text.replace(str(repository.path), "<repository>")


def _git_process(
    repository: CodingRepository,
    arguments: list[str],
    *,
    input_text: str | None = None,
    timeout_seconds: int = _COMMAND_TIMEOUT_SECONDS,
    max_output_bytes: int = _MAX_COMMAND_OUTPUT,
) -> _ProcessResult:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required for repository history tools.")
    return _run_process(
        [
            git,
            "-C",
            str(repository.path),
            "--no-pager",
            "-c",
            "core.quotepath=false",
            *arguments,
        ],
        cwd=repository.path,
        environment=_command_environment(repository),
        input_text=input_text,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )


def _require_git_success(
    result: _ProcessResult,
    repository: CodingRepository,
    operation: str,
) -> str:
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        detail = _sanitize_output(detail, repository)[:500]
        raise ValueError(f"{operation} failed: {detail}")
    return result.stdout


def _resolve_revision(repository: CodingRepository, revision: str) -> str:
    normalized = revision.strip() if isinstance(revision, str) else ""
    if _REVISION_PATTERN.fullmatch(normalized):
        revision_expression = normalized
    elif _BRANCH_PATTERN.fullmatch(normalized) and all(
        component not in {"", ".", ".."}
        and not component.startswith(".")
        and not component.endswith((".", ".lock"))
        for component in normalized.split("/")
    ):
        revision_expression = f"refs/heads/{normalized}"
    else:
        raise ValueError(
            "revision must be HEAD, HEAD~N, a local branch, or a 7-40 character "
            "hexadecimal commit ID."
        )

    resolved_result = _git_process(
        repository,
        [
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{revision_expression}^{{commit}}",
        ],
    )
    if resolved_result.returncode != 0:
        raise ValueError(f"Unknown commit revision: {normalized}")
    resolved = resolved_result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", resolved):
        raise ValueError(f"Unable to resolve commit revision: {normalized}")

    ancestor = _git_process(
        repository,
        ["merge-base", "--is-ancestor", resolved, repository.base_commit],
    )
    if ancestor.returncode != 0:
        raise ValueError(
            "revision is not reachable from the pinned repository snapshot."
        )
    return resolved


def _bounded_text(text: str, maximum_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return text, False
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore"), True


def code_list_files(
    repo_id: str,
    path: str = ".",
    glob: str = "**/*",
    max_results: int = 100,
) -> dict[str, Any]:
    """List regular files in an allowlisted coding repository without following symlinks."""
    maximum = _validate_limit(max_results, "max_results", _MAX_LIST_RESULTS)
    file_glob = _validate_glob(glob)
    repository = get_coding_repository(repo_id)

    with get_coding_lock():
        base, normalized_path = _resolve_repository_path(
            repository, path, expect_directory=True
        )
        matches = []
        truncated = False
        for candidate, relative in _iter_regular_files(repository, base):
            if not _matches_glob(relative, file_glob):
                continue
            if len(matches) >= maximum:
                truncated = True
                break
            matches.append({"path": relative, "size_bytes": candidate.stat().st_size})

    return {
        "repo_id": repository.repo_id,
        "path": normalized_path,
        "glob": file_glob,
        "files": matches,
        "count": len(matches),
        "truncated": truncated,
        "source": _SOURCE,
    }


def code_read_file(
    repo_id: str,
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict[str, Any]:
    """Read a bounded UTF-8 line range from a regular repository file."""
    if (
        isinstance(start_line, bool)
        or not isinstance(start_line, int)
        or start_line < 1
    ):
        raise ValueError("start_line must be a positive integer.")
    if end_line is not None and (
        isinstance(end_line, bool)
        or not isinstance(end_line, int)
        or end_line < start_line
    ):
        raise ValueError(
            "end_line must be an integer greater than or equal to start_line."
        )
    if end_line is not None and end_line - start_line + 1 > _MAX_READ_LINES:
        raise ValueError(f"A single read may return at most {_MAX_READ_LINES} lines.")

    repository = get_coding_repository(repo_id)
    with get_coding_lock():
        candidate, normalized_path = _resolve_repository_path(
            repository, path, expect_file=True
        )
        file_size = candidate.stat().st_size
        if file_size > _MAX_READ_BYTES:
            raise ValueError(f"File exceeds the {_MAX_READ_BYTES}-byte read limit.")
        data = candidate.read_bytes()
        if b"\x00" in data:
            raise ValueError("Only UTF-8 text files can be read.")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Only UTF-8 text files can be read.") from exc

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)
    if total_lines and start_line > total_lines:
        raise ValueError(f"start_line exceeds the file length of {total_lines} lines.")
    requested_end = (
        end_line if end_line is not None else start_line + _MAX_READ_LINES - 1
    )
    actual_end = min(requested_end, total_lines) if total_lines else None
    content = "" if actual_end is None else "".join(lines[start_line - 1 : actual_end])
    return {
        "repo_id": repository.repo_id,
        "path": normalized_path,
        "content": content,
        "start_line": start_line,
        "end_line": actual_end,
        "total_lines": total_lines,
        "truncated": (
            end_line is None and actual_end is not None and actual_end < total_lines
        ),
        "source": _SOURCE,
    }


def code_search_text(
    repo_id: str,
    pattern: str,
    path_glob: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> dict[str, Any]:
    """Search repository text with ripgrep and return stable file/line/column matches."""
    if (
        not isinstance(pattern, str)
        or not pattern
        or len(pattern) > _MAX_PATTERN_LENGTH
    ):
        raise ValueError(f"pattern must contain 1-{_MAX_PATTERN_LENGTH} characters.")
    if "\x00" in pattern:
        raise ValueError("pattern must not contain NUL characters.")
    if not isinstance(case_sensitive, bool):
        raise ValueError("case_sensitive must be a boolean.")
    maximum = _validate_limit(max_results, "max_results", _MAX_SEARCH_RESULTS)
    file_glob = _validate_glob(path_glob, "path_glob")
    repository = get_coding_repository(repo_id)
    ripgrep = shutil.which("rg")
    if ripgrep is None:
        raise RuntimeError("ripgrep is required for code_search_text.")

    arguments = [
        ripgrep,
        "--json",
        "--no-config",
        "--no-follow",
        "--no-ignore-parent",
        "--sort=path",
        "--max-filesize=256K",
        "--glob=!.git/**",
        "--fixed-strings",
    ]
    if not case_sensitive:
        arguments.append("--ignore-case")
    if file_glob is not None:
        arguments.extend(["--glob", file_glob])
    arguments.extend(["--regexp", pattern, "."])

    with get_coding_lock():
        result = _run_process(
            arguments,
            cwd=repository.path,
            environment=_command_environment(repository),
            timeout_seconds=_COMMAND_TIMEOUT_SECONDS,
            max_output_bytes=4 * _MAX_COMMAND_OUTPUT,
        )
    if result.returncode not in {0, 1}:
        detail = _sanitize_output(result.stderr.strip(), repository)[:500]
        raise ValueError(f"ripgrep search failed: {detail or 'unknown error'}")
    if result.output_truncated:
        raise ValueError("ripgrep output exceeded the bounded search limit.")

    matches: list[dict[str, Any]] = []
    response_bytes = 2
    truncated = False
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ripgrep returned malformed JSON output.") from exc
        if event.get("type") != "match":
            continue
        data = event["data"]
        relative = data["path"]["text"].removeprefix("./")
        if any(
            _is_git_metadata_component(component)
            for component in PurePosixPath(relative).parts
        ):
            continue
        line_number = int(data["line_number"])
        all_columns = [
            int(submatch["start"]) + 1 for submatch in data.get("submatches", [])
        ]
        if not all_columns:
            continue
        if len(matches) >= maximum:
            truncated = True
            break

        columns = all_columns[:_MAX_SEARCH_COLUMNS]
        line_text = data["lines"]["text"].rstrip("\r\n")
        excerpt, excerpt_truncated = _bounded_text(line_text, _MAX_SEARCH_EXCERPT_BYTES)
        match = {
            "path": relative,
            "line": line_number,
            "column": columns[0],
            "columns": columns,
            "columns_truncated": len(columns) < len(all_columns),
            "text": excerpt,
            "text_truncated": excerpt_truncated,
        }
        match_size = len(json.dumps(match, ensure_ascii=True).encode("utf-8"))
        separator_size = 1 if matches else 0
        if response_bytes + separator_size + match_size > _MAX_SEARCH_RESPONSE_BYTES:
            truncated = True
            break
        matches.append(match)
        response_bytes += separator_size + match_size

    matches.sort(key=lambda item: (item["path"], item["line"], item["column"]))
    return {
        "repo_id": repository.repo_id,
        "pattern": pattern,
        "path_glob": file_glob,
        "case_sensitive": case_sensitive,
        "matches": matches,
        "count": len(matches),
        "truncated": truncated,
        "engine": "ripgrep-fixed-string",
        "source": _SOURCE,
    }


def git_log(
    repo_id: str,
    path: str | None = None,
    max_count: int = 20,
) -> dict[str, Any]:
    """Return bounded commit history reachable from the pinned repository snapshot."""
    maximum = _validate_limit(max_count, "max_count", 100)
    repository = get_coding_repository(repo_id)
    normalized_path = None
    path_arguments: list[str] = []
    if path is not None:
        _, normalized_path = _resolve_repository_path(
            repository, path, must_exist=False
        )
        path_arguments = ["--", normalized_path]

    record_format = "%H%x00%h%x00%an%x00%aI%x00%s%x1e"
    with get_coding_lock():
        result = _git_process(
            repository,
            [
                "log",
                f"--max-count={maximum}",
                f"--format={record_format}",
                repository.base_commit,
                *path_arguments,
            ],
        )
    output = _require_git_success(result, repository, "git_log")
    commits = []
    for record in output.split("\x1e"):
        fields = record.strip("\r\n").split("\x00") if record.strip("\r\n") else []
        if len(fields) != 5:
            continue
        sha, short_sha, author, authored_at, subject = fields
        commits.append(
            {
                "sha": sha,
                "short_sha": short_sha,
                "author": author,
                "authored_at": authored_at,
                "subject": subject,
            }
        )
    return {
        "repo_id": repository.repo_id,
        "path": normalized_path,
        "commits": commits,
        "count": len(commits),
        "source": _SOURCE,
    }


def _commit_changed_files(
    repository: CodingRepository,
    revision: str,
    path: str | None,
) -> list[str]:
    arguments = [
        "diff-tree",
        "--root",
        "--no-commit-id",
        "--name-only",
        "-r",
        revision,
    ]
    if path is not None:
        arguments.extend(["--", path])
    result = _git_process(repository, arguments)
    output = _require_git_success(result, repository, "git_show")
    return sorted(line for line in output.splitlines() if line)


def git_show(
    repo_id: str,
    revision: str,
    path: str | None = None,
) -> dict[str, Any]:
    """Show metadata and a bounded patch for one reachable commit."""
    repository = get_coding_repository(repo_id)
    normalized_path = None
    path_arguments: list[str] = []
    if path is not None:
        _, normalized_path = _resolve_repository_path(
            repository, path, must_exist=False
        )
        path_arguments = ["--", normalized_path]

    with get_coding_lock():
        resolved = _resolve_revision(repository, revision)
        metadata_result = _git_process(
            repository,
            ["show", "--no-patch", "--format=%H%x00%h%x00%an%x00%aI%x00%s", resolved],
        )
        metadata = _require_git_success(metadata_result, repository, "git_show").strip()
        fields = metadata.split("\x00")
        if len(fields) != 5:
            raise RuntimeError("Git returned malformed commit metadata.")
        patch_result = _git_process(
            repository,
            [
                "show",
                "--format=",
                "--patch",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                resolved,
                *path_arguments,
            ],
            max_output_bytes=_MAX_COMMAND_OUTPUT,
        )
        patch = _require_git_success(patch_result, repository, "git_show")
        changed_files = _commit_changed_files(repository, resolved, normalized_path)

    bounded_patch, truncated = _bounded_text(patch, _MAX_DIFF_OUTPUT)
    sha, short_sha, author, authored_at, subject = fields
    return {
        "repo_id": repository.repo_id,
        "requested_revision": revision,
        "sha": sha,
        "short_sha": short_sha,
        "author": author,
        "authored_at": authored_at,
        "subject": subject,
        "path": normalized_path,
        "changed_files": changed_files,
        "patch": bounded_patch,
        "truncated": truncated or patch_result.output_truncated,
        "source": _SOURCE,
    }


def _git_diff_arguments(
    base: str,
    head: str,
    path: str | None,
    context_lines: int,
    *,
    names_only: bool,
) -> list[str]:
    arguments = [
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
    ]
    arguments.append("--name-only" if names_only else f"--unified={context_lines}")
    arguments.append(base)
    if head != "WORKTREE":
        arguments.append(head)
    if path is not None:
        arguments.extend(["--", path])
    return arguments


def git_diff(
    repo_id: str,
    base: str,
    head: str,
    path: str | None = None,
    context_lines: int = 3,
) -> dict[str, Any]:
    """Compare two reachable commits, or compare a commit with the current worktree."""
    if isinstance(context_lines, bool) or not isinstance(context_lines, int):
        raise ValueError("context_lines must be an integer.")
    if context_lines < 0 or context_lines > 20:
        raise ValueError("context_lines must be between 0 and 20.")
    repository = get_coding_repository(repo_id)
    normalized_path = None
    if path is not None:
        _, normalized_path = _resolve_repository_path(
            repository, path, must_exist=False
        )

    with get_coding_lock():
        resolved_base = _resolve_revision(repository, base)
        if not isinstance(head, str):
            raise ValueError("head must be a commit revision or WORKTREE.")
        resolved_head = (
            "WORKTREE"
            if head.strip().upper() == "WORKTREE"
            else _resolve_revision(repository, head)
        )
        patch_result = _git_process(
            repository,
            _git_diff_arguments(
                resolved_base,
                resolved_head,
                normalized_path,
                context_lines,
                names_only=False,
            ),
            max_output_bytes=_MAX_COMMAND_OUTPUT,
        )
        patch = _require_git_success(patch_result, repository, "git_diff")
        names_result = _git_process(
            repository,
            _git_diff_arguments(
                resolved_base,
                resolved_head,
                normalized_path,
                context_lines,
                names_only=True,
            ),
        )
        names = _require_git_success(names_result, repository, "git_diff")

    bounded_patch, truncated = _bounded_text(patch, _MAX_DIFF_OUTPUT)
    return {
        "repo_id": repository.repo_id,
        "base": resolved_base,
        "head": resolved_head,
        "path": normalized_path,
        "context_lines": context_lines,
        "changed_files": sorted(line for line in names.splitlines() if line),
        "patch": bounded_patch,
        "truncated": truncated or patch_result.output_truncated,
        "source": _SOURCE,
    }


def git_status(
    repo_id: str,
    path: str | None = None,
    include_untracked: bool = True,
    max_entries: int = 200,
) -> dict[str, Any]:
    """Return a bounded summary of the repository worktree and index state."""
    if not isinstance(include_untracked, bool):
        raise ValueError("include_untracked must be a boolean.")
    maximum = _validate_limit(max_entries, "max_entries", _MAX_STATUS_ENTRIES)
    repository = get_coding_repository(repo_id)
    normalized_path = None
    path_arguments: list[str] = []
    if path is not None:
        _, normalized_path = _resolve_repository_path(
            repository, path, must_exist=False
        )
        path_arguments = ["--", normalized_path]

    untracked_mode = "all" if include_untracked else "no"
    with get_coding_lock():
        head_sha = _resolve_revision(repository, "HEAD")
        result = _git_process(
            repository,
            [
                "status",
                "--porcelain=v1",
                "--branch",
                f"--untracked-files={untracked_mode}",
                *path_arguments,
            ],
        )
    output = _require_git_success(result, repository, "git_status")
    if result.output_truncated:
        raise ValueError("git_status output exceeded its bounded output limit.")

    lines = output.splitlines()
    branch = None
    detached = False
    if lines and lines[0].startswith("## "):
        branch_text = lines.pop(0)[3:]
        detached = branch_text.startswith("HEAD (")
        if not detached:
            branch = branch_text.split("...", 1)[0].split(" ", 1)[0]

    all_entries: list[dict[str, Any]] = []
    conflict_codes = {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}
    for line in lines:
        if len(line) < 4:
            raise RuntimeError("Git returned malformed status output.")
        code = line[:2]
        status_path = line[3:]
        index_code, worktree_code = code
        untracked = code == "??"
        conflicted = code in conflict_codes or "U" in code
        staged = not untracked and index_code != " "
        unstaged = not untracked and worktree_code != " "
        if conflicted:
            category = "conflicted"
        elif untracked:
            category = "untracked"
        elif staged and unstaged:
            category = "staged_and_unstaged"
        elif staged:
            category = "staged"
        else:
            category = "unstaged"
        all_entries.append(
            {
                "path": status_path,
                "index_status": None if index_code == " " else index_code,
                "worktree_status": None if worktree_code == " " else worktree_code,
                "category": category,
            }
        )

    entries = all_entries[:maximum]
    return {
        "repo_id": repository.repo_id,
        "path": normalized_path,
        "branch": branch,
        "detached": detached,
        "head_sha": head_sha,
        "clean": not all_entries,
        "include_untracked": include_untracked,
        "entries": entries,
        "count": len(entries),
        "total_count": len(all_entries),
        "summary": {
            "staged": sum(
                1 for entry in all_entries if entry["index_status"] not in {None, "?"}
            ),
            "unstaged": sum(
                1
                for entry in all_entries
                if entry["worktree_status"] not in {None, "?"}
            ),
            "untracked": sum(
                1 for entry in all_entries if entry["category"] == "untracked"
            ),
            "conflicted": sum(
                1 for entry in all_entries if entry["category"] == "conflicted"
            ),
        },
        "truncated": len(all_entries) > maximum,
        "source": _SOURCE,
    }
