"""Command-line entry point: `mcpbench run ...`."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .client import LLMClient
from .runner import run_suite
from .schema import Suite, Task

SUITE_DIR = Path(__file__).resolve().parent.parent / "suites"


def _tasks_from_json(data) -> list[Task]:
    items = data if isinstance(data, list) else data.get("tasks", [])
    return [Task(**t) for t in items]


def load_suite(name_or_path: str) -> Suite:
    if name_or_path == "all":
        tasks: list[Task] = []
        for path in sorted(SUITE_DIR.glob("*.json")):
            tasks.extend(_tasks_from_json(json.loads(path.read_text())))
        return Suite(name="all", tasks=tasks)

    path = Path(name_or_path)
    if not path.exists():
        path = SUITE_DIR / f"{name_or_path}.json"
    if not path.exists():
        raise SystemExit(f"suite not found: {name_or_path} (looked in {SUITE_DIR})")
    return Suite(name=path.stem, tasks=_tasks_from_json(json.loads(path.read_text())))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="mcpbench", description="MCP tool-selection benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a suite against an endpoint and record responses")
    run.add_argument("--endpoint", default="http://localhost:8080", help="OpenAI-compatible base URL")
    run.add_argument("--model", default="local", help="model name sent in the request")
    run.add_argument("--suite", default="all", help="suite name, path, or 'all'")
    run.add_argument("--output", default=None, help="report JSON path (default: results/<suite>_<ts>.json)")
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--no-think", action="store_true", help="disable Qwen3 thinking mode")
    run.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    run.add_argument("--quiet", action="store_true", help="suppress per-task lines")

    args = parser.parse_args(argv)
    if args.cmd != "run":
        parser.error("unknown command")

    suite = load_suite(args.suite)
    client = LLMClient(args.endpoint, model=args.model)
    n = min(len(suite.tasks), args.limit) if args.limit else len(suite.tasks)
    print(f"Running suite '{suite.name}' ({n} tasks) against {args.endpoint}  [no_think={args.no_think}]\n")

    results = run_suite(
        client,
        suite,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        no_think=args.no_think,
        limit=args.limit,
        verbose=not args.quiet,
    )
    client.close()

    errors = sum(1 for r in results if r.error)
    print(f"\nRan {len(results)} tasks ({errors} errors).")

    out = Path(args.output or f"results/{suite.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "endpoint": args.endpoint,
        "suite": suite.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "no_think": args.no_think,
            "limit": args.limit,
        },
        "results": [r.model_dump() for r in results],
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
