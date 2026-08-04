# LayerMCP Agent Guidance

This repository's implementation is authoritative for executable semantics.
Inspect the current evaluator, MCP server, routers, benchmark builders, and
tests before proposing schemas, defaults, tool contracts, or reporting rules.
Do not infer implementation behavior from old results, prose, branch names, or
prior agent summaries when the code can answer the question directly.

## Evaluator and benchmark invariants

- Obtain the effective `benchmark_mode` default from
  `evaluation.evaluate.DEFAULT_BENCHMARK_MODE`.
- Obtain the effective `workflow_execution_mode` default from
  `evaluation.evaluate.DEFAULT_WORKFLOW_EXECUTION_MODE`.
- Inventory and reporting utilities must distinguish fields literally present
  in raw JSON from defaults applied by the evaluator. Label both clearly when
  both are reported.
- JSON files under `benchmark/archive/` and `benchmark/**/fixtures/` are not
  active runnable benchmarks.
- Empty or provenance-only placeholders must be explicitly documented or
  allowlisted and must not be counted as runnable active benchmarks. Do not
  assume that an arbitrary empty `[]` benchmark is an intentional placeholder.
- Aggregate mixed-domain or mixed-task files per row. Never add an entire file's
  counts to every domain or task type found in that file.
- Benchmark classification may use benchmark-relative paths or filenames, but
  never absolute checkout directory names.
- Clearly label replay/offline, public/source-derived, and controlled datasets.
  Do not silently combine these categories.
- Report single-step, multi-step, replay, and public/source-derived results
  separately when their semantics or evidence differ.
- Do not create benchmark formats that the current evaluator cannot load and
  execute unless the work explicitly includes the required evaluator support.

## Scope and documentation

Keep changes focused and preserve unrelated work. Change documentation only
when public behavior, benchmark coverage, validation procedure, or durable
project guidance actually changes. Do not update documentation merely to make
a patch appear more complete.

## Validation

Run validation proportional to the change. Targeted tests for every touched
area are required. The standard checks are:

```bash
python -m compileall analysis benchmark evaluation mcp_server tests
git diff --check
```

When reliable in the current environment, also run full discovery:

```bash
python -m unittest discover -s tests
```

Some HPC and local environments may hang at MCP subprocess integration points.
Report all results honestly. If full discovery or unrelated tests fail, skip,
or hang, identify the exact command and observed condition; do not hide,
relabel, or silently omit it.
