# Benchmark Datasets

This folder contains JSON datasets for evaluating single-tool routing and
teacher-forced multi-step routing. Each single-step record asks a model/router
to choose one tool from an available tool list and provide the expected
arguments for that tool. Multi-step records contain an ordered sequence of
those routing decisions.

Generate a current inventory of active Math, Enterprise, Coding, and Finance
benchmark JSON files with:

```bash
python -m analysis.benchmark_inventory
python -m analysis.benchmark_inventory --json
python -m analysis.benchmark_inventory \
  --markdown-out /tmp/benchmark-inventory.md \
  --json-out /tmp/benchmark-inventory.json
```

The inventory excludes `benchmark/archive/`, domain fixture directories, and
cache files. Empty provenance-only JSON placeholders are listed separately and
excluded from runnable active totals. The utility reads benchmark data but does
not modify it.

Generate the reporting-only minimal scorecard for completed single-step run
directories with:

```bash
python -m analysis.minimal_scorecard RUN_DIR [RUN_DIR ...] --output SCORECARD.md
```

The report treats Final Outcome Accuracy as the primary success metric and Tool
Selection Accuracy as the second headline metric. Exact Reference Argument
Match is a secondary diagnostic against one reference call; it is not SVCA.
Final Outcome Accuracy uses only rows with a recorded outcome score as its
denominator, and every headline table displays `scored/total` Final Outcome
Coverage alongside it.
Valid Arguments / Schema-Valid Tool Call remains unavailable until full raw JSON
Schema validation is implemented. Reports also list the final-outcome matcher
names observed separately for each model, including valid PR #29 mixtures of
`finance_query_table_rows_v1` and `recursive_json_subset_v1`.
Grounded and offline/replay benchmark modes are always reported in separate
metric rows. A missing `benchmark_mode` uses
`evaluation.evaluate.DEFAULT_BENCHMARK_MODE`, but the report visibly marks its
mode source as `defaulted (missing)`. An explicitly null mode also uses the
evaluator default and is marked `defaulted (null)`; neither is pooled with an
explicit mode value.

Every loaded summary must provide a nonempty tool-registry fingerprint,
fingerprint version, tool count, and tool pool. Legacy artifacts missing any of
that metadata cannot be pooled into a verified scorecard.

## Current Layout

| Path | Records | Domains | Purpose |
| --- | ---: | --- | --- |
| `archive/root/tool_routing_smoke.json` | 8 | coding, enterprise automation, finance, mathematics | Small smoke test across the original four domains. |
| `archive/root/tool_routing_controlled.json` | 40 | coding, enterprise automation, finance, mathematics | Older controlled synthetic benchmark across the original four domains. |
| `archive/root/tool_routing_phase2_seed.json` | 16 | coding, enterprise automation, finance, mathematics | Seed set for phase-2 routing work. |
| `math/math_public.json` | 400 | mathematics | Expanded public-derived DeepMind Mathematics and GSM8K routing set. |
| `math/math_public_math_dataset.json` | 77 | mathematics | Executable public MATH-derived math routing set. |
| `math/math_controlled.json` | 51 | mathematics | Controlled math routing set. |
| `math/math_multistep_controlled.json` | 50 | mathematics | Controlled dependent math workflows with 105 executable steps. |
| `enterprise/enterprise_controlled.json` | 35 | enterprise automation | First controlled enterprise fixture suite. |
| `enterprise/enterprise_tau2_single_step.json` | 293 | enterprise automation | Executable tau2 retail single-step/adapted diagnostic benchmark. |
| `enterprise/enterprise_public_workflows.json` | 69 | enterprise automation | Teacher-forced routing over tau2 retail reference action trajectories with step-level source grounding. |
| `archive/enterprise/enterprise_v2_controlled_legacy.json` | 48 | enterprise automation | Controlled retail-style enterprise suite. |
| `archive/enterprise/enterprise_public_adapted_legacy.json` | 24 | enterprise automation | Public tau3 Retail-adapted enterprise suite. |
| `coding/coding_smoke.json` | 7 | coding | Small direct coding tool smoke set. |
| `coding/coding_controlled.json` | 35 | coding | Controlled coding routing set. |
| `coding/coding_upstream_inspired.json` | 28 | coding | Generated coding prompts adapted from upstream tool usage patterns. |
| `coding/coding_codesearchnet_public_derived.json` | 97 | coding | Public CodeSearchNet-derived coding search set. |
| `coding/coding_conala_public_derived.json` | 133 | coding | Public CoNaLa-derived curated-intent search set. |
| `coding/coding_sweagent_multistep.json` | 5 | coding | Source-faithful SWE-agent workflows executed against bounded repository fixtures. |
| `coding/coding_nebius_sweagent_replay_multistep.json` | 33 | coding | Offline replay of selected Nebius SWE-agent trajectories containing at most five calls. |
| `coding/coding_nebius_swerebench_openhands_replay_multistep.json` | 0 | coding | Provenance-only zero-result placeholder; excluded from benchmark runs and results. |
| `finance/finance_smoke.json` | 10 | finance | Small direct finance tool smoke set. |
| `finance/finance_controlled.json` | 50 | finance | Controlled finance routing set. |
| `finance/finance_upstream_inspired.json` | 40 | finance | Generated finance prompts adapted from upstream tool usage patterns. |
| `finance/finance_tatqa_public_derived.json` | 15 | finance | Public TAT-QA-derived finance table set. |
| `finance/finance_convfinqa_multistep.json` | 10 | finance | Paper-authored ConvFinQA conversations adapted to grounded tool calls. |
| `finance/finance_finqa_test_single.json` | 642 | finance | One-operation FinQA gold programs adapted to grounded tool calls. |
| `finance/finance_finqa_test_multistep.json` | 490 | finance | Two- to five-operation FinQA gold programs adapted to grounded tool calls. |
| `finance/finance_finretrieval_replay_multistep.json` | 485 | finance | Offline replay of selected correct FinRetrieval model trajectories containing at most five calls. |

The files under `archive/root/` are legacy mixed-domain benchmarks. The domain
folders are the preferred place for active datasets.

## Standard Record Schema

New datasets should use the same core fields:

- `id`: stable unique identifier.
- `domain`: broad domain, such as `mathematics` or `enterprise_automation`.
- `task_type`: `single_tool_routing` or `multi_step_tool_routing`.
- `benchmark_mode`: `grounded_tool_execution` by default, or
  `offline_trace_replay` for coordinate-keyed trajectory replay.
- `difficulty`: simple level label for analysis.
- `source`: how the example was created, such as `controlled_synthetic`, `public_math_derived`, or `public_adapted`.
- `query`: natural-language user request.
- `expected_tool`: correct tool name for a single-step row.
- `expected_args`: correct JSON arguments for a single-step row.
- `expected_answer`: expected single-step tool output when known, or `null` if
  the output is stateful or not fixed.
- `expected_steps`: ordered gold step labels for a multi-step row, including
  each step's tool, arguments, answer, dependencies, and visible grounding.
- `perturbation_type`: what kind of routing challenge the example tests.
- `notes`: short human-readable provenance or rationale.

The evaluator exposes every row to the full live tool registry returned by MCP
`list_tools()`. Benchmark rows do not define their own candidate tool menus.

The replay-tool expansion changed the full registry from 51 tools to 60 by
adding five coding replay tools and four finance replay tools. Any model result
produced against the former 51-tool registry is stale and not directly
comparable with a current result. All full-registry evaluations must be rerun
against the 60-tool registry after this change.

Every result record and summary stores the sorted tool names and a versioned
SHA-256 fingerprint over the tool names, input schemas, and descriptions.
Results are directly comparable only when their registry fingerprints match.

## Evaluation and Reporting Protocols

Multi-step evaluation uses `teacher_forced_step_routing_v1`. For every expected
step, the evaluator constructs a prompt from the overall task, the gold
current-step instruction and grounding context, and bounded gold prior-step
context. A prediction does not determine the instruction or context supplied
to the next step. Sequence accuracy therefore measures controlled ordered
routing under teacher forcing; it must not be reported as autonomous planning,
autonomous decomposition, or end-to-end task completion.

Every benchmark result must retain and be grouped by one of these modes:

- `grounded_tool_execution`: controlled or source-faithful queries targeting
  bounded fixtures or allowlisted repositories.
- `offline_trace_replay`: reproduction of recorded offline tool-call
  coordinates and outputs.

Report these modes separately. In particular, Nebius SWE-agent coding traces
and FinRetrieval finance traces are `offline_trace_replay`; their scores must
not be aggregated with live/source-faithful `grounded_tool_execution` results.
Summaries provide separate accuracy sections for every benchmark mode present.
The empty OpenHands JSON file is retained only to document the deterministic
five-call selection result. It contributes zero queries and must not be passed
to evaluation or included in result summaries.

Public or adapted datasets should also include provenance fields when available:

- `source_dataset`
- `source_domain`
- `source_task_id`
- `source_row_index`
- `source_category`
- `source_level`
- `source_action`
- `provenance_type`

Not every public source has every provenance field. Use the fields that clearly apply.

## Naming Standard

Prefer this filename pattern for new benchmark files:

```text
benchmark/<domain>/<domain>_<source_or_purpose>.json
```

Examples:

```text
benchmark/math/math_public.json
benchmark/math/math_public_math_dataset.json
benchmark/math/math_controlled.json
benchmark/enterprise/enterprise_controlled.json
benchmark/enterprise/enterprise_tau2_single_step.json
benchmark/coding/coding_sweagent_multistep.json
benchmark/finance/finance_finqa_test_multistep.json
benchmark/archive/enterprise/enterprise_public_adapted_legacy.json
```

All active domain benchmark files follow this domain-local convention. Archived
legacy files may retain their historical names.

Use `controlled` for examples written specifically to target a tool and argument schema. Use `public_derived` or `public_adapted` when the query came from, or was adapted from, a public dataset.

It is also fine to use a clearer suffix when it identifies the purpose or source more precisely, such as `smoke`, `upstream_inspired`, `codesearchnet_public_derived`, or `tatqa_public_derived`. Prefer clarity over forcing every domain into one file.

Avoid using `v1`, `v2`, or phase names as random subdivisions. If a version is needed, it should mean a real tool-suite or schema version, and the README in that domain folder should explain what changed.

## Four-Domain Baseline

The original four domains were:

- `mathematics`
- `enterprise_automation`
- `finance`
- `coding`

The benchmarks under `archive/root/` cover all four domains in one file. The
newer direction is to keep each domain in its own folder, using the same core
schema and clear source-oriented filenames across domains.
