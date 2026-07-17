# Benchmark Datasets

This folder contains JSON datasets for evaluating single-tool routing. Each record asks a model/router to choose one tool from an available tool list and provide the expected arguments for that tool.

## Current Layout

| Path | Records | Domains | Purpose |
| --- | ---: | --- | --- |
| `tool_routing_smoke.json` | 8 | coding, enterprise automation, finance, mathematics | Small smoke test across the original four domains. |
| `tool_routing_controlled.json` | 40 | coding, enterprise automation, finance, mathematics | Older controlled synthetic benchmark across the original four domains. |
| `tool_routing_phase2_seed.json` | 16 | coding, enterprise automation, finance, mathematics | Seed set for phase-2 routing work. |
| `math/tool_routing_math_controlled.json` | 51 | mathematics | Controlled math routing set. |
| `math/tool_routing_math_public_derived.json` | 77 | mathematics | Public MATH-derived math routing set. |
| `enterprise/tool_routing_enterprise_v1.json` | 35 | enterprise automation | First controlled enterprise fixture suite. |
| `enterprise/tool_routing_enterprise_v2_controlled.json` | 48 | enterprise automation | Controlled retail-style enterprise suite. |
| `enterprise/tool_routing_enterprise_public_adapted.json` | 24 | enterprise automation | Public tau3 Retail-adapted enterprise suite. |
| `coding/tool_routing_coding_smoke.json` | 7 | coding | Small direct coding tool smoke set. |
| `coding/tool_routing_coding_controlled.json` | 35 | coding | Controlled coding routing set. |
| `coding/tool_routing_coding_upstream_inspired.json` | 28 | coding | Generated coding prompts adapted from upstream tool usage patterns. |
| `coding/tool_routing_coding_codesearchnet_public_derived.json` | 15 | coding | Public CodeSearchNet-derived coding search set. |
| `finance/tool_routing_finance_smoke.json` | 10 | finance | Small direct finance tool smoke set. |
| `finance/tool_routing_finance_controlled.json` | 50 | finance | Controlled finance routing set. |
| `finance/tool_routing_finance_upstream_inspired.json` | 40 | finance | Generated finance prompts adapted from upstream tool usage patterns. |
| `finance/tool_routing_finance_public_derived.json` | 15 | finance | Public FinQA-derived finance table set. |
| `finance/tool_routing_finance_tatqa_public_derived.json` | 15 | finance | Public TAT-QA-derived finance table set. |

The root-level files are legacy mixed-domain benchmarks. The domain folders are the preferred place for new datasets.

## Standard Record Schema

New datasets should use the same core fields:

- `id`: stable unique identifier.
- `domain`: broad domain, such as `mathematics` or `enterprise_automation`.
- `task_type`: currently `single_tool_routing`.
- `difficulty`: simple level label for analysis.
- `source`: how the example was created, such as `controlled_synthetic`, `public_math_derived`, or `public_adapted`.
- `query`: natural-language user request.
- `available_tools`: tools shown to the router for that example.
- `expected_tool`: correct tool name.
- `expected_args`: correct JSON arguments for the selected tool.
- `expected_answer`: expected tool output when known, or `null` if the output is stateful or not fixed.
- `perturbation_type`: what kind of routing challenge the example tests.
- `notes`: short human-readable provenance or rationale.

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
tool_routing_<domain>_<source>.json
```

Examples:

```text
tool_routing_math_controlled.json
tool_routing_math_public_derived.json
tool_routing_enterprise_controlled_v2.json
tool_routing_enterprise_public_adapted.json
```

Use `controlled` for examples written specifically to target a tool and argument schema. Use `public_derived` or `public_adapted` when the query came from, or was adapted from, a public dataset.

It is also fine to use a clearer suffix when it identifies the purpose or source more precisely, such as `smoke`, `upstream_inspired`, `codesearchnet_public_derived`, or `tatqa_public_derived`. Prefer clarity over forcing every domain into one file.

Avoid using `v1`, `v2`, or phase names as random subdivisions. If a version is needed, it should mean a real tool-suite or schema version, and the README in that domain folder should explain what changed.

## Four-Domain Baseline

The original four domains were:

- `mathematics`
- `enterprise_automation`
- `finance`
- `coding`

The older root-level benchmarks cover all four domains in one file. The newer direction is to keep each domain in its own folder, using the same core schema and clear source-oriented filenames across domains.

