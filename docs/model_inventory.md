# Baseline model inventory

This is the model identity and runtime inventory for the LayerMCP frozen
baseline. It describes what the repository and its saved run metadata establish;
it does not infer an upstream checkpoint revision from a local directory name.

## Scope and reproducibility

The reported baseline results come from saved, validated runs at frozen commit
`4abe8ae6360db16486dcb3c5a92b452042ba7de7`. Every inspected saved
`run_metadata.json` records this commit, the model selector, the expected
upstream model name, a local checkpoint path, prompt and reasoning metadata,
tool-registry metadata, and generation metadata.

The archive does **not** record an upstream Hugging Face revision/commit or
per-shard digest for any checkpoint. Consequently, a local `checkpoint` path
in a saved run is evidence of the path used by that run, not a cryptographic
identity for the upstream weights. Model metadata below must therefore be read
alongside the corresponding saved `run_metadata.json` and preserved launcher.

The primary-source links below identify the publisher's released model where
the repository identifier can be established. They do not establish that the
archive used the current `main` revision of that upstream repository.

## At a glance

| Report display name | Repository-recorded upstream identifier | Published parameter count | Baseline condition |
| --- | --- | --- | --- |
| Phi-4 direct | `microsoft/phi-4` | 14B dense | direct |
| Llama 3.1 8B direct | `meta-llama/Llama-3.1-8B-Instruct` | 8B | direct |
| Qwen 3.6 direct | `Qwen/Qwen3.6` | not established from a specific upstream checkpoint | native thinking disabled |
| Qwen 3.6 native reasoning | `Qwen/Qwen3.6` | not established from a specific upstream checkpoint | native thinking enabled |
| Gemma 4 direct | `google/gemma-4-26b-a4b-it` | 25.2B total; 3.8B active | native thinking disabled |
| Gemma 4 native reasoning | `google/gemma-4-26b-a4b-it` | 25.2B total; 3.8B active | native thinking enabled |
| GPT-OSS 20B Harmony low reasoning | `openai/gpt-oss-20b` | 21B total; 3.6B active | Harmony, low reasoning effort |

The published counts in this table are the publishers' terminology. In
particular, the GPT-OSS public repository name contains `20b`, while its model
card reports 21B total parameters; Gemma's `26B-A4B` name is rounded, while
its model card reports 25.2B total and 3.8B active parameters.

## Common baseline mechanics

The single-step and multi-step Slurm launchers
([`scripts/slurm/run_single_step.sbatch`](../scripts/slurm/run_single_step.sbatch)
and [`scripts/slurm/run_multi_step.sbatch`](../scripts/slurm/run_multi_step.sbatch))
select a local checkpoint, set a template/condition label, and write
`run_metadata.json` before evaluation. Their `generation` object records
`temperature: 0.0`, `max_tokens`, the effective generated-token limit, and
`seed: "not_applicable_deterministic_greedy"`. The same condition,
reasoning-method, and effective-limit fields are checked in saved summaries
and artifacts.

The evaluated local routers call their token generators with `temperature=0.0`.
The generation limit is 128 tokens for Phi-4 and Llama, and 4096 tokens for
Qwen, Gemma, and GPT-OSS. These are LayerMCP benchmark settings, not claims
about publisher-recommended decoding parameters. The saved run metadata, not
this page, is the run-specific source of truth.

`build_tool_call_prompt` in
[`models/routers/structured_tool_call.py`](../models/routers/structured_tool_call.py)
defines the portable structured-call fallback: one JSON object with
`name` and `arguments`, alongside a JSON-serialized MCP tool catalog and input
schemas. `parse_tool_call` distinguishes native calls, recognized structured
JSON, Qwen's native function tags, and GPT-OSS Harmony calls.

## Phi-4 direct

- **Report display name:** Phi-4 direct.
- **Upstream model and published size:** [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4), whose Microsoft model card calls it a 14B-parameter dense decoder-only Transformer.
- **Local convention:** `checkpoints/phi-4/`; override with `LAYERMCP_PHI4_CHECKPOINT`.
- **LayerMCP runtime:** [`models/routers/phi4_local_router.py`](../models/routers/phi4_local_router.py), using the local PyTorch implementation in [`models/architectures/phi4_pytorch/`](../models/architectures/phi4_pytorch/).
- **Condition:** direct only. The launchers reject a Phi-4 `reasoning` condition and save `reasoning_method: "none"`.
- **Tool-call/chat protocol:** the router builds the repository structured-tool prompt and encodes it with the Phi tokenizer chat template when available (otherwise its explicit Phi fallback chat markers). The parser expects a complete structured call such as `{"name":"<tool>","arguments":{...}}`.
- **Generation record:** launcher template ID `tool_name_only_v1`, 128-token limit, and the common generation fields above. The ID is a launcher metadata label; the runtime prompt itself asks for a structured JSON call, not merely a bare tool name.

## Llama 3.1 8B direct

- **Report display name:** Llama 3.1 8B direct.
- **Upstream model and published size:** [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Meta's model card lists the text-only Llama 3.1 model at 8B parameters.
- **Local convention:** `checkpoints/llama-3.1-8b-instruct/`; override with `LAYERMCP_LLAMA31_8B_CHECKPOINT`.
- **LayerMCP runtime:** [`models/routers/llama31_8b_local_router.py`](../models/routers/llama31_8b_local_router.py), using [`models/architectures/llama31_8b_pytorch/`](../models/architectures/llama31_8b_pytorch/).
- **Condition:** direct only. The launchers reject `reasoning` and save `reasoning_method: "none"`.
- **Tool-call/chat protocol:** the runtime supplies OpenAI-style function descriptors (`type: "function"`, name, description, and JSON Schema) to the tokenizer's native chat-template path. It falls back to the repository structured-tool prompt if that native path is unavailable, then uses the common structured-call parser.
- **Generation record:** template ID `structured_tool_call_v1`, 128-token limit, and the common generation fields above.

## Qwen 3.6: direct and native reasoning

- **Report display names:** Qwen 3.6 direct; Qwen 3.6 native reasoning.
- **Repository-recorded identifier:** `Qwen/Qwen3.6` in the routers, launchers, and saved metadata.
- **Publisher source and parameter count:** the repository and saved metadata do not identify a specific public Qwen 3.6 size/revision. `Qwen/Qwen3.6` does not currently resolve here to a specific publisher model card, while the publisher's [Qwen3.6 collection](https://huggingface.co/collections/Qwen/qwen36) includes multiple releases. The exact upstream checkpoint and published parameter count for this baseline are therefore **not established**. Do not substitute a particular Qwen3.6 model card or its parameter count for the baseline without an archived checkpoint revision or weight fingerprint.
- **Local convention:** `checkpoints/qwen-3.6/`; override with `LAYERMCP_QWEN36_CHECKPOINT`.
- **LayerMCP runtime:** [`models/routers/qwen36_local_router.py`](../models/routers/qwen36_local_router.py), using [`models/architectures/qwen36_pytorch/`](../models/architectures/qwen36_pytorch/).
- **Condition:** both rows use the same local checkpoint convention and router. Direct passes `enable_thinking=False` and records `reasoning_method: "native_disabled"`; native reasoning passes `enable_thinking=True` and records `reasoning_method: "native_enabled"`. They are runtime configurations, not separately documented base-model checkpoints.
- **Tool-call/chat protocol:** the runtime passes native function descriptors to the Qwen chat template. Its parser recognizes Qwen native `<function=...><parameter=...>` calls as well as a complete structured JSON call. The launcher records template ID `tool_name_only_v1`; as with Phi-4, that label should not be read as a claim that the runtime only emits a bare name.
- **Generation record:** 4096-token limit and the common generation fields above.

## Gemma 4: direct and native reasoning

- **Report display names:** Gemma 4 direct; Gemma 4 native reasoning.
- **Upstream model and published size:** repository identifier `google/gemma-4-26b-a4b-it`, corresponding in case to the official [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) model card. The card reports 25.2B total parameters and 3.8B active parameters for the 26B A4B MoE model.
- **Local convention:** `checkpoints/gemma-4/`; override with `LAYERMCP_GEMMA4_CHECKPOINT`.
- **LayerMCP runtime:** [`models/routers/gemma4_local_router.py`](../models/routers/gemma4_local_router.py), using [`models/architectures/gemma4_pytorch/`](../models/architectures/gemma4_pytorch/).
- **Condition:** both rows are runtime configurations of the same local checkpoint convention. Direct passes `enable_thinking=False` and saves `reasoning_method: "native_disabled"`; native reasoning passes `enable_thinking=True` and saves `reasoning_method: "native_enabled"`.
- **Tool-call/chat protocol:** the router puts the repository structured-tool JSON prompt into Gemma's chat template and selects its `enable_thinking` option. It uses the common structured-call parser for the generated call. The launcher records template ID `tool_name_only_v1`; the generated-call contract remains structured JSON with `name` and `arguments`.
- **Generation record:** 4096-token limit and the common generation fields above.

## GPT-OSS 20B Harmony low reasoning

- **Report display name:** GPT-OSS 20B Harmony low reasoning.
- **Upstream model and published size:** [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b). OpenAI's model card reports 21B total parameters and 3.6B active parameters.
- **Local convention:** `checkpoints/gpt-oss-20b/original/`; override with `LAYERMCP_GPT_OSS_CHECKPOINT`.
- **LayerMCP runtime:** [`models/routers/gpt_oss_local_router.py`](../models/routers/gpt_oss_local_router.py), using [`models/architectures/gpt_oss_pytorch/`](../models/architectures/gpt_oss_pytorch/).
- **Condition:** the router and launchers accept only `reasoning_mode=reasoning` with `reasoning_effort=low`; saved metadata records `reasoning_method: "harmony"`. This is not a direct/no-reasoning comparison row.
- **Tool-call/chat protocol:** the runtime renders the model's Harmony tool prompt with function schemas and Low reasoning effort. The benchmark parser accepts exactly one complete Harmony commentary-channel `functions.<name>` call with a JSON object and a terminating `<|call|>` token. This matches the upstream card's requirement to use the Harmony format.
- **Generation record:** template ID `harmony_structured_context_sql_v2`, 4096-token limit, and the common generation fields above.

## What is not verified

- No saved baseline metadata field records a Hugging Face revision, publisher commit, or weight-shard checksum. This documentation deliberately does not supply one.
- The Qwen baseline's exact public upstream checkpoint and published parameter count are unresolved from the repository and archive evidence. Its two conditions must be treated as condition settings of the repository-recorded local checkpoint, not as independently identified published releases.
- The standard local checkpoint locations are runtime conventions. They do not imply that a checkout contains those weights, nor do they identify a particular copy of them.
