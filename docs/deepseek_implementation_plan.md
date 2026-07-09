# DeepSeek Implementation Plan

DeepSeek implementation should live in Tony's ModelSurgery repo, not in
LayerMCP. LayerMCP should remain the benchmark, tool-schema, prompt, scoring,
and comparison harness.

## First Planned Target

The first planned native DeepSeek target is **DeepSeek-LLM-7B**, unless Tony
changes preference.

DeepSeek-LLM-7B is preferred as the first target because it is a broad-domain
base/instruct-style family that better matches LayerMCP's current benchmark
domains:

- finance
- mathematics
- coding
- enterprise automation

It is a better first architecture target than a coding-specialized model when
the goal is broad tool-selection behavior.

## Later DeepSeek-Coder Comparison

DeepSeek-Coder can be a later coding-specific comparison. It is useful for
software-engineering routing and repo-navigation tasks, but it is less ideal as
the first implementation target for broad-domain MCP routing.

## Why Not R1 Distilled Qwen/Llama First

DeepSeek-R1 distilled Qwen/Llama models are not ideal as the first native
DeepSeek architecture implementation because they inherit Qwen or Llama
architectures. They can be useful baselines, but they do not validate a native
DeepSeek architecture path inside ModelSurgery.

## Why Not Full DeepSeek-V3/R1 First

Full DeepSeek-V3/R1-class models are likely too large for the current local
setup and would add complexity before the smaller native architecture path is
validated.

## ModelSurgery Repo Notes

From the current ModelSurgery repo analysis:

- Phi-4 is the cleanest structural template.
- Phi exposes layers directly, for example `generator.model.layers[i]`.
- DeepSeek likely needs LLaMA-style separate `q`, `k`, `v`, and `o` attention
  projections.
- DeepSeek likely needs gated MLP projections: `gate`, `up`, and `down`.

Those implementation details belong in ModelSurgery's PyTorch model, weights,
inference, and API layers.

## Minimum DeepSeek Validation Milestones

1. Hugging Face baseline generation works.
2. Config is inspected.
3. Custom PyTorch model skeleton matches config.
4. Weights load.
5. Generation works.
6. Layers are accessible.
7. Tool-call API works.
8. LayerMCP baseline accuracy runs.
9. Layer modification experiment runs.
10. Accuracy comparison is saved.

## LayerMCP Role After DeepSeek Is Ready

Once ModelSurgery can return tool calls for DeepSeek, LayerMCP should:

- send the same benchmark samples used for Qwen baselines
- pass available tool schemas to the ModelSurgery API
- parse normalized tool calls
- score against `expected_tool`
- save baseline results
- rerun after layer changes in ModelSurgery
- compare accuracy, hallucination rate, and parser failures

LayerMCP should not add DeepSeek architecture code.
