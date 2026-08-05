# LayerMCP Project Ground Truth

## Research objective

LayerMCP evaluates lightweight and open-source language models on MCP-style
tool selection and tool execution. Once its benchmark and evaluation
foundations are reliable, the project studies the internal mechanisms behind
tool-use behavior and explores selective fine-tuning. Executable repository
behavior, validated benchmark data, and reproducible results take precedence
over assumptions recorded in old prose or experiments.

## Project phases

1. **Benchmark and evaluation standardization.** Establish comparable,
   executable evaluation across Mathematics, Finance, Coding, and Enterprise
   automation.
2. **Interpretability and mechanistic analysis.** Study how models represent
   tool choice, arguments, routing errors, and successful tool-use behavior.
3. **Selective adaptation.** Evaluate approaches such as LoRA, QLoRA, and layer
   freezing only after stable benchmarks and evaluator semantics are in place.

## Current benchmark domains

- Mathematics
- Finance
- Coding
- Enterprise automation

## Benchmark families

- **Controlled:** internally authored cases targeting known tool contracts.
- **Smoke:** small suites for fast end-to-end checks, not headline results.
- **Public/source-derived:** examples traceable to public sources with explicit
  provenance and documented transformations.
- **Replay/offline:** routing over recorded trajectories or bounded offline
  artifacts; report separately from live execution benchmarks.
- **Single-step:** one expected tool call and argument payload per sample.
- **Multi-step:** ordered expected steps with workflow- and step-level metrics;
  the evaluator's execution mode determines how state and history are handled.

These labels describe different evidence and must not be silently aggregated
when doing so would obscure benchmark semantics.

## Core metrics

- Tool-selection accuracy
- Exact argument-match accuracy
- Execution-success rate
- No-tool-call rate
- Final-outcome accuracy
- Workflow- and step-level metrics for multi-step benchmarks

Metric interpretation must follow the current evaluator implementation.
Execution success alone does not prove semantic correctness, and teacher-forced
step routing must not be described as autonomous end-to-end planning.

## Evidence standard

Research claims should be supported by the relevant combination of:

- structural validation of benchmark schemas and tool contracts;
- evaluator and integration tests;
- reproducible benchmark generation with pinned provenance where applicable;
- model-run sample and summary outputs; and
- explicit separation of controlled, public/source-derived, replay/offline,
  single-step, and multi-step results.

Failures, skips, environmental limitations, and incomplete runs are part of
the evidence and must be reported rather than hidden.

## Non-goals

- Do not chase arbitrary datasets without a clear research or tool mapping.
- Do not rewrite documentation for every minor implementation change.
- Do not merge benchmark formats that the evaluator cannot execute correctly.
- Do not treat prompt-only improvements as benchmark improvements without
  evaluation evidence.
- Do not begin broad fine-tuning work before evaluation foundations are stable.

## Timeline and status

According to team-provided historical context, the initial capstone evaluated
lightweight language-model tool use across Math, Statistics, Finance, and SQL.
This statement is project history supplied by the team rather than a claim
independently substantiated by an in-repository report. The current summer
research extends that work into standardized MCP benchmarks, broader domains,
multi-step workflows, and planning for interpretability and selective
fine-tuning.

The project is currently strengthening benchmark classification, executable
grounding, evaluator semantics, and reproducibility while expanding
multi-domain coverage.

## How to update this file

Update this document only when project goals, phases, benchmark policy,
evaluator semantics, or evidence standards change. A pull request modifying it
must explicitly explain why the project ground truth changed.
