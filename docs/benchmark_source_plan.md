# Benchmark Source Plan

LayerMCP will use two benchmark source categories over time: tool-use benchmarks that already provide function-calling structure, and domain-task benchmarks that provide realistic tasks and ground truth but need MCP-style wrapping.

The immediate repository implementation does not integrate or download public datasets. Instead, it creates a controlled offline MCP-style benchmark inspired by these categories, using deterministic tools and fixtures.

## A. Tool-Use Benchmark Sources

- BFCL
- tau-bench
- WorkArena
- WildToolBench

These sources already include tool or function-calling structure. They are useful references for schemas, tool selection, tool arguments, multi-step tool paths, and execution success metrics.

## B. Domain-Task Benchmark Sources

- FinanceBench
- FinQA
- TAT-QA
- GSM8K
- MATH / MATH-500
- SWE-bench
- HumanEval
- CodeSearchNet

These sources provide realistic finance, math, and coding tasks with ground truth. They generally need MCP-style wrapping before they can be used for tool-routing experiments. The wrapping should define available tools, expected tools, expected arguments, distractors, and expected answers.

## C. Immediate Controlled Dataset Plan

The immediate plan is to build small deterministic examples with:

- `id`
- `domain`
- `task_type`
- `difficulty`
- `source`
- `query`
- `available_tools`
- `expected_tool`
- `expected_args`
- `expected_answer`
- `perturbation_type`
- `notes`

The first controlled datasets should cover finance, coding, mathematics, and enterprise automation. They should include same-domain and cross-domain distractor tools where possible, remain single-tool for now, and be executable fully offline.
