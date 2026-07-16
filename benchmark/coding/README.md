# Coding Tool-Routing Datasets

These datasets evaluate selection and argument generation for the seven read-only
coding tools backed by the deterministic `example/research-mcp` fixture.

## Files

- `tool_routing_coding_smoke.json` contains 7 direct examples, one per tool.
- `tool_routing_coding_controlled.json` contains 35 balanced examples, five per
  tool, spanning direct requests, same-domain distractors, parameter-specific
  requests, paraphrases, and difficult indirect requests.

Every example exposes the same tool menu:

```text
code_list_files
code_read_file
code_search_text
git_log
git_show
git_diff
git_status
```

The examples use the repository ID `example/research-mcp` and fixture version
`coding_fixture_v1`. The fixture is created deterministically by
`mcp_server/coding_state.py`, so file contents and Git history remain stable
across offline runs.

## Schema and Scoring

Each JSON record follows the evaluator's current benchmark schema:

- `query` is the natural-language routing request.
- `available_tools` is the candidate tool menu for that example.
- `expected_tool` is the routing label.
- `expected_args` is the exact argument-generation label.
- `expected_answer` is a partial semantic oracle for checking the fixture output.
- `difficulty` and `perturbation_type` identify the controlled variation.
- `fixture_id`, `fixture_version`, and `provenance_type` record dataset provenance.

The current evaluator scores tool choice and exact argument match. It can also
report whether the predicted call executes successfully. `expected_answer` is
included for fixture validation and future answer-level scoring; it is not
currently scored by `evaluation/evaluate.py`.

## Run

From the repository root:

```bash
python evaluation/evaluate.py \
  --dataset benchmark/coding/tool_routing_coding_smoke.json

python evaluation/evaluate.py \
  --dataset benchmark/coding/tool_routing_coding_controlled.json
```

Add `--call-predicted-tools` to execute the model's predicted calls.

## Public-Dataset Boundary

These files are controlled synthetic benchmarks; they are not represented as
SWE-bench or CodeSearchNet derivatives. CodeSearchNet is a natural future source
for repository file-access and lexical-search prompts, while SWE-bench is a
natural future source for Git history and diff tasks. However, the current coding
server intentionally allowlists one local fixture. Raw public repository IDs and
base commits therefore cannot be executed by these tools.

A public-adapted coding benchmark should be added only alongside versioned local
repository snapshots (or a multi-repository fixture loader), with the upstream
dataset row or instance ID recorded on every adapted example. No external corpus
or raw repository snapshot is vendored here.
