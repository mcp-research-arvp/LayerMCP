# Coding Tool-Routing Datasets

These datasets evaluate selection and argument generation for the seven
read-only coding tools backed by deterministic, allowlisted repository fixtures.

## Files

- `tool_routing_coding_smoke.json` contains 7 direct examples, one per tool.
- `tool_routing_coding_controlled.json` contains 35 balanced examples, five per
  tool, spanning direct requests, same-domain distractors, parameter-specific
  requests, paraphrases, and difficult indirect requests.
- `tool_routing_coding_upstream_inspired.json` contains 28 generated examples,
  four per tool, based on documented upstream usage patterns and adapted to the
  local fixture and tool signatures.
- `tool_routing_coding_codesearchnet_public_derived.json` contains 15
  self-contained search instructions wrapping exact human-evaluation queries
  from the CodeSearchNet Challenge.
- `tool_routing_coding_sweagent_multistep.json` contains one exact SWE-bench
  issue and three ordered read-only exploration actions from the official
  SWE-agent demonstration trajectory.
- `fixtures/codesearchnet_public_annotations.json` defines the offline
  annotation repository used by those 15 executable rows.
- `fixtures/CODESEARCHNET_ATTRIBUTION.md` records the paper, pinned sources,
  hashes, MIT license, and the changes made for this repository.
- `fixtures/CODESEARCHNET_LICENSE.txt` preserves the exact MIT notice from the
  pinned CodeSearchNet revision.
- `fixtures/sweagent_marshmallow_1867.json` defines the bounded repository
  excerpt needed to execute the selected trajectory actions.
- `fixtures/SWEAGENT_MARSHMALLOW_1867_ATTRIBUTION.md` records the SWE-agent
  paper, trajectory hash, repository commit, gold pull request, and adaptation
  boundary.

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

The three generated datasets use repository ID `example/research-mcp` and
fixture version `coding_fixture_v1`. The public-derived dataset uses repository
ID `codesearchnet-public-v1` and fixture version
`coding_codesearchnet_fixture_v1`. Both fixtures are created deterministically
by `mcp_server/coding_state.py`, so file contents and Git history remain stable
across offline runs.

The SWE-agent multi-step dataset uses repository ID
`swebench-marshmallow-1867` and fixture version
`coding_sweagent_marshmallow_1867_fixture_v1`. It preserves the issue text and
trajectory actions exactly and maps them mechanically onto the existing
read-only coding tools.

## Schema and Scoring

Each JSON record follows the evaluator's current benchmark schema:

- `query` is the natural-language routing request.
- `available_tools` is the candidate tool menu for that example.
- `expected_tool` is the routing label.
- `expected_args` is the exact argument-generation label.
- `expected_answer` is a partial semantic oracle for checking the fixture output.
- `difficulty` and `perturbation_type` identify the controlled variation.
- `fixture_id`, `fixture_version`, and `provenance_type` record dataset provenance.

Rows in the upstream-inspired set also include `query_origin`,
`inspiration_repository`, `inspiration_url`, and `inspiration_reference`. These
fields distinguish generated queries from text copied out of a public corpus.
In the CodeSearchNet-derived set, `query` is the generated instruction presented
to the router, while `original_query` preserves the exact published search text.
The `query_origin` and `original_query_origin` fields keep those two origins
separate, and `query_wrapper_id` versions the instruction template.

The current evaluator scores tool choice and exact argument match. It can also
report whether the predicted call executes successfully. `expected_answer` is
included for fixture validation and future answer-level scoring; it is not
currently scored by `evaluation/evaluate.py`.

For `multi_step_tool_routing`, the evaluator processes `expected_steps` in
order, carries prior predicted calls and observations into the next routing
prompt, and reports both per-step accuracy and complete ordered-sequence
accuracy.

## Run

From the repository root:

```bash
python evaluation/evaluate.py \
  --dataset benchmark/coding/tool_routing_coding_smoke.json

python evaluation/evaluate.py \
  --dataset benchmark/coding/tool_routing_coding_controlled.json

python evaluation/evaluate.py \
  --dataset benchmark/coding/tool_routing_coding_upstream_inspired.json

python evaluation/evaluate.py \
  --dataset benchmark/coding/tool_routing_coding_codesearchnet_public_derived.json

python evaluation/evaluate.py \
  --dataset benchmark/coding/tool_routing_coding_sweagent_multistep.json
```

Add `--call-predicted-tools` to execute the model's predicted calls.

## Public SWE-agent Trajectory

The selected SWE-agent demonstration contains 14 actions for SWE-bench instance
`marshmallow-code__marshmallow-1867`. The current LayerMCP coding catalog is
read-only, so the multi-step benchmark retains the three reusable exploration
actions: listing the repository, locating `fields.py`, and opening the relevant
source window. Installation, temporary file creation, execution, edits,
cleanup, and submission are recorded in the attribution boundary but are not
misrepresented as supported calls.

The trajectory is pinned to SWE-agent revision
`3ea751c087f32b16e039a2233dd6eefecef325d5`, and the Marshmallow repository
excerpt is pinned to base commit
`bfd2593d4b416122e30cdefe0c72d322ef471611`.

## Upstream-Inspired Queries

The upstream repositories do not ship a natural-language query benchmark for
these operations. They provide tool descriptions, argument schemas, and command
examples instead. The 28 upstream-inspired queries are therefore generated
synthetic prompts, not verbatim public queries.

Their documented behavior comes from:

- the [MCP filesystem server tool API](https://github.com/modelcontextprotocol/servers/blob/main/src/filesystem/README.md#tools)
  for file listing and reading;
- the [ripgrep guide](https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md)
  for recursive, fixed-string, case-sensitive, and glob-scoped content search;
- the [MCP Git server tool API](https://github.com/modelcontextprotocol/servers/blob/main/src/git/README.md#tools)
  for status, log, show, and diff workflows.

Every generated query is grounded in arguments supported by LayerMCP and has an
`expected_answer` subset verified against `example/research-mcp`. Unsupported
upstream features such as ripgrep regular expressions, Git log date filters, and
three-dot Git diffs were not adapted.

## Public-Derived CodeSearchNet Queries

CodeSearchNet published 99 natural-language queries for its human-relevance
evaluation. The public-derived file selects 15 of those exact query strings and
pairs each with one selected Python annotation record rated relevance 3. Each
selected `(Language, Query, GitHubUrl)` tuple occurs exactly once in the pinned
annotation file. Every row records the zero-based source indexes, exact
annotation record and target URL, pinned publication revision, source-file
hashes, paper, license, and adaptation boundary.

The router does not receive the short source string by itself. Each row keeps it
unchanged in `original_query` and places it inside a consistent instruction in
`query` that names repository `codesearchnet-public-v1`, limits the lookup to
`resources/annotationStore_selected.jsonl`, requests exact case-sensitive text,
and bounds the result count to one. This makes every expected argument
inferable from the prompt rather than relying on hidden dataset context.

The offline fixture contains normalized copies of only the selected
CodeSearchNet query and annotation records. It does not copy the target source
code referenced by the annotations, so those upstream repositories and their
licenses are not redistributed. `code_search_text` performs fixed-string lookup
over the normalized annotation JSONL. This tests selection and argument
generation for the lexical-search tool; it does not reproduce CodeSearchNet's
semantic code retrieval, human-label aggregation, candidate corpus, or NDCG
evaluation.

The source files are pinned to CodeSearchNet's human-evaluation publication
commit `bb121a53a559e99a6849409355ee5c83803f2e87`. The CodeSearchNet paper and
repository describe the 99 queries and human relevance judgments:

- [CodeSearchNet Challenge paper](https://arxiv.org/abs/1909.09436)
- [Pinned query list](https://github.com/github/CodeSearchNet/blob/bb121a53a559e99a6849409355ee5c83803f2e87/resources/queries.csv)
- [Pinned annotation store](https://github.com/github/CodeSearchNet/blob/bb121a53a559e99a6849409355ee5c83803f2e87/resources/annotationStore.csv)
- [MIT license](https://github.com/github/CodeSearchNet/blob/bb121a53a559e99a6849409355ee5c83803f2e87/LICENSE)

SWE-bench issue statements are not included in this single-tool dataset. Their
intended task requires repository modification and test-based patch validation,
while the current seven coding tools are read-only. HumanEval is likewise
deferred until sandboxed execution and test-runner tools exist. Adding either
benchmark now would assign a subjective first tool rather than faithfully
represent its published task.
