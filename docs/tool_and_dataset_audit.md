# Tool and Dataset Audit

## Current repo status

LayerMCP currently contains a small, runnable MCP tool-routing prototype. It has one MCP server, three deterministic offline MCP tools, one JSON benchmark file with four examples, one Qwen-based router, and one evaluation harness.

Relevant files:

| Area | File | Current role |
|---|---|---|
| MCP tool definitions | `mcp_server/tool_impls.py` | Implements `calculator`, `customer_lookup`, and `github_search`. |
| MCP tool registration | `mcp_server/server.py` | Creates a `FastMCP` server and registers the three implemented tools. |
| Benchmark data | `benchmark/tool_routing.json` | Contains four toy tool-routing examples. |
| Model routing | `models/qwen_router.py` | Loads a Hugging Face causal LM and prompts it to choose a tool name. |
| Evaluation | `evaluation/evaluate.py` | Loads benchmark data, starts the MCP server, discovers tools, calls the router, computes metrics, and optionally executes predicted tools. |
| Package and CLI setup | `pyproject.toml` | Declares dependencies, package metadata, and CLI entrypoints. |
| Requirements install path | `requirements.txt` | Mirrors runtime dependencies for `pip install -r requirements.txt`. |
| Usage documentation | `README.md` | Describes setup, current tools, benchmark format, and evaluation flow. |

The current known runnable baseline is:

- `layermcp-evaluate --call-predicted-tools` runs successfully.
- Live discovered tools are `calculator`, `customer_lookup`, and `github_search`.
- The current benchmark has 4 examples.
- The current toy run achieved 4/4 correct, 0 hallucinations, and 4 executed tool calls when `--call-predicted-tools` was enabled.

## Current MCP tools

All implemented tools live in `mcp_server/tool_impls.py` and are registered in `mcp_server/server.py`.

| Tool | Implementation | Registration | Signature | Return format | Offline deterministic | Likely domain | Executable through evaluation | Description quality |
|---|---|---|---|---|---|---|---|---|
| `calculator` | `mcp_server/tool_impls.py`, function `calculator` | `mcp_server/server.py`, `mcp.tool()(calculator)` | `calculator(expression: str) -> dict[str, Any]` | `{"expression": str, "result": int | float}` | Yes. Uses Python AST parsing with an allowlist of arithmetic operators. | Mathematics | Yes, when predicted and `tool_args.expression` is present. | Sufficient for simple experiments. The docstring says it safely evaluates arithmetic, but future benchmark work may need richer descriptions and explicit schema examples. |
| `customer_lookup` | `mcp_server/tool_impls.py`, function `customer_lookup` | `mcp_server/server.py`, `mcp.tool()(customer_lookup)` | `customer_lookup(customer_id: str) -> dict[str, Any]` | `{"customer_id": str, "status": "premium" | "standard", "source": "offline-fixture"}` | Yes. Status is derived from the last digit in the ID. | Enterprise automation | Yes, when predicted and `tool_args.customer_id` is present. | Sufficient for toy routing. The docstring makes the offline mock nature clear, but it does not describe realistic enterprise workflows. |
| `github_search` | `mcp_server/tool_impls.py`, function `github_search` | `mcp_server/server.py`, `mcp.tool()(github_search)` | `github_search(query: str) -> dict[str, Any]` | `{"query": str, "source": "offline-fixture", "results": [{"repository": str, "title": str}, ...]}` | Yes. It returns fixed mock GitHub-style results based on parsed query keywords. | Coding | Yes, when predicted and `tool_args.query` is present. | Sufficient for toy routing. It identifies itself as an offline GitHub-style search, but future experiments may need more realistic repo, issue, PR, and code-search descriptions. |

No finance-domain MCP tools are currently implemented. No multi-step or multi-tool tools are currently implemented.

## Current benchmark files

### `benchmark/tool_routing.json`

This is the only benchmark JSON file currently present.

Summary:

- Number of examples: 4
- Format: top-level JSON list of objects
- Fields present in every example: `query`, `expected_tool`, `tool_args`
- Includes `id`: no
- Includes `domain`: no
- Includes `difficulty`: no
- Includes `available_tools`: no
- Includes `expected_tool`: yes
- Includes `expected_args`: no, but equivalent executable arguments are stored under `tool_args`
- Includes `expected_answer`: no
- Includes source or benchmark origin: no
- Single-tool or multi-tool: single-tool only
- Executable with currently live tools: yes, all 4 examples have `expected_tool` values matching live MCP tools and `tool_args` matching those tool signatures

Current examples:

| Query | Fields | Expected tool | Tool args | Executable now |
|---|---|---|---|---|
| `What is 25 * 17?` | `query`, `expected_tool`, `tool_args` | `calculator` | `{"expression": "25 * 17"}` | Yes |
| `Find customer 12345` | `query`, `expected_tool`, `tool_args` | `customer_lookup` | `{"customer_id": "12345"}` | Yes |
| `Search GitHub for authentication bugs` | `query`, `expected_tool`, `tool_args` | `github_search` | `{"query": "authentication bugs"}` | Yes |
| `Calculate 91 * 74` | `query`, `expected_tool`, `tool_args` | `calculator` | `{"expression": "91 * 74"}` | Yes |

The file is a minimal routing benchmark, not yet a full benchmark schema for tool-use and domain-task research. It lacks stable IDs, domain labels, difficulty labels, source attribution, distractor controls, expected final answers, expected argument assertions, and multi-tool path annotations.

## Current example classification

| Query | Classification | Domain | Notes |
|---|---|---|---|
| `What is 25 * 17?` | Custom toy example | Math | Direct arithmetic routed to `calculator`. This is tool-use style, but not from a known math benchmark. |
| `Find customer 12345` | Custom toy example | Enterprise automation | Direct lookup routed to `customer_lookup`. It resembles an enterprise fixture, not a sourced domain benchmark. |
| `Search GitHub for authentication bugs` | Custom toy example | Coding | Direct search routed to `github_search`. It resembles coding/repo automation, not a sourced coding benchmark. |
| `Calculate 91 * 74` | Custom toy example | Math | Direct arithmetic routed to `calculator`. This is another simple toy tool-use example. |

All current examples are single-step, single-tool, custom toy examples. They already look like function/tool-calling tasks because they include a user query, expected tool, and executable arguments. They are not yet domain-task benchmark content adapted from finance, math, coding, or enterprise datasets.

## Tool-use vs domain-task status

Current status:

- Tool-use benchmark structure exists in a minimal form.
- The examples have `query`, `expected_tool`, and `tool_args`.
- The evaluator discovers live MCP tools and routes over those tools.
- Tool execution works for examples that include `tool_args`.

Current limitations:

- There is no explicit `available_tools` list per example. The evaluator always uses the live server catalog.
- There is no `expected_args` field distinct from execution `tool_args`.
- There is no argument-correctness scoring.
- There is no `expected_answer` or tool-output correctness assertion.
- There are no sourced domain-task examples from finance, coding, mathematics, or enterprise automation datasets.
- There is no benchmark split between pure tool-use examples and domain-task examples wrapped as MCP calls.
- There are no same-domain or cross-domain distractor sets.

The project currently has the scaffold for a tool-use benchmark, but not yet a complete domain-task benchmark.

## Evaluation flow

The evaluator is implemented in `evaluation/evaluate.py`.

Flow:

1. `main()` parses CLI arguments.
2. `_async_main()` loads the selected dataset with `_load_dataset()`.
3. `_load_dataset()` opens the JSON file, parses it with `json.load`, and validates only that the top-level object is a list.
4. `_evaluate_with_server()` starts an MCP stdio server session through `_run_server_session()`.
5. `_run_server_session()` launches `mcp_server/server.py` using the current Python executable, then initializes an MCP `ClientSession`.
6. `_evaluate_with_server()` calls `session.list_tools()` and builds `available_tools = [tool.name for tool in listed_tools.tools]`.
7. For each sample, the evaluator reads `sample["query"]` and `sample["expected_tool"]`.
8. It passes the query and live `available_tools` list to `models.qwen_router.choose_tool()`.
9. Correctness is checked only by exact string equality: `predicted == expected`.
10. Hallucination count increments when the predicted value equals `HALLUCINATED_TOOL`.
11. If `--call-predicted-tools` is enabled and the prediction is not `HALLUCINATED_TOOL`, the evaluator reads `sample.get("tool_args")` and calls `session.call_tool(predicted, tool_args)`.
12. The evaluator prints summary metrics to stdout.

Current metrics:

- Total examples
- Tool-selection accuracy
- Hallucination count
- Average router latency
- Executed tool-call count

Argument checking:

- The evaluator does not check whether predicted arguments match expected arguments.
- The model does not generate arguments; benchmark-provided `tool_args` are passed to whichever tool is predicted.
- This means `--call-predicted-tools` validates that the selected tool can run with benchmark-provided arguments, not that the model produced correct arguments.

Result logging:

- No structured results are saved.
- Per-example outputs and final metrics are printed only to stdout.

Important limitations before expanding:

- Dataset validation is minimal.
- Missing fields produce runtime `KeyError` or skipped tool calls rather than structured failures.
- Per-example domains, IDs, difficulty, source, and expected answers are unavailable.
- Execution success is not separately measured from selection accuracy.
- Tool-call exceptions are not caught and converted into per-example failure records.
- The benchmark cannot currently represent multi-tool paths.
- The evaluator cannot compare multiple controlled variants such as shuffled tool order, renamed tools, generic tool names, or changed descriptions.

## Router behavior

The router is implemented in `models/qwen_router.py`.

Current model:

- Default model: `Qwen/Qwen2.5-3B-Instruct`
- Override mechanism: `LAYERMCP_MODEL_NAME` environment variable
- Loading: shared Hugging Face/PyTorch utilities in `models/model_loader.py`
- CUDA behavior: centralized in `models/model_loader.py`
- Caching: model and tokenizer are cached in-process with `@lru_cache(maxsize=1)`

Prompt construction:

- `_build_prompt(query, available_tools)` creates a plain-text instruction prompt.
- Available tools are represented as a bullet list of tool names only, one per line.
- The model is instructed to return exactly one tool name from the list, or `hallucinated_tool` if none match.
- No tool descriptions, function signatures, JSON schemas, argument schemas, or examples are included in the prompt.

Output parsing:

- `_extract_tool_name(response, available_tools)` lowercases and strips the generated text.
- It first checks exact equality with a known available tool name.
- It then checks whether any available tool name is a substring of the model response.
- It then checks whether `hallucinated_tool` appears as a substring.
- Otherwise it returns `hallucinated_tool`.

Parsing type:

- Exact-match first
- Substring-based fallback
- Not JSON-based
- Fragile for future experiments where tool names overlap, generic names are used, renamed tools share tokens, or the model emits explanations containing multiple tool names

Support for future forced-choice labels:

- The current router can be adapted to A/B/C/D-style forced-choice labels, but it does not currently implement them.
- Current parsing expects tool names, not labels.
- For logit lens or controlled forced-choice experiments, the prompt and parser should be changed to map stable labels to tools and score exact label tokens.

## Coverage gaps

| Capability | Current support | Gap |
|---|---|---|
| Finance tools | No | No implemented finance MCP tool or finance benchmark examples. |
| Coding tools | Minimal | `github_search` is a coding-adjacent offline fixture, but there are no realistic code-analysis, repo-navigation, test, issue, or PR tools. |
| Math tools | Minimal | `calculator` supports simple arithmetic, but not symbolic math, proof checking, multi-step reasoning, or sourced math benchmark tasks. |
| Enterprise automation tools | Minimal | `customer_lookup` is an offline fixture, but there are no workflow, ticketing, CRM, policy, email, calendar, or approval tools. |
| Same-domain distractor tools | No | There are no competing tools within a domain, such as `price_option` vs `calculate_var`, or `github_search` vs `repo_file_lookup`. |
| Cross-domain distractor tools | Minimal | The three current tools are from different rough domains, but there are no controlled distractor sets. |
| Argument correctness evaluation | No | The model does not generate arguments and the evaluator does not compare arguments. |
| Multi-tool/tool-path evaluation | No | The schema and evaluator assume one expected tool per example. |
| Structured result logging | No | Results are printed to stdout only. |
| Controlled perturbations | No | No support for shuffled tool order, renamed tools, generic tool names, hidden descriptions, changed descriptions, or adversarial distractors. |
| Tool descriptions in routing prompt | No | The router prompts with names only. |
| Dataset provenance | No | Examples do not include source or benchmark-origin metadata. |
| Difficulty labels | No | No difficulty field or stratified evaluation. |
| Stable IDs | No | Examples cannot be tracked across benchmark variants. |
| Tests | No obvious test suite | No automated tests are present for tools, dataset validation, router parsing, or evaluator behavior. |

## Recommended next implementation steps

1. Expand the benchmark schema before adding many examples.

   Add stable fields such as `id`, `domain`, `task_type`, `difficulty`, `source`, `query`, `available_tools`, `expected_tool`, `expected_args`, `expected_answer`, and optional `expected_tool_path`. Keep `tool_args` only if it remains distinct from model-predicted arguments.

2. Add result logging.

   Save per-example records to JSONL or CSV with query, available tools, expected tool, predicted tool, correctness, latency, hallucination flag, execution status, tool result summary, and error details.

3. Make output parsing stricter.

   Replace substring parsing with exact constrained outputs. For future logit-lens work, add an A/B/C/D label mode where each tool has a stable label and the parser accepts only one label.

4. Add focused tests for the current scaffold.

   Cover each tool implementation, MCP registration smoke tests, dataset schema validation, router parsing edge cases, and evaluator behavior with a lightweight fake router or fake model.

5. Add controlled benchmark files.

   Create separate benchmark variants for baseline names, shuffled tool order, same-domain distractors, cross-domain distractors, generic tool names, renamed tools, and changed tool descriptions.

6. Expand the toolset by domain.

   Add deterministic offline tools for finance, coding, math, and enterprise automation. Start with small, auditable fixtures before connecting to any external APIs.

7. Separate tool-use examples from domain-task examples.

   Maintain one benchmark family for direct tool-use/function-calling tasks and another for domain-task prompts that require wrapping domain reasoning into tool calls.

8. Add argument and answer correctness.

   Once the router or agent generates arguments, compare predicted arguments against `expected_args` and compare tool outputs or final responses against `expected_answer`.

9. Document the benchmark contract.

   Add a schema reference describing required fields, optional fields, allowed domains, task types, evaluation assumptions, and how controlled variants should be generated.

10. Delay Phase 2 methods until Phase 1 is more complete.

   Logit lens, probing, activation patching, attribution patching, ablation, LoRA, QLoRA, and fine-tuning should wait until the benchmark schema, tool coverage, logging, parsing, and tests are stable enough to produce interpretable measurements.
