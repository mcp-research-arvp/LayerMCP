# Tool Fixture Ground Truth

This note clarifies what the current LayerMCP fixtures and benchmark labels mean. The current repository uses deterministic offline MCP tools and controlled benchmark examples. It does not yet evaluate real public-dataset task success.

## Current MCP Tools

All live tools are implemented in `mcp_server/tool_impls.py` and registered in `mcp_server/server.py`.

| Tool | Domain | Implementation | Registered | Offline deterministic | Fixture data or logic |
|---|---|---|---|---|---|
| `calculator` | Mathematics | `mcp_server/tool_impls.py` | `mcp_server/server.py` | Yes | Parses arithmetic with Python `ast` and evaluates only allowlisted operators: add, subtract, multiply, divide, floor divide, modulo, power, unary plus, and unary minus. |
| `unit_converter` | Mathematics | `mcp_server/tool_impls.py` | `mcp_server/server.py` | Yes | Supports fixed formula conversions for km/miles, miles/km, Celsius/Fahrenheit, and Fahrenheit/Celsius. |
| `stock_price_api` | Finance | `mcp_server/tool_impls.py` | `mcp_server/server.py` | Yes | Uses `_STOCK_PRICES`: `AAPL = 214.35`, `MSFT = 497.12`, `TSLA = 182.44`, `NVDA = 141.67`. These are fake offline prices. |
| `github_search` | Coding | `mcp_server/tool_impls.py` | `mcp_server/server.py` | Yes | Tokenizes the query and returns two mock GitHub-style results from fixed repositories: `example/research-mcp` and `example/tool-routing`. No GitHub API is called. |
| `read_code_file` | Coding | `mcp_server/tool_impls.py` | `mcp_server/server.py` | Yes | Uses `_CODE_FILES` with fixed fake contents for `src/auth.py`, `src/payments.py`, `tests/test_auth.py`, and `README.md`. |
| `customer_lookup` | Enterprise automation | `mcp_server/tool_impls.py` | `mcp_server/server.py` | Yes | Validates a mock customer ID and derives status from the last digit: odd means `premium`, even means `standard`. |
| `ticket_router` | Enterprise automation | `mcp_server/tool_impls.py` | `mcp_server/server.py` | Yes | Routes text by keyword rules into `billing`, `account`, `security`, or `technical_support`. Billing keywords are checked before account/security keywords. |

## Current Benchmark Files

The repository currently has these benchmark files:

| File | Examples | Schema style | Purpose |
|---|---:|---|---|
| `benchmark/tool_routing.json` | 4 | Original minimal schema | Toy routing baseline with `query`, `expected_tool`, and `tool_args`. |
| `benchmark/tool_routing_smoke.json` | 8 | Expanded schema | Small four-domain smoke benchmark, 2 examples per domain. |
| `benchmark/tool_routing_controlled.json` | 40 | Expanded schema | Controlled Phase 1 benchmark, 10 examples per domain. |
| `benchmark/tool_routing_phase2_seed.json` | 16 | Expanded schema plus `phase2_focus` | Small seed set for logit-lens analysis. |

## What Benchmark Ground Truth Currently Checks

The primary ground-truth label is `expected_tool`.

Current evaluation accuracy means:

- The router selected the expected MCP tool name.
- For example, the model chose `calculator` when `expected_tool` was `calculator`.
- Hallucination count tracks whether the router returned `hallucinated_tool`.

When `--call-predicted-tools` is used, the evaluator also attempts to execute the predicted tool using benchmark-provided arguments:

- Old schema: `tool_args`
- New schema: `expected_args`
- Backward-compatible fallback: if `expected_args` is missing, the evaluator uses `tool_args`

This execution confirms that the selected tool can run with fixture arguments, but it does not mean the model generated those arguments.

## Tool-Selection Ground Truth vs Final-Answer Ground Truth

Tool-selection ground truth answers this question:

> Did the model choose the correct tool name from the available tools?

Final-answer or task ground truth answers a different question:

> Did the full system solve the underlying task, including argument generation, tool execution, reasoning, and final answer validation?

The current benchmark mostly measures tool-selection ground truth. Some newer examples include `expected_answer`, but the evaluator does not yet use that field for scoring. Current benchmark accuracy therefore means correct tool-name selection, not full argument-generation accuracy and not real dataset task success.

Examples:

- A correct `stock_price_api` selection means the model chose the finance lookup tool, not that it independently produced the ticker or verified a real market price.
- A correct `unit_converter` selection means the model chose the conversion tool, not that it generated the conversion arguments itself.
- A correct `read_code_file` selection means the model chose file reading over search, not that it solved a repository task or passed tests.
- A correct `ticket_router` selection means the model chose the ticket-routing tool, not that it completed a real workflow or state transition.

## Current Fixture Ground Truth

The tool fixtures provide deterministic outputs:

- `calculator` output is derived from safe arithmetic evaluation.
- `unit_converter` output is derived from fixed formulas and rounded to 4 decimals.
- `stock_price_api` output is a fake lookup table, not live market data.
- `github_search` output is a mock search response, not live GitHub data.
- `read_code_file` output is a mock in-memory file table, not the repository filesystem.
- `customer_lookup` output is a deterministic ID rule, not a customer database.
- `ticket_router` output is deterministic keyword matching, not a trained classifier or workflow engine.

These fixtures are useful for controlled experiments because they remove network variability and make results reproducible.

## Known Ambiguities

Some labels are intentionally simplified:

- `github_search` and `read_code_file` are both coding-domain tools, but one represents search and the other represents file retrieval.
- `customer_lookup` and `ticket_router` are both enterprise automation tools, but one is entity lookup and the other is workflow routing.
- `calculator` and `unit_converter` are both mathematics-adjacent tools, but conversion examples are not scored as general math reasoning.
- `stock_price_api` is finance-domain, but its prices are fake fixture values and should not be interpreted as financial data.
- `expected_answer` exists in newer benchmark files, but it is currently documentation/metadata unless a future evaluator explicitly scores it.

## Future Dataset Wrapping

Future public or realistic datasets will need to be wrapped into MCP-style benchmark examples.

Finance datasets should provide:

- financial tables
- filings or supporting documents
- questions
- numeric or textual answers
- tool wrappers for lookup, extraction, calculation, or verification

Math datasets should provide:

- math problems
- final answers
- optional solution traces
- tool wrappers for arithmetic, symbolic computation, unit conversion, or answer checking

Coding datasets should provide:

- prompts
- tests
- repository snapshots or file contents
- expected patches or outputs
- tool wrappers for code search, file reading, test execution, issue lookup, or patch inspection

Enterprise benchmarks should provide:

- workflows
- policies
- user or ticket state
- allowed actions
- state-transition checks
- tool wrappers for lookup, routing, approval, policy search, and workflow execution

Once those datasets are wrapped, LayerMCP can distinguish tool-selection accuracy from argument correctness, execution success, and final task success. That distinction is not fully implemented in the current fixture-only benchmark.
