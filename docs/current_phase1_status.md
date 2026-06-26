# Current Phase 1 Status

LayerMCP is functionally healthy as a toy Phase 1 prototype, but it is not yet a full Phase 1 benchmark.

## Baseline Live Tools

At the time of the audit, the MCP server discovered these tools:

- `calculator`
- `customer_lookup`
- `github_search`

After the controlled Phase 1 scaffold expansion, the intended live toolset also includes:

- `stock_price_api`
- `unit_converter`
- `read_code_file`
- `ticket_router`

## Current Benchmark

The current default benchmark is:

- `benchmark/tool_routing.json`

It contains 4 custom toy examples.

## Confirmed Successful Command

```powershell
layermcp-evaluate --call-predicted-tools
```

## Observed Result

The observed toy result was:

- 4/4 correct
- 0 hallucinations
- Average latency about 35.70 seconds
- 4 executed tool calls

## Status Summary

The repository can run end to end, discover MCP tools, route benchmark queries, optionally call predicted tools, and print summary metrics. It is ready for a controlled Phase 1 expansion, but it does not yet include broad domain coverage, structured benchmark variants, argument correctness scoring, robust result analysis, or Phase 2 interpretability methods.
