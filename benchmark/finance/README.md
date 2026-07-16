# Finance Tool-Routing Datasets

These datasets evaluate tool selection and argument generation for ten read-only
finance research tools. The main fixture is deterministic and offline: LMCP and
TBLR are fictional companies, their filings and market values are synthetic,
and no tool makes a network request.

## Tool Menu

Every row exposes the tools in this order:

```text
finance_lookup_company
finance_search_filings
finance_get_filing_section
finance_get_company_facts
finance_get_financial_statement
finance_parse_xbrl
finance_query_table
finance_extract_pdf_tables
finance_get_market_quote
finance_get_market_time_series
```

The catalog covers the operations needed for filing retrieval and abstraction,
XBRL extraction, local table analytics, PDF-table retrieval, and market-data
lookup. Its behavior is informed by these upstream interfaces:

- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
  for submissions and company facts;
- [EdgarTools](https://github.com/dgunning/edgartools) for company, filing,
  section, fact, and statement workflows;
- [Arelle's Python API](https://arelle.readthedocs.io/en/latest/python_api/python_api.html)
  for XBRL parsing patterns;
- [DuckDB `SELECT`](https://duckdb.org/docs/current/sql/statements/select) for
  local analytical-query patterns;
- [Camelot](https://camelot-py.readthedocs.io/en/master/) for page- and
  flavor-oriented PDF-table extraction patterns; and
- [Alpha Vantage](https://www.alphavantage.co/documentation/) for quote and
  time-series request patterns.

These are local research abstractions, not bundled copies of the upstream
projects or live clients for their services.

## Files

- `tool_routing_finance_smoke.json` contains 10 direct examples, one per tool.
- `tool_routing_finance_controlled.json` contains 50 balanced examples, five per
  tool: direct, same-domain distractor, parameter-specific, paraphrased, and
  difficult indirect requests.
- `tool_routing_finance_upstream_inspired.json` contains 40 generated examples,
  four per tool, adapted from documented upstream usage patterns.
- `tool_routing_finance_public_derived.json` contains 15 table-reasoning rows
  adapted from the official FinQA public test split.
- `fixtures/finqa_public_test_cells.json` contains the 201 normalized source
  cells needed to execute those 15 FinQA rows.
- `fixtures/FINQA_LICENSE.txt` preserves the FinQA MIT notice.

The three generated datasets use fixture ID `example/finance-research` and
version `finance_fixture_v1`. The FinQA-derived dataset uses table ID
`finqa-public-test-v1` and pins upstream revision
`0f16e2867befa6840783e58be38c9efb9229d742` in every row and in the fixture
provenance.

## Runtime Boundaries

- Filing, fact, statement, XBRL, PDF, and market results come only from the
  server-owned fixture.
- `finance_query_table` loads one allowlisted dataset into a private in-memory
  SQLite table named `data`. It accepts a single bounded `SELECT` or `WITH`
  statement and denies writes, metadata access, unapproved functions, external
  files, extensions, and attached databases. The response reports `sqlite3` as
  the actual engine; the query interface is DuckDB-inspired but does not claim
  to execute DuckDB.
- `finance_extract_pdf_tables` returns pre-extracted fixture tables. It records
  the requested Camelot-style flavor but does not parse a live PDF.
- Market tools return a dated synthetic snapshot ending on 2025-01-10, not
  current or investment-grade market data.

## Schema and Provenance

Each benchmark row follows the evaluator's current schema:

- `query` is the natural-language routing request.
- `available_tools` is the fixed candidate menu.
- `expected_tool` and `expected_args` are the routing and argument labels.
- `expected_answer` is a partial semantic oracle verified against the tool.
- `difficulty` and `perturbation_type` describe the controlled variation.

Generated upstream-inspired rows use
`query_origin: generated_from_upstream_documentation` and include an inspiration
repository, URL, and reference. They are synthetic prompts, not copied public
queries.

Public-derived rows retain the original FinQA question and record the exact
split, zero-based row index, example ID, source program, execution answer,
revision, URL, copyright, license, and adaptation method. This narrow fixture
tests routing and bounded SQL argument generation; it is not a reproduction of
the full FinQA retrieval or program-generation benchmark.

## Run

From the repository root:

```bash
python evaluation/evaluate.py \
  --dataset benchmark/finance/tool_routing_finance_smoke.json

python evaluation/evaluate.py \
  --dataset benchmark/finance/tool_routing_finance_controlled.json

python evaluation/evaluate.py \
  --dataset benchmark/finance/tool_routing_finance_upstream_inspired.json

python evaluation/evaluate.py \
  --dataset benchmark/finance/tool_routing_finance_public_derived.json
```

Add `--call-predicted-tools` to execute the model's predicted calls.
