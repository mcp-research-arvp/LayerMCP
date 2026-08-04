# Finance Tool-Routing Datasets

These datasets evaluate tool selection and argument generation for ten baseline
read-only finance research tools plus four FinRetrieval-only replay tools. The
main fixture is deterministic and offline: LMCP and TBLR are fictional
companies, their filings and market values are synthetic, and no tool makes a
network request.

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

FinRetrieval workflows additionally expose:

```text
finance_discover_companies
finance_discover_company_series
finance_get_company_fundamentals
finance_search_web_archive
```

These four tools replay selected released trajectories from a checked-in
fixture. They do not access Daloopa, a browser, or the live web.

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

- `finance_smoke.json` contains 10 direct examples, one per tool.
- `finance_controlled.json` contains 50 balanced examples, five per
  tool: direct, same-domain distractor, parameter-specific, paraphrased, and
  difficult indirect requests.
- `finance_upstream_inspired.json` contains 40 generated examples,
  four per tool, adapted from documented upstream usage patterns.
- `finance_tatqa_public_derived.json` contains 15 exact questions
  selected from the official TAT-QA test-gold split and adapted to bounded SQL.
- `finance_convfinqa_multistep.json` contains ten exact
  ConvFinQA development-split conversations (35 paper-authored turns) adapted
  into ordered, executable retrieval-and-calculation workflows.
- `finance_finqa_test_single.json` contains the 642 remaining
  FinQA test questions whose gold programs have one operation.
- `finance_finqa_test_multistep.json` contains the 490 remaining
  FinQA test questions whose gold programs have two to five operations. They
  provide 1,111 ordered calls.
- `finance_finretrieval_replay_multistep.json` contains 485 exact
  FinRetrieval questions with one selected correct multi-call model trajectory
  of at most five calls each, totaling 1,490 ordered calls.
- `fixtures/finqa_test_program_results_cells.json` contains one compact row per
  gold operation for the other 1,132 FinQA test questions (1,753 rows).
- `fixtures/tatqa_public_test_gold_cells.json` contains the normalized source
  tables needed to execute the 15 TAT-QA rows.
- `fixtures/finretrieval_replay.json` preserves 1,596 unique normalized,
  deterministic replay records for all 498 selected source trajectories (1,608
  calls), including traces omitted from the benchmark by the five-call limit.
- `fixtures/FINQA_LICENSE.txt` preserves the FinQA MIT notice.
- `fixtures/FINRETRIEVAL_ATTRIBUTION.md` records the FinRetrieval paper,
  pinned artifacts and hashes, MIT dataset-card declaration, selection rule,
  tool normalization, and replay boundary.
- `fixtures/TATQA_ATTRIBUTION.md` records the TAT-QA paper, pinned source, CC BY
  4.0 license, and the changes made for this repository.
- `fixtures/convfinqa_dev_cells.json` contains the 20 normalized evidence rows
  needed by the selected ConvFinQA conversations.
- `fixtures/CONVFINQA_ATTRIBUTION.md` and `fixtures/CONVFINQA_LICENSE.txt`
  record the pinned ConvFinQA source, archive hash, paper, and MIT license.

The three generated datasets use fixture ID `example/finance-research` and
version `finance_fixture_v1`. The FinQA-derived dataset uses table ID
`finqa-public-test-v1` and pins upstream revision
`0f16e2867befa6840783e58be38c9efb9229d742`. The TAT-QA-derived dataset uses
table ID `tatqa-public-test-gold-v1` and pins revision
`870accc41953dcde885aabeb963d94aabdc0fbc3`. Each revision is recorded in every
corresponding benchmark row and in its fixture provenance.

The ConvFinQA multi-step file uses table ID `convfinqa-dev-v1` and pins
revision `cf3eed2d5984960bf06bb8145bcea5e80b0222a6`. Its conversation turns
and gold programs are preserved exactly. Only the mapping from each gold
program to the existing `finance_query_table` or `calculator` argument schema
is a LayerMCP adaptation.

Together, the original 15-row FinQA adaptation and the two expanded files cover
all 1,147 official test questions. The expansion uses table ID
`finqa-public-test-program-results-v1` and the same pinned revision
`0f16e2867befa6840783e58be38c9efb9229d742`. One gold operation maps to one
call: operation zero and table/comparison operations use
`finance_query_table`; later arithmetic operations use `calculator`.

The FinRetrieval workflow file pins revision
`86a111357cffa181b3ba0a6b5ce94625d4511176`. The importer selects 498 questions
with a released correct, executable multi-call trajectory, then retains the 485
whose selected trajectory has at most five calls. Questions 253 and 455 have no
eligible correct trajectory. Another 13 source indexes are omitted only because
their selected traces exceed the benchmark limit: 12, 29, 38, 75, 108, 122,
321, 322, 359, 377, 466, 480, and 486. The complete selected-trace replay
fixture is preserved; LayerMCP does not manufacture replacement traces.

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
- FinRetrieval replay tools accept only argument combinations present in the
  checked-in fixture. They return compact recorded results and provenance,
  reject unknown arguments, and perform no network access.

## Schema and Provenance

Each benchmark row follows the evaluator's current schema:

- `query` is the natural-language routing request.
- `prompt_context` is compact, parseable JSON containing the fixture grounding
  that is actually shown to the router after the unchanged query.
- `expected_tool` and `expected_args` are the routing and argument labels.
- `expected_answer` is a partial semantic oracle verified against the tool.
- `difficulty` and `perturbation_type` describe the controlled variation.

During evaluation, each row is exposed to the full MCP tool registry.

Adding the four FinRetrieval replay tools, together with five coding replay
tools, changed the full registry from 51 tools to 60. Previous full-registry
model results are stale and not directly comparable with current results.
Rerun them against the 60-tool registry before making model comparisons.

Finance results must be grouped by `benchmark_mode`. FinRetrieval coordinate
replay is `offline_trace_replay`; all other finance datasets are
`grounded_tool_execution`. Report these groups separately. Reproducing a
recorded Daloopa or web call coordinate does not demonstrate live retrieval or
independent finance reasoning.

A `multi_step_tool_routing` row instead provides an ordered `expected_steps`
list and uses evaluation protocol `teacher_forced_step_routing_v1`. Each step
contains a current-step instruction or exact gold operation, prompt context,
expected tool, arguments, partial expected answer, dependency IDs, and source
program/call metadata. The evaluator constructs every prompt from the overall
task, the gold current-step instruction and grounding context, and bounded gold
prior-step context: every declared dependency plus up to two other recent
steps. A prediction does not determine the next step's instruction or history.
The per-step and complete ordered-sequence metrics therefore measure
teacher-forced step routing, not autonomous planning or unconstrained
decomposition from only the top-level query. A workflow's published
`expected_final_answer` is retained in evaluation records, but the routing
evaluator does not synthesize or score an overall natural-language answer; its
executable semantic metric applies to individual tool outputs and the complete
output sequence.

Finance prompt contexts use four versioned JSON kinds:

- `finance_table_query_grounding_v1` supplies the valid dataset ID, the `data`
  table schema, source filters, and relevant normalized evidence rows. The
  compact FinQA operation-result fixture supplies schema and lookup coordinates
  but deliberately does not place the stored result in the prompt.
- `finance_calculator_call_grounding_v1` supplies the fully resolved expression
  for a paper-derived calculation step, including values that would otherwise
  depend on executing an earlier call.
- `finance_recorded_call_grounding_v1` exposes the exact normalized input from a
  selected FinRetrieval trace and explicitly labels it as offline trace replay,
  rather than something the model must infer from the research question.
- `finance_normalized_call_grounding_v1` supplies canonical fixture arguments
  for direct calls where aliases or opaque document IDs would make exact-label
  scoring ambiguous.

Every context is limited to 16,000 characters. Regenerate or validate all
checked-in contexts deterministically with:

```bash
python benchmark/finance/apply_grounding.py
python benchmark/finance/apply_grounding.py --check
```

Generated upstream-inspired rows use
`query_origin: generated_from_upstream_documentation` and include an inspiration
repository, URL, and reference. They are synthetic prompts, not copied public
queries.

The 15 FinQA rows retain the original public-test question and record the exact
split, zero-based row index, example ID, source program, execution answer,
revision, URL, copyright, license, and adaptation method. They come from the
official dataset released for the [FinQA paper](https://aclanthology.org/2021.emnlp-main.300/).

The 1,132-row FinQA expansion retains every remaining official question,
program, per-operation published result, execution answer, source coordinate,
revision, file hash, and license. Intermediate step prompts are exact canonical
gold operations, not invented dialogue. The compact fixture stores operation
results rather than copying the full annual-report context.

The 485 FinRetrieval benchmark rows retain official questions and ordered inputs
from selected correct model trajectories containing no more than five calls.
These paths are model-generated research traces, not expert-authored gold plans.
The replay fixture preserves all 498 selected source trajectories. Original tool
names, call IDs, configurations, output hashes, score status, and pinned parquet
hashes are retained. Daloopa aliases and three recorded web clients are
mechanically normalized to four offline LayerMCP tools.

The 15 TAT-QA rows retain the official test-gold question text unchanged,
including source punctuation and whitespace. Each row records its source
context and question indexes, table and question UIDs, derivation, answer,
scale, pinned revision and file hash, paper, and license. TAT-QA was introduced
in [Zhu et al., ACL-IJCNLP 2021](https://aclanthology.org/2021.acl-long.254/),
and its dataset is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The attribution file
describes the cell normalization and SQL adaptation.

These fixtures test finance tool routing, bounded SQL argument generation, and
ordered replay-call prediction. They do not reproduce live Daloopa/web
retrieval or the full numerical-reasoning task from the source papers.
FinanceBench is not bundled: its open release is CC BY-NC 4.0 and requires real
filing/PDF evidence outside the current synthetic filing fixture.

## Expansion Counts

The new finance-only batch contains:

| Source | Single-call | Multi-call | Total |
|---|---:|---:|---:|
| Remaining FinQA test questions | 642 | 490 | 1,132 |
| Correct FinRetrieval trajectories (at most five calls) | 0 | 485 | 485 |
| **New batch** | **642** | **975** | **1,617** |

Including the pre-existing finance datasets, the repository contains 1,757
finance workflows: 772 single-call and 985 multi-call.

The importers are deterministic and offline:

```bash
python benchmark/finance/build_finqa_expansion.py \
  --source-test /path/to/FinQA/dataset/test.json

python benchmark/finance/build_finretrieval_expansion.py \
  --questions /path/to/questions.parquet \
  --scores /path/to/scores.parquet \
  --tool-traces /path/to/tool_traces.parquet
```

The FinRetrieval importer requires DuckDB only while reading parquet source
files; DuckDB is not a LayerMCP runtime dependency.

## Run

From the repository root:

```bash
python evaluation/evaluate.py \
  --dataset benchmark/finance/finance_smoke.json

python evaluation/evaluate.py \
  --dataset benchmark/finance/finance_controlled.json

python evaluation/evaluate.py \
  --dataset benchmark/finance/finance_upstream_inspired.json

python evaluation/evaluate.py \
  --dataset benchmark/finance/finance_tatqa_public_derived.json

python evaluation/evaluate.py \
  --dataset benchmark/finance/finance_finqa_test_single.json

python evaluation/evaluate.py \
  --dataset benchmark/finance/finance_finqa_test_multistep.json

python evaluation/evaluate.py \
  --dataset benchmark/finance/finance_finretrieval_replay_multistep.json
```

Add `--call-predicted-tools` to execute the model's predicted calls.
