# FinRetrieval Attribution and Adaptation Boundary

## Source

- Dataset: FinRetrieval
- Authors: Eric Y. Kim and Jie Huang
- Dataset repository: <https://huggingface.co/datasets/daloopa/finretrieval>
- Pinned revision:
  `86a111357cffa181b3ba0a6b5ce94625d4511176`
- Paper: <https://arxiv.org/abs/2603.04403>
- Dataset-card license declaration: MIT

Pinned artifact hashes:

- `questions.parquet`:
  `4f5a4b20d5163390502fd84a21c87581578341c97edbf2726177c7412b88c4a9`
- `scores.parquet`:
  `29eb5238e92153ce88bd5b68063d9e8aca4d4d74fa5107a9cc79e6da78fdc0b9`
- `tool_traces.parquet`:
  `96d15a4d9bc9f9effaa0b95edb87f52445207a2417d6339b18e6e79df920595c`

The pinned repository contains no separate license-notice file. LayerMCP records
the MIT declaration from the dataset card without inventing a copyright notice.

## Selection

The released dataset contains 500 questions and 14 model/configuration runs per
question. LayerMCP selects 498 source trajectories for which the release
contains a correct multi-call trajectory satisfying all of these conditions:

1. the score row has `is_correct = true`;
2. the trajectory contains at least two calls;
3. no selected call has `is_error = true`;
4. no selected output contains a recorded `Output validation error`; and
5. every source tool maps to one of the bounded offline tools described below.

For each question, the deterministic importer prefers an all-Daloopa trajectory,
then the fewest calls, the fewest web calls, and finally the lexicographically
first configuration name.

Questions 253 and 455 are excluded because the release contains no correct
trajectory for either one. LayerMCP does not relabel them as single-call examples
or manufacture replacement traces.

The executable benchmark retains 485 of those questions: only selected
trajectories containing two to five calls. Source indexes 12, 29, 38, 75, 108,
122, 321, 322, 359, 377, 466, 480, and 486 are omitted because their selected
trajectories contain more than five calls. Their records remain in the replay
fixture so the pinned source adaptation and any shared replay results are not
discarded.

## Mechanical Adaptation

- Official question text and ordered source tool inputs are retained for every
  benchmark workflow.
- Daloopa client aliases are normalized to:
  - `finance_discover_companies`
  - `finance_discover_company_series`
  - `finance_get_company_fundamentals`
- `WebSearch`, `google_search`, and `google_search_agent` calls are normalized to
  `finance_search_web_archive`.
- Original source tool names, inputs, call IDs, configurations, and canonical
  output hashes remain attached to benchmark steps.
- Recorded outputs are compacted into a checked-in replay fixture. Large company
  series responses retain the entries needed by later calls; web results are
  bounded excerpts.
- Runtime tools perform no network, browser, or Daloopa requests.

The selected trajectories are successful model-generated traces from the
research release. They are not expert-authored gold tool plans. The LayerMCP
multi-step evaluator is teacher-forced per expected step, so this adaptation
tests ordered tool and argument prediction rather than an unconstrained
autonomous agent loop.
