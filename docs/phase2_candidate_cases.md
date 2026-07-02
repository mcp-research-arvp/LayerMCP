# Phase 2 Candidate Cases

This note summarizes candidate examples and layer regions for the next Phase 2
step: activation patching. It is based on the existing local logit-lens run:

- CSV: `results/logit_lens_20260625T001102576538Z.csv`
- Summary: `results/logit_lens_20260625T001102576538Z_summary.json`
- Benchmark: `benchmark/tool_routing_phase2_seed.json`
- Model: `Qwen/Qwen2.5-3B-Instruct`

No activation patching has been implemented yet.

## Candidate Late-Layer Region

The strongest average correct-minus-best-wrong margins occur in late layers:

| Layer | Average margin | Layer prediction accuracy |
|---:|---:|---:|
| 32 | 4.3089 | 75.00% |
| 33 | 6.1080 | 81.25% |
| 34 | 6.4894 | 81.25% |
| 35 | 7.4565 | 81.25% |
| 36 | 4.9815 | 87.50% |

Recommended first patching window: layers 32-36, with layers 33-35 as the
highest-priority region.

Why this region is useful:

- The average margin sharply improves after layer 31.
- Most successful examples separate strongly in this window.
- The two final-layer failures become strongly wrong in the same late region,
  which makes them good candidates for clean/corrupt intervention tests.

## Strongest Successful Examples

These examples have correct final-layer predictions and the largest final-layer
margins.

| Sample | Domain | Focus | Correct tool | Final margin | Strongest layer |
|---|---|---|---|---:|---:|
| `phase2_finance_002` | finance | `cross_domain_distractor` | `stock_price_api` | 10.2948 | 34 |
| `phase2_enterprise_003` | enterprise automation | `easy_routing` | `customer_lookup` | 9.6478 | 34 |
| `phase2_coding_004` | coding | `same_domain_confusion` | `github_search` | 8.0607 | 34 |
| `phase2_finance_001` | finance | `easy_routing` | `stock_price_api` | 7.7278 | 33 |
| `phase2_enterprise_001` | enterprise automation | `easy_routing` | `customer_lookup` | 7.2517 | 35 |

Why these are useful:

- They provide clean runs where the correct label is strongly represented.
- Their strongest layers fall in the late candidate region.
- They can act as donors for activation patching into weaker or failing cases.

## Weakest And Failure Examples

| Sample | Domain | Focus | Correct tool | Final correct | Final margin | Notes |
|---|---|---|---|---|---:|---|
| `phase2_enterprise_004` | enterprise automation | `known_failure` | `ticket_router` | no | -4.9830 | Correct label briefly appears earlier, then loses strongly to the final wrong label. |
| `phase2_finance_004` | finance | `cross_domain_distractor` | `stock_price_api` | no | -3.4441 | Same target tool as successful finance examples, but late layers favor the wrong label. |
| `phase2_math_001` | mathematics | `easy_routing` | `calculator` | yes | 0.6287 | Correct final prediction, but weak final margin. |
| `phase2_math_003` | mathematics | `easy_routing` | `calculator` | yes | 1.3815 | Correct final prediction appears late and remains relatively weak. |

Why these are useful:

- The two failures are natural corrupt examples for testing whether late-layer
  activations can recover the correct tool label.
- The weak math successes are useful controls: they are correct, but fragile.
- Comparing failure cases with weak successes can help separate "wrong route"
  behavior from low-confidence correct routing.

## Known-Failure Example

Primary known-failure case:

- `phase2_enterprise_004`
- Query: "This is not a customer lookup; route a phishing security report."
- Expected tool: `ticket_router`
- Final predicted label: wrong
- Final margin: -4.9830
- Weakest late-layer margin: layer 34 at -18.9229

Why this is useful:

- It is explicitly marked `known_failure`.
- The candidate correct label is briefly recoverable at earlier layers, including
  small positive margins around layers 15, 22, 23, and 24.
- Late layers strongly amplify the wrong final label, making it a good target for
  patching late residual-stream activations from a clean `ticket_router` case.

## Recommended Clean/Corrupt Pairs

### Enterprise Ticket Routing

- Clean: `phase2_enterprise_002`
  - Query asks to route a billing ticket.
  - Correct tool: `ticket_router`
  - Final margin: 6.2017
  - Strongest layer: 35
- Corrupt: `phase2_enterprise_004`
  - Query asks to route a phishing security report.
  - Correct tool: `ticket_router`
  - Final prediction is wrong.
  - Final margin: -4.9830

Why this pair:

- Same domain.
- Same expected tool.
- Same broad `customer_lookup` vs `ticket_router` confusion family.
- The clean case separates strongly in the late candidate region while the
  corrupt case collapses there.

### Finance Stock Lookup

- Clean: `phase2_finance_002`
  - Query asks for Apple/AAPL price and explicitly rejects arithmetic.
  - Correct tool: `stock_price_api`
  - Final margin: 10.2948
  - Strongest layer: 34
- Corrupt: `phase2_finance_004`
  - Query asks for TSLA market data from the offline quote fixture.
  - Correct tool: `stock_price_api`
  - Final prediction is wrong.
  - Final margin: -3.4441

Why this pair:

- Same domain.
- Same expected tool.
- Both are cross-domain distractor cases.
- The clean case has one of the strongest final margins in the run, while the
  corrupt case turns strongly wrong in late layers.

### Coding Search Versus File Read

- Clean search donor: `phase2_coding_004`
  - Correct tool: `github_search`
  - Final margin: 8.0607
  - Strongest layer: 34
- Clean file-read donor: `phase2_coding_002`
  - Correct tool: `read_code_file`
  - Final margin: 6.7330
  - Strongest layer: 34

Why this pair:

- These are both successful same-domain confusion examples.
- They test whether the late-layer representation distinguishes two coding
  tools with similar surface context.
- They are useful after the first failure-recovery patches, as a finer-grained
  same-domain control pair.

### Math Fragility Control

- Strong unit-conversion case: `phase2_math_004`
  - Correct tool: `unit_converter`
  - Final margin: 6.8022
  - Strongest layer: 34
- Weak calculator case: `phase2_math_001`
  - Correct tool: `calculator`
  - Final margin: 0.6287
  - Strongest layer: 32

Why this pair:

- Both are final-correct, so this is not a failure-recovery pair.
- It can test whether patching high-confidence math routing activations into a
  weak correct example increases confidence without flipping the route.

## Recommended Patch Order

1. Start with `phase2_enterprise_002` -> `phase2_enterprise_004` on layers 32-36.
2. Repeat with `phase2_finance_002` -> `phase2_finance_004` on layers 32-36.
3. Narrow to layers 33-35 if the broad patch window has an effect.
4. Use coding and math control pairs to check whether effects are specific to
   tool choice rather than generic confidence.

Expected signal:

- If late-layer residual activations are causally involved, patching clean
  activations into corrupt examples should improve the correct-label margin or
  flip the final predicted label.
- If margins do not move, the logit-lens visibility may be observational only,
  or the relevant causal site may be a narrower component such as an attention
  head or MLP block.
