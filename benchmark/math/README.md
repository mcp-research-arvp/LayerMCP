# Math Tool-Routing Datasets

This folder contains math-only single-step and guided multi-step routing
benchmarks. Each expected call identifies the math tool, arguments, and
deterministic tool result.

## Tools Covered

The math tool menu is implemented in `mcp_server/math_tools.py`.

| Tool | What it is for |
| --- | --- |
| `calculator` | Direct numeric expression evaluation. |
| `simplify_expression` | Symbolic simplification. |
| `solve_equation` | Solving an equation for a named variable. |
| `factor_expression` | Symbolic factorization. |
| `expand_expression` | Symbolic expansion. |
| `differentiate_expression` | Symbolic differentiation. |
| `convert_units` | Unit conversion. |
| `integer_factorization` | Prime factorization of an integer expression. |
| `gcd_lcm` | Greatest common divisor, least common multiple, or both. |
| `modular_arithmetic` | Modular residues, modular powers, and modular inverses. |
| `base_arithmetic` | Arithmetic in non-decimal bases. |

## Dataset Files

### `math_controlled.json`

- Records: 51
- Source: `controlled_synthetic`
- Domain: `mathematics`
- Purpose: controlled examples written specifically to map a math query to one relevant tool and expected argument shape.

Breakdown by expected tool:

| Tool | Records |
| --- | ---: |
| `calculator` | 5 |
| `simplify_expression` | 5 |
| `solve_equation` | 5 |
| `factor_expression` | 5 |
| `expand_expression` | 5 |
| `differentiate_expression` | 5 |
| `convert_units` | 5 |
| `integer_factorization` | 4 |
| `gcd_lcm` | 4 |
| `modular_arithmetic` | 4 |
| `base_arithmetic` | 4 |

The early controlled records use difficulty labels such as `easy`, `medium`, and `hard`. The newer v2 controlled records use more targeted IDs such as `math_v2_controlled_modular_arithmetic_001`.

### `math_public_math_dataset.json`

- Records: 77
- Source: `public_math_derived`
- Source dataset: `math`
- Domain: `mathematics`
- Purpose: executable examples derived from public MATH benchmark problems and converted into MCP-style single-tool routing records.

Each record keeps provenance fields such as:

- `source_dataset`
- `source_row_index`
- `source_category`
- `source_level`

Breakdown by expected tool:

| Tool | Records |
| --- | ---: |
| `calculator` | 10 |
| `simplify_expression` | 10 |
| `solve_equation` | 10 |
| `factor_expression` | 8 |
| `expand_expression` | 10 |
| `integer_factorization` | 8 |
| `gcd_lcm` | 8 |
| `modular_arithmetic` | 8 |
| `base_arithmetic` | 5 |

### `math_public.json`

- Records: 400
- Sources: pinned DeepMind Mathematics Dataset and GSM8K
- Domain: `mathematics`
- Purpose: tool-balanced, executable public-derived single-step coverage.

The file contains 100 GSM8K calculator rows and 300 deterministically generated
DeepMind rows. Every record has a non-null expected answer, source revision,
stable source coordinate, canonical source hash, license, and transformation
notes. DeepMind records additionally store the module, generation seed, and
accepted generated index. See
`fixtures/PUBLIC_EXPANSION_ATTRIBUTION.md` and
`build_public_expansion.py`.

Breakdown by expected tool:

| Tool | Records |
| --- | ---: |
| `calculator` | 120 |
| `simplify_expression` | 25 |
| `solve_equation` | 35 |
| `factor_expression` | 15 |
| `expand_expression` | 25 |
| `differentiate_expression` | 40 |
| `convert_units` | 40 |
| `integer_factorization` | 25 |
| `gcd_lcm` | 25 |
| `modular_arithmetic` | 20 |
| `base_arithmetic` | 30 |

This public-derived set currently does not include `convert_units` or `differentiate_expression` examples because the selected public MATH records were focused on arithmetic, algebra, number theory, and base arithmetic.

Difficulty comes from the public source levels:

| Difficulty | Records |
| --- | ---: |
| `level_1` | 14 |
| `level_2` | 17 |
| `level_3` | 23 |
| `level_4` | 17 |
| `level_5` | 6 |

### `math_multistep_controlled.json`

- Workflows: 50
- Expected steps: 105

Each workflow uses `exact_normalized_json` for its final-step outcome contract.
The evaluator compares the complete normalized final tool result to
the expected object, including the exact set of object keys. This diagnostic is
reported separately from tool selection, argument accuracy, and all-step
outcome metrics; the guided protocol does not score a user-facing final answer.
- Source: `controlled_synthetic`
- Domain: `mathematics`
- Task type: `multi_step_tool_routing`
- Purpose: deterministic sequencing coverage across existing math tools.

Every workflow contains two or three connected calls. Later calls consume a
resolved result from an earlier call, such as an arithmetic product passed to
integer factorization, a derivative set equal to zero and solved, or a
converted magnitude used in final arithmetic. Step-level `prompt_context`
provides authoritative current-step inputs for guided rollout. Rebuild
the artifact with:

```bash
python benchmark/math/build_multistep_controlled.py
```

This controlled synthetic set is an optional diagnostic. It is not the primary
public-derived Mathematics baseline and is excluded from automatic headline
runs.

## Public-derived MathQA multi-step benchmark

> A deterministic, source-ID-preserving multi-step LayerMCP benchmark derived from the executable operation programs in the public MathQA test split.

- Workflows: 200 selected from a strict eligible population of 1,498
- Expected steps: 892 selected from 6,638 eligible source-program operations
- Source split and identity: original MathQA `test.json`, `test:<zero-based-row-index>`
- Classification: public-derived deterministic program adaptation
- Official site: <https://math-qa.github.io/math-QA/>
- Paper: <https://aclanthology.org/N19-1245/>
- Archive: <https://math-qa.github.io/math-QA/data/MathQA.zip>
- Pinned AllenAI mirror revision: `c4f1cc784c04c4957b50c97858f23893b633eea6`

| Input | SHA-256 |
| --- | --- |
| `MathQA.zip` | `7344f30456a7aef3176d4866cc953b35b41bec44eda6b00cdbcfde2876b2f07a` |
| `test.json` | `dfe7bc4691caf26842ccd4cf14e8f978327bab4f2989e0b076a4e6b38a9371d1` |
| `train.json` | `00e8919347d65dbba9289bf04ed998a6c48dbf451ca909eeb66a35f2419c2bf6` |
| `dev.json` | `aeca12424fc8f32d0e35c09b7738986a446c12f110c8071f3cc8913712f04bf3` |
| `operation_list.txt` | `3ea53c71034e52605f8203f1525d7f03bd4a0ddc3e4384c93b3ee1e4ddc1e8ca` |
| `constant_list.txt` | `611b5d1089f0d10e6486b5ffeaee92c95efc5df9e385b809d4d142181e2bb4a0` |

The official ZIP contains no embedded license file. Apache-2.0 is the license
declared by the pinned AllenAI dataset card; the manifest records that caveat.
The complete source archive remains outside tracked repository content.

The builder maps `linear_formula` mechanically. It preserves each source row,
coordinate, canonical row hash, source formulas, operation order, backward
references, selected option, and raw DSL calls. No rationale is parsed and no
model creates, rewrites, repairs, or expands a step or dependency. Eligibility
requires two or more exactly parsed operations; supported `nK`, `const_*`, and
backward `#K` arguments; valid destination-tool inputs; finite execution; final
numeric agreement at relative and absolute tolerance `1e-9`; and no exact
question overlap with train/development. `power(base, exponent)` always uses the
actual resolved source exponent. The existing LayerMCP calculator accepts
exponents with magnitude at most 10; rows outside that bound are excluded as
`destination_tool_exponent_limit`, never altered. The Google Trax executor is
corroborating downstream evidence only. Its deviations do not override the
MathQA `linear_formula`, `annotated_formula`, rationale, or standard operation
meaning. The build fails unless the corrected strict population is exactly
1,498 workflows and 6,638 operations.

The exhaustive exclusion classes for the 1,487 rejected test rows are: numeric
option parsing limitation 35; strict-tolerance mismatch 226; annotated-versus-
linear formula disagreement 1; program result matching another option 81;
destination-tool constraints 16; unsupported operations 63; unresolved source
program/selected-answer disagreement 802; fewer than two operations 254; and
nonnumeric selected option 9. Invalid/forward references, nonfinite/tool
failures, and train/development overlap each have count zero. The unresolved
rows are excluded without asserting that either their program or answer key is
wrong and without repairing or interpreting them.

The destination-tool constraints comprise exponent limit 9, division by zero
3, and negative square-root input 4. Unsupported operations are `choose` 19,
`factorial` 21, `log` 9, `max` 3, `min` 2, `negate_prob` 2, `permutation` 1,
`square_edge_by_perimeter` 1, `surface_cylinder` 1,
`triangle_area_three_edges` 2, `volume_cone` 1, and `volume_sphere` 1. Every
excluded row retains its source coordinate, canonical row hash, exact reason,
and broader exclusion class in the manifest.

Selection is stratified by source category and program-length bucket (`2`, `3`,
`4`, `5`, `6-7`, `8-10`, `11+`). It uses proportional quotas, floor allocation,
then descending fractional remainder with lexicographic stratum-key ties. Rows
within each stratum are ranked by SHA-256 of coordinate plus canonical row hash,
and selected rows are finally restored to source order. There is no manual,
tool-diversity, or performance-based selection.

Selected category counts are gain 43, general 85, geometry 11, other 10,
physics 50, probability 1. Program lengths are 2:34, 3:52, 4:38, 5:26, 6:21,
7:10, 8:5, 9:6, 10:3, 11:1, 12:2, 13:1, 18:1. Operations are add 155, divide
272, floor 2, inverse 7, lcm 2, multiply 262, negate 2, power 15,
rectangle_area 1, reminder 2, speed 2, sqrt 3, subtract 166, and
volume_rectangular_prism 1. Tools are calculator 888, gcd_lcm 2, and
modular_arithmetic 2. The manifest contains full eligible distributions,
rejection details, stratum sizes, quotas, allocations, and all selected hashes.

The audited mapping sends `gcd`/`lcm` to `gcd_lcm` and upstream `reminder` to
`modular_arithmetic(operation="mod")`. Fixed `calculator` templates implement
`add`, `subtract`, `multiply`, `divide`, `power`, `negate`, `inverse`, `floor`,
`sqrt`, `speed`, rectangle/square/cube, volume/surface, circle, and circumference
operations listed in `mathqa_operation_mapping.json`. Pi is always
`3.141592653589793`. Factorial, choose/combination, permutation, logarithms,
min/max, trigonometry, `surface_cylinder`, and unlisted operations are excluded.
No LayerMCP tool was added.

For the 888 calculator steps, `expected_args.expression` remains the exact
mechanical translation of the MathQA reference program. Exact Argument Match
therefore measures fidelity to that source program. Calculator outcome gold
contains only the computed `result`; Step Outcome and Final Step Outcome use
recursive subset matching so an equivalent expression with the same value is
correct even though its argument diagnostic is different. The four structured
non-calculator steps retain their complete semantic result objects, and changed
semantic fields remain incorrect. No fuzzy expression matcher is used.

```bash
python benchmark/math/build_mathqa_public_multistep.py \
  --source-archive /path/to/MathQA.zip
python benchmark/math/build_mathqa_public_multistep.py \
  --source-archive /path/to/MathQA.zip --check
```

| Artifact | SHA-256 |
| --- | --- |
| benchmark | `6c6c14b431ecedbbba6002ef7324716703e27f739ab207374dd16b0f294515d2` |
| selected fixture | `33b3371165ba15f49a2a9f995a57e55b788481f070ecdfc375bd359b23ad58c0` |
| mapping | `3cb5eb2c46c948a25d16eb3e50e6aa57ea5655bfacf653b0291fde6dc011bd2f` |
| manifest | `ca49dad10675634acfab981ab697051de48cc3fe34362f53691d23c4d0e00a1e` |
| selected-source subset | `4c23c53096819c128a11f1496e6eb24cb99797a7eec22cd62a1dcc0a35f9311e` |
| generation | `df5109e5fcd2b6267b73dfa85edbd6f769eb3f8692da72821737ddb19c6ecfa7` |

`expected_final_answer` preserves the published option as provenance only.
Evaluation scores final-step and all-step tool outcomes, not a separate
user-facing answer. Final-step outcome reuses the final step's recursive subset
outcome score; it does not compare the calculator's echoed expression. This is
not an unmodified official MathQA score: LayerMCP filters rows and translates
the public DSL into its tool interfaces. Launch as
`DATASET_GROUP=math_public_mathqa`; artifacts live at
`domains/math/math_public_mathqa_multistep/` and report as `MathQA
public-derived`. Controlled Math remains a separately reported diagnostic.

MathQA is the primary public-derived multi-step Mathematics outcome and
dependency benchmark, but it is not an official MathQA score. It measures
guided execution of published operation programs rather than autonomous
decomposition. Because 888 of 892 selected steps map to `calculator`, tool
selection is highly imbalanced and must not be a headline claim. Report
argument accuracy, step outcome, all-step correctness, and final-step outcome
prominently. Keep `math_controlled` as a separately labeled short diagnostic
for broader Mathematics tool-family routing, never combine its score with
MathQA, and do not rebalance the public subset by tool because that would
distort the source-derived distribution.

## Schema Notes

All Math datasets use these common top-level fields:

- `id`
- `domain`
- `task_type`
- `difficulty`
- `source`
- `query`
- `perturbation_type`
- `notes`

The three single-step datasets store the expected call and result directly on
each row:

- `expected_tool`
- `expected_args`
- `expected_answer`

The multi-step datasets use `expected_steps`; every
step contains:

- `expected_tool`: the tool selected for that step.
- `expected_args`: the deterministic arguments for that call.
- `expected_answer`: the expected result from executing that call.
- `prompt_context`: authoritative current-step inputs and sequencing context.
  For a dependent step, the evaluator withholds this field when it could reveal
  a gold-resolved prior value and instead exposes predicted rollout history.
- `depends_on`: links to prior step IDs whose outputs or results feed the
  current operation. Independent prerequisite steps may have an empty list.

The public-derived file also includes provenance fields. New math datasets should follow the same naming and schema style:

```text
math_<source_or_purpose>.json
```

Current examples are `math_public.json`, `math_public_math_dataset.json`,
`math_controlled.json`, `math_multistep_controlled.json`, and
`math_public_mathqa_multistep.json`.
