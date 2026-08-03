# Math Tool-Routing Datasets

This folder contains math-only single-step and teacher-forced multi-step routing
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
- Source: `controlled_synthetic`
- Domain: `mathematics`
- Purpose: deterministic sequencing coverage across existing math tools.

Every workflow contains two or three connected calls. Later calls consume a
resolved result from an earlier call, such as an arithmetic product passed to
integer factorization, a derivative set equal to zero and solved, or a
converted magnitude used in final arithmetic. Step-level `prompt_context`
provides authoritative current-step inputs for teacher-forced routing. Rebuild
the artifact with:

```bash
python benchmark/math/build_multistep_controlled.py
```

## Schema Notes

Math datasets use the standard benchmark schema:

- `id`
- `domain`
- `task_type`
- `difficulty`
- `source`
- `query`
- `expected_tool`
- `expected_args`
- `expected_answer`
- `perturbation_type`
- `notes`

The public-derived file also includes provenance fields. New math datasets should follow the same naming and schema style:

```text
math_<source_or_purpose>.json
```

Current examples are `math_public.json`, `math_public_math_dataset.json`,
`math_controlled.json`, and `math_multistep_controlled.json`.
