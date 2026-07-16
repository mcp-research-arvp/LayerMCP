# Math Tool-Routing Datasets

This folder contains math-only single-tool routing benchmarks. Each query is paired with the math tool that should be called and the arguments that should be passed.

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

### `tool_routing_math_controlled.json`

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

### `tool_routing_math_public_derived.json`

- Records: 77
- Source: `public_math_derived`
- Source dataset: `math`
- Domain: `mathematics`
- Purpose: examples derived from public MATH benchmark problems and converted into MCP-style single-tool routing records.

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

This public-derived set currently does not include `convert_units` or `differentiate_expression` examples because the selected public MATH records were focused on arithmetic, algebra, number theory, and base arithmetic.

Difficulty comes from the public source levels:

| Difficulty | Records |
| --- | ---: |
| `level_1` | 14 |
| `level_2` | 17 |
| `level_3` | 23 |
| `level_4` | 17 |
| `level_5` | 6 |

## Schema Notes

Both math files use the standard benchmark schema:

- `id`
- `domain`
- `task_type`
- `difficulty`
- `source`
- `query`
- `available_tools`
- `expected_tool`
- `expected_args`
- `expected_answer`
- `perturbation_type`
- `notes`

The public-derived file also includes provenance fields. New math datasets should follow the same naming and schema style:

```text
tool_routing_math_<source>.json
```

For example, use `tool_routing_math_controlled_v2.json` only if the version marks a real tool-suite or schema change.

