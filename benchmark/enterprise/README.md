# Enterprise Tool-Routing Datasets

This folder contains enterprise automation single-tool routing benchmarks. Each query asks for one business or retail action, and the expected answer identifies the tool and arguments the router should select.

## What v1 and v2 Mean

`v1` and `v2` are tool-suite versions, not random dataset splits.

- `v1` is the first small controlled enterprise fixture suite. It uses simple offline business tools such as customer lookup, order lookup, ticket routing, policy checks, and knowledge-base search.
- `v2` is the newer retail-style enterprise suite. Its primary executable
  benchmark uses 12 stable retail tool names backed by the pinned tau2 retail
  database, including user lookup, order/product inspection, order edits,
  returns, exchanges, and human transfer.

The standardized active enterprise files are:

```text
enterprise_controlled.json
enterprise_tau2_single_step.json
```

The retired datasets are archived as
`../archive/enterprise/enterprise_v2_controlled_legacy.json` and
`../archive/enterprise/enterprise_public_adapted_legacy.json`.

## Tools Covered

### v1 controlled tools

| Tool | Records |
| --- | ---: |
| `customer_lookup` | 5 |
| `get_order` | 5 |
| `update_order_status` | 5 |
| `create_support_ticket` | 5 |
| `ticket_router` | 5 |
| `search_knowledge_base` | 5 |
| `check_policy` | 5 |

### v2 retail tools

| Tool | Controlled records | Public-adapted records |
| --- | ---: | ---: |
| `find_user_id_by_email` | 4 | 2 |
| `find_user_id_by_name_zip` | 4 | 2 |
| `get_user_details` | 4 | 2 |
| `get_order_details` | 4 | 2 |
| `get_product_details` | 4 | 2 |
| `cancel_pending_order` | 4 | 2 |
| `modify_pending_order_items` | 4 | 2 |
| `modify_pending_order_address` | 4 | 2 |
| `modify_user_address` | 4 | 2 |
| `return_delivered_order_items` | 4 | 2 |
| `exchange_delivered_order_items` | 4 | 2 |
| `transfer_to_human_agents` | 4 | 2 |

## Dataset Files

### `enterprise_controlled.json`

- Records: 35
- Source: `controlled_synthetic`
- Domain: `enterprise_automation`
- Purpose: first controlled enterprise benchmark, using simple deterministic offline fixtures.

Difficulty breakdown:

| Difficulty | Records |
| --- | ---: |
| `easy` | 16 |
| `medium` | 12 |
| `hard` | 7 |

### `../archive/enterprise/enterprise_v2_controlled_legacy.json`

- Records: 48
- Source: `controlled_synthetic`
- Domain: `enterprise_automation`
- Purpose: controlled examples written to map retail-style enterprise queries to one of the 12 frozen retail tools.

Each tool has 4 controlled examples. The examples cover direct wording, distractors, paraphrases, and indirect requests.

These rows retain IDs from the retired small LayerMCP fixture and are kept for
historical routing-only comparison. They are not the primary executable retail
benchmark after the tau2-native migration.

Difficulty breakdown:

| Difficulty | Records |
| --- | ---: |
| `easy` | 12 |
| `medium` | 24 |
| `hard` | 12 |

### `../archive/enterprise/enterprise_public_adapted_legacy.json`

- Records: 24
- Source: `public_adapted`
- Source dataset: `tau3_retail`
- Raw source location: `data/raw/tau3_retail`
- Domain: `enterprise_automation`
- Purpose: examples adapted from public tau3 Retail tasks into single-tool routing examples.

Each v2 retail tool has 2 public-adapted examples. Records include provenance fields such as:

- `source_dataset`
- `source_domain`
- `source_task_id`
- `source_action`
- `provenance_type`

One hand-audited row per retail tool (12 rows total) includes a concise
`expected_answer` subset verified against deterministic gold-tool execution.
The remaining public-adapted rows retain `expected_answer: null`.

These rows also retain retired small-fixture IDs and are preserved as
historical routing-only data.

Difficulty breakdown:

| Difficulty | Records |
| --- | ---: |
| `medium` | 22 |
| `hard` | 2 |


### `enterprise_public_workflows.json`

- Rows: 69
- Task type: multi-step tool routing
- Source dataset: pinned tau2-bench retail tasks
- Source splits: 45 train workflows, 24 test workflows
- Purpose: source-faithful public Enterprise workflow evaluation using original tau2 retail user-scenario fields.

This benchmark differs from `enterprise_tau2_single_step.json`. The single-step file extracts individual tau2 gold actions into standalone executable requests. This workflow file preserves original tau2 scenario fields and keeps only fully supported, fully executable multi-action workflows against the pinned LayerMCP tau2 retail fixture.

Each row contains `expected_steps` from the tau2 evaluation criteria actions.
Evaluation is teacher-forced gold-action routing, not autonomous end-to-end
planning: every step supplies a natural-language operation and authoritative
step-level source facts while excluding earlier and later actions. Step outputs
are deterministic LayerMCP retail-tool results from a fresh pinned retail
fixture.

### `enterprise_tau2_single_step.json`

- Records: 293
- Source: pinned tau2-bench retail tasks
- Domain: `enterprise_automation`
- Purpose: one deduplicated, independently executable tau2-native gold retail
  action per row.

The source action name, arguments, and native tau2 entity IDs are preserved;
no entity remapping is applied. Every row records the source task/action IDs,
action index, split, original arguments, canonical task hash, committed fixture
hash, license, transformation notes, and a non-null expected-answer subset
verified by deterministic double execution. Rows are unique by
`expected_tool` plus canonical `expected_args`.
See `fixtures/TAU2_RETAIL_ATTRIBUTION.md` and
`build_tau2_retail_expansion.py`.

The three `get_item_details` actions and one
`modify_pending_order_payment` action are excluded because those low-coverage
tools are not registered in LayerMCP. Thirteen tau2 `calculate` actions remain
outside the retail tranche rather than being relabeled as retail tools.

## Schema Notes

All enterprise files use the standard benchmark schema:

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

The public-adapted file also includes provenance fields. Tests in
`tests/test_enterprise_v2_controlled_benchmark.py` enforce the v2
controlled/public schema and executable tool arguments.

New enterprise datasets should keep the same field names and use
`enterprise_automation` as the domain.
