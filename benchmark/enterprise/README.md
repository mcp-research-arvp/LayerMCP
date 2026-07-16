# Enterprise Tool-Routing Datasets

This folder contains enterprise automation single-tool routing benchmarks. Each query asks for one business or retail action, and the expected answer identifies the tool and arguments the router should select.

## What v1 and v2 Mean

`v1` and `v2` are tool-suite versions, not random dataset splits.

- `v1` is the first small controlled enterprise fixture suite. It uses simple offline business tools such as customer lookup, order lookup, ticket routing, policy checks, and knowledge-base search.
- `v2` is the newer retail-style enterprise suite. It uses a frozen set of 12 retail tools adapted from tau3 Retail-style workflows, including user lookup, order/product inspection, order edits, returns, exchanges, and human transfer.

For future files, prefer names that say both the source and the version, such as:

```text
tool_routing_enterprise_controlled_v2.json
tool_routing_enterprise_public_adapted_v2.json
```

The current filenames are kept as-is for compatibility with tests and existing runs.

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

### `tool_routing_enterprise_v1.json`

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

### `tool_routing_enterprise_v2_controlled.json`

- Records: 48
- Source: `controlled_synthetic`
- Domain: `enterprise_automation`
- Purpose: controlled examples written to map retail-style enterprise queries to one of the 12 frozen retail tools.

Each tool has 4 controlled examples. The examples cover direct wording, distractors, paraphrases, and indirect requests.

Difficulty breakdown:

| Difficulty | Records |
| --- | ---: |
| `easy` | 12 |
| `medium` | 24 |
| `hard` | 12 |

### `tool_routing_enterprise_public_adapted.json`

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

Difficulty breakdown:

| Difficulty | Records |
| --- | ---: |
| `medium` | 22 |
| `hard` | 2 |

## Schema Notes

All enterprise files use the standard benchmark schema:

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

The public-adapted file also includes provenance fields. Tests in `tests/test_enterprise_v2_controlled_benchmark.py` enforce the v2 controlled/public schema, tool menu, and executable tool arguments.

New enterprise datasets should keep the same field names, use `enterprise_automation` as the domain, and only introduce a new version label when the tool menu or schema actually changes.

