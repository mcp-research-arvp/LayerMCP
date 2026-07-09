# LayerMCP

LayerMCP is a research-oriented MCP tool-selection benchmark and evaluation
harness. Its current job is to supply deterministic tools, tool schemas,
benchmark samples, prompts, expected tool labels, scoring, and analysis outputs
for experiments on model tool routing.

The repository is intentionally not a custom model architecture repo. Future
DeepSeek and layer-modification work should live in Tony's ModelSurgery repo.
LayerMCP should evaluate those models once they can return tool calls through
an API or adapter.

## Current Scope

LayerMCP currently supports two runnable research tracks:

1. **Phase 1: MCP tool-routing evaluation**
   - Starts a local stdio MCP server.
   - Discovers the live tool catalog from that server.
   - Runs a router against benchmark samples.
   - Scores predicted tool name vs. `expected_tool`.
   - Optionally calls the predicted offline fixture tool with expected args.
   - Writes JSONL sample records and JSON summary files under `results/`.

2. **Phase 2: logit-lens layer-wise analysis**
   - Loads a Hugging Face causal LM with hidden states enabled.
   - Converts each tool-choice sample into a forced-choice label task.
   - Projects hidden states through the LM head layer by layer.
   - Tracks correct-tool label separation from the strongest wrong label.
   - Writes CSV, JSON summary, optional PNG plot, and sample-summary CSV outputs.

## Current Limitations

- The current tools are deterministic offline fixtures, not live integrations.
- Current accuracy is mainly **tool-selection accuracy**.
- Argument generation and argument scoring are not fully implemented yet.
- External/public tool-use datasets are not integrated yet.
- The default router is still the Hugging Face Qwen router.
- DeepSeek architecture implementation belongs in Tony's ModelSurgery repo, not
  in LayerMCP.
- LayerMCP will evaluate DeepSeek later once a ModelSurgery model/API can return
  normalized tool calls.

## Repository Layout

```text
LayerMCP/
├── analysis/
│   └── logit_lens.py
├── benchmark/
│   ├── tool_routing.json
│   ├── tool_routing_smoke.json
│   ├── tool_routing_controlled.json
│   └── tool_routing_phase2_seed.json
├── docs/
├── evaluation/
│   └── evaluate.py
├── mcp_server/
│   ├── server.py
│   └── tool_impls.py
├── models/
│   ├── model_loader.py
│   ├── qwen_router.py
│   ├── router_interface.py
│   └── model_surgery_router.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Deterministic Offline MCP Tools

The local MCP server currently exposes these fixture tools:

- `calculator`
- `customer_lookup`
- `github_search`
- `stock_price_api`
- `unit_converter`
- `read_code_file`
- `ticket_router`

They are designed for reproducible routing experiments and tests. They do not
call external APIs.

## Benchmark Files

- `benchmark/tool_routing.json` - small legacy/simple routing benchmark.
- `benchmark/tool_routing_smoke.json` - lightweight smoke benchmark using the
  newer schema.
- `benchmark/tool_routing_controlled.json` - controlled Phase 1 benchmark.
- `benchmark/tool_routing_phase2_seed.json` - Phase 2 seed set for logit-lens
  experiments across finance, mathematics, coding, and enterprise automation.

Benchmark samples center on:

- `query`
- `available_tools`
- `expected_tool`
- `expected_args` or legacy `tool_args`
- metadata such as domain, difficulty, source, and perturbation type

## Model Loading And Default Router

The default router is `models/qwen_router.py`.

By default it loads:

```text
Qwen/Qwen2.5-3B-Instruct
```

You can override the model with:

```powershell
$env:LAYERMCP_MODEL_NAME = "some/model-name"
```

Model loading is centralized in `models/model_loader.py`, which handles:

- default model resolution
- `LAYERMCP_MODEL_NAME`
- dtype selection
- optional quantization config
- Hugging Face tokenizer/model loading

The lightweight router contract is documented in `models/router_interface.py`:

```python
choose_tool(user_query: str, available_tools: list[dict]) -> str
```

Existing code still uses the Qwen router as the default. The interface is there
to keep future adapters, including ModelSurgery, aligned without forcing a large
class-based rewrite.

## Running Phase 1 Evaluation

Install the package:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Run the evaluator:

```powershell
python evaluation\evaluate.py
```

Or with the console script:

```powershell
layermcp-evaluate
```

Evaluate a specific benchmark:

```powershell
python evaluation\evaluate.py --dataset benchmark\tool_routing_smoke.json
```

Evaluate DeepSeek ModelSurgery on a tiny smoke benchmark:

```powershell
$env:MODEL_SURGERY_REPO = "C:\Users\rishi\OneDrive\Desktop\Rishi's Stuff\Learning\Application_Templates\application-templates"
$env:DEEPSEEK_CHECKPOINT = "C:\path\to\deepseek-llm-7b-base"
$env:DEEPSEEK_ROUTER_TOKENS = "4"
python evaluation\evaluate.py --router deepseek_model_surgery --dataset benchmark\tool_routing_smoke.json
```

Evaluate the Phase 2 seed set the same way, keeping the first run small because
DeepSeek is slow on CPU:

```powershell
python evaluation\evaluate.py --router deepseek_model_surgery --dataset benchmark\tool_routing_phase2_seed.json
```

Optionally call the predicted MCP tool using the expected args stored in the
benchmark sample:

```powershell
python evaluation\evaluate.py --call-predicted-tools
```

The evaluator starts the MCP server automatically. You do not need to run
`mcp_server/server.py` separately for normal evaluation runs.

## Running Phase 2 Logit-Lens Analysis

Run the seed analysis:

```powershell
python analysis\logit_lens.py --max-examples 2
```

Run with plotting:

```powershell
python analysis\logit_lens.py --plot --group-by phase2_focus
```

Outputs are written under `results/`.

## ModelSurgery Integration Plan

Tony's ModelSurgery repo should supply the modifiable model implementation:

- Hugging Face baseline validation
- custom PyTorch architecture
- safetensors/checkpoint loading
- inference wrapper
- OpenAI-style `/v1/chat/completions` API
- normalized `tool_calls`
- direct layer access for interventions

LayerMCP should supply:

- benchmark samples
- available tool schemas
- prompts
- expected tool labels
- scoring and result files
- before/after comparison of baseline vs. modified-layer runs

The easiest first integration is API-based:

```text
LayerMCP benchmark sample
-> available tools
-> ModelSurgery /v1/chat/completions
-> normalized tool_calls
-> LayerMCP compares selected tool with expected_tool
```

Direct Python integration may be useful later when experiments need direct
access to layers for randomization, activation patching, or fine-tuning. Those
interventions should happen in ModelSurgery; LayerMCP should remain the
benchmark and scoring harness.

See:

- `docs/model_surgery_integration_plan.md`
- `docs/deepseek_implementation_plan.md`
- `docs/latest_deepseek_readiness_update.md`

## Development Notes

Run lightweight tests:

```powershell
python -m unittest discover -s tests
```

Generated outputs in `results/` are ignored by git.

## License

MIT License. See `LICENSE` if present.
