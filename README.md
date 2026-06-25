# LayerMCP

> Investigating whether transformer layer subsets drive MCP tool-routing and domain reasoning — and whether selectively fine-tuning those layers can replace full-model adaptation.

![Status](https://img.shields.io/badge/status-active%20research-blue)
![Timeline](https://img.shields.io/badge/timeline-6%20months-informational)
![Target Venues](https://img.shields.io/badge/venues-NeurIPS%20%7C%20ICLR%20%7C%20ICML%20%7C%20ACL%20%7C%20EMNLP%20%7C%20AAAI-purple)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

Modern open-source LLMs are capable of tool use, function calling, and domain-specific reasoning — but little is known about *where* inside the network these capabilities reside. This project tests the hypothesis that **MCP tool-selection and domain-specialization behaviors are concentrated in a small, identifiable subset of transformer layers**, rather than being uniformly distributed across the entire model. If true, it becomes possible to create efficient domain experts by modifying only those layers — avoiding the expense of full fine-tuning while matching or exceeding its quality.

The project spans mechanistic interpretability, efficient fine-tuning, and agentic evaluation, applied to four open-source model families across four high-value domains.

---

## Research Questions

1. **Localization** — Are MCP tool-selection and domain-reasoning behaviors concentrated in specific transformer layers, attention heads, or MLP blocks, and does this vary across model architectures?
2. **Selective FT efficacy** — Can interpretability-guided, layer-selective fine-tuning match full fine-tuning, LoRA, and QLoRA at lower computational cost and parameter budget?
3. **Cross-domain generalization** — Does selective layer adaptation produce genuine domain experts while preserving general reasoning (i.e., avoiding catastrophic forgetting)?

---

## Project Phases

| Phase | Duration | Goal |
|-------|----------|------|
| **1 — MCP Benchmark** | Months 1–2 | Build a standardized evaluation harness for tool selection, function calling, and domain reasoning across finance, coding, math, and enterprise workflows |
| **2 — Layer Attribution** | Months 2–4 | Instrument models with mechanistic interpretability techniques to localize tool-routing and domain-reasoning behavior to specific layers/heads |
| **3 — Selective Fine-Tuning** | Months 3–5 | Experimentally fine-tune only the identified layers and compare against full FT, LoRA, and QLoRA on quality, compute, memory, and speed |
| **4 — Domain Experts** | Months 4–6 | Produce lightweight specialist models for finance, software engineering, mathematics, and cybersecurity; evaluate retention of general capability |

---

## Models Studied

| Model Family | Architecture | Notes |
|---|---|---|
| **GPT-OSS** | Mixture-of-Experts (MoE) | Layer localization includes expert routing analysis |
| **Gemma** (Google) | Dense decoder | Multiple sizes; clean baseline |
| **Qwen** (Alibaba) | Dense decoder | Strong multilingual & coding baselines |
| **Llama** (Meta) | Dense decoder | Llama 3.x series; widely studied |

---

## Domains

- **Quantitative Finance** — instrument pricing, risk calculation, market data retrieval
- **Software Engineering** — code generation, tool-augmented debugging, repo navigation
- **Mathematics** — multi-step symbolic and numerical reasoning
- **Cybersecurity / Enterprise Automation** — policy lookup, secure API orchestration, workflow automation

---

## Methods & Techniques

### Mechanistic Interpretability
- **Activation patching / causal intervention** — swap activations between contrastive input pairs to prove causal contribution of specific layers
- **Representation probing** — linear classifiers on hidden states to test decodability of tool-choice at each layer
- **Gradient attribution** — score component importance by gradient signal toward the tool-selection output
- **Attention analysis** — identify heads that attend to tool descriptions, function signatures, and schema tokens

### Fine-Tuning Approaches (compared)
- **Full fine-tuning** — update all weights; expensive upper bound
- **LoRA** — low-rank adapters injected uniformly; ~0.1–1% parameters
- **QLoRA** — LoRA over 4-bit quantized model; fits large models on a single GPU
- **Selective-layer FT** *(proposed)* — interpretability-guided update of identified layers/heads/MLPs only

---

## Benchmarks & Metrics

### Related Benchmarks Used as Baselines
- [BFCL (Berkeley Function-Calling Leaderboard)](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [ToolBench](https://github.com/OpenBMB/ToolBench)
- [τ-bench / τ²-bench](https://github.com/sierra-research/tau-bench)
- [API-Bank](https://github.com/AmbitionXiang/API-Bank)

### Metrics Collected
| Metric | Description |
|--------|-------------|
| **Tool-selection accuracy** | Correct tool chosen (name + arguments) |
| **Hallucinated-tool rate** | Calls to non-existent tools |
| **Execution success rate** | Tool call produces correct runtime result |
| **Reasoning quality** | Correctness of intermediate reasoning steps |
| **Token efficiency** | Tokens consumed per successful task completion |
| **Latency** | Wall-clock time to first tool call and full completion |

---

## Expected Contributions

- **MCP Benchmarking Framework** — reusable evaluation harness for tool-calling and domain reasoning, released openly
- **Layer-Attribution Infrastructure** — tooling for activation patching and probing on dense and MoE transformers
- **Interpretability Findings** — per-architecture maps of where tool-selection and domain reasoning live
- **Selective-Layer FT Method** — efficient adaptation technique guided by interpretability findings
- **Domain Expert Models** — lightweight specialists for finance, coding, math, and cybersecurity
- **Publications** — targeting NeurIPS, ICLR, ICML, ACL, EMNLP, AAAI

---

## Timeline

```
Month:  1       2       3       4       5       6
        |-------|-------|-------|-------|-------|
Phase 1 [===Benchmark Build===]
Phase 2         [========Layer Attribution========]
Phase 3                 [=====Selective FT Exps=====]
Phase 4                         [===Domain Experts===]
Write-up                                 [==========]
```

---

## Team

Graduate research team of 3–4 students with the following role coverage:

- **Benchmarking & Evaluation** — MCP harness, metric design, baseline comparisons
- **Mechanistic Interpretability** — activation patching, probing, attention analysis
- **Fine-Tuning & Training** — LoRA / QLoRA / selective FT infrastructure, compute management
- **Domain Expert Adaptation** — per-domain dataset curation, catastrophic-forgetting evaluation

---

---

## Getting Started

The research direction above is still the intent of the project, but the current runnable
repo structure is the local MCP tool-routing prototype below.

### Current Repository Structure

```text
LayerMCP/
├── benchmark/
│   ├── __init__.py
│   └── tool_routing.jsonl
├── evaluation/
│   ├── __init__.py
│   ├── dataset.py
│   └── evaluate.py
├── mcp_server/
│   ├── __init__.py
│   ├── server.py
│   └── tool_impls.py
├── models/
│   ├── __init__.py
│   └── qwen_router.py
├── tests/
│   ├── test_dataset.py
│   ├── test_evaluate_helpers.py
│   ├── test_qwen_router.py
│   └── test_tool_impls.py
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

### Prerequisites

- **Git** and **Python 3.10+**
- Enough RAM/VRAM to load `Qwen/Qwen2.5-3B-Instruct` when running model-backed evaluation
- Optional `HF_TOKEN` for faster Hugging Face downloads and higher rate limits

### 1. Clone the Repo and Install the Project

**Linux / macOS**

```bash
git clone https://github.com/mcp-research-arvp/LayerMCP.git
cd LayerMCP
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Windows PowerShell**

```powershell
git clone https://github.com/mcp-research-arvp/LayerMCP.git
cd LayerMCP
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

This installs the lightweight runtime and developer tools, then registers:

- `layermcp-server`
- `layermcp-evaluate`

Install model training/evaluation dependencies only when you need to load Hugging Face models:

```bash
pip install -e ".[train]"
```

Install vLLM support separately for fast post-training inference:

```bash
pip install -e ".[serve]"
```

You can combine extras when setting up a full research environment:

```bash
pip install -e ".[dev,train]"
```

Run the no-model smoke tests with:

```bash
python -m unittest discover -s tests
```

### 2. Start the MCP Server

Run the server directly:

```bash
python -m mcp_server.server
```

Or use the installed entrypoint:

```bash
layermcp-server
```

The server will usually appear to do nothing. That is expected. It is a stdio MCP server, so it waits silently for a client connection.

### 3. Run the Evaluation Harness

The evaluator starts the MCP server automatically. You do not need to start `mcp_server\server.py` first for evaluation runs.

Evaluate routing only:

```bash
python -m evaluation.evaluate
```

Evaluate routing and execute the predicted MCP tool for each sample:

```bash
python -m evaluation.evaluate --call-predicted-tools
```

Or use the installed entrypoint:

```bash
layermcp-evaluate --call-predicted-tools
```

The evaluator uses `models.qwen_router` by default. Install the `train` extra before running model-backed evaluation.

### 4. Available CLI Flags

- `--dataset <path>` -- use a different benchmark JSONL file
- `--server <path>` -- use a different MCP server entrypoint
- `--call-predicted-tools` -- execute the predicted tool with `expected_arguments`
- `--help` -- show the built-in CLI help

### 5. Current MCP Tools

The server currently exposes:

- `calculator`
- `customer_lookup`
- `github_search`

These are deterministic offline fixtures for research and testing:

- `calculator` safely evaluates simple arithmetic
- `customer_lookup` returns deterministic mock customer data
- `github_search` returns deterministic mock GitHub-style results without calling the live GitHub API

### 6. Benchmark Format

The benchmark file is `benchmark/tool_routing.jsonl`. Each line is one JSON
object:

```json
{"id":"math_easy_multiply_001","domain":"math","query":"What is 25 * 17?","tools":["calculator","customer_lookup","github_search"],"expected_tool":"calculator","expected_arguments":{"expression":"25 * 17"},"expected_result":{"expression":"25 * 17","result":425},"difficulty":"easy"}
```

Required fields:

- `id` -- unique stable sample identifier
- `domain` -- research domain label such as `math`, `enterprise`, or `software_engineering`
- `query` -- user-facing request
- `tools` -- tool names available for that sample
- `expected_tool` -- expected tool name, or `hallucinated_tool` when no tool should be called
- `expected_arguments` -- expected tool arguments as a JSON object
- `expected_result` -- expected deterministic tool result, or `null` for no-tool samples
- `difficulty` -- one of `easy`, `medium`, or `hard`

The loader also accepts legacy JSON-list files for migration, but JSONL is the
default format for larger benchmark runs.

### 7. Runtime Flow

1. `evaluation/evaluate.py` launches `mcp_server/server.py` as a child process.
2. The MCP client connects over stdio and calls `initialize`.
3. The evaluator calls `list_tools` to get the live tool catalog from the server.
4. Each benchmark row is validated against the richer JSONL schema.
5. The evaluator checks that the row's listed tools exist on the server.
6. The router predicts one tool name from the row's `tools` list.
7. If `--call-predicted-tools` is enabled, the evaluator calls the predicted tool with the sample's `expected_arguments`.

### Notes

- The evaluation path no longer uses a hardcoded static tool list.
- The router defaults to `Qwen/Qwen2.5-3B-Instruct`. You can override that with the `LAYERMCP_MODEL_NAME` environment variable.
- If the model is not already cached locally, the first run will download it from the Hugging Face Hub.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{layeraware2026,
  title   = {Layer-Aware Adaptation of Open-Source LLMs for MCP Tool Selection and Domain Expertization},
  author  = {},
  year    = {2026},
  note    = {Work in progress}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
