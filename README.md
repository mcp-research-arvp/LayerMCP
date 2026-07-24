# LayerMCP: Layer-Aware Adaptation of Open-Source LLMs for MCP Tool Selection and Domain Expertization

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
│   ├── coding/
│   │   ├── fixtures/
│   │   ├── README.md
│   │   ├── tool_routing_coding_controlled.json
│   │   ├── tool_routing_coding_codesearchnet_public_derived.json
│   │   ├── tool_routing_coding_smoke.json
│   │   └── tool_routing_coding_upstream_inspired.json
│   ├── finance/
│   │   ├── fixtures/
│   │   ├── README.md
│   │   ├── tool_routing_finance_controlled.json
│   │   ├── tool_routing_finance_public_derived.json
│   │   ├── tool_routing_finance_smoke.json
│   │   ├── tool_routing_finance_tatqa_public_derived.json
│   │   └── tool_routing_finance_upstream_inspired.json
│   └── tool_routing.json
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py
├── mcp_server/
│   ├── __init__.py
│   ├── coding_state.py
│   ├── coding_tools.py
│   ├── finance_state.py
│   ├── finance_tools.py
│   ├── server.py
│   └── tool_impls.py
├── models/
│   ├── __init__.py
│   ├── routers/
│   │   ├── qwen_hf_router.py
│   │   └── gpt_oss_local_router.py
│   └── architectures/
│       └── gpt_oss_pytorch/
├── .gitignore
├── pyproject.toml
└── README.md
```

### Prerequisites

- **Git** and **Python 3.10+**
- **ripgrep** for `code_search_text`
- Enough RAM/VRAM to load the router you choose
- Optional `HF_TOKEN` for faster Hugging Face downloads and higher rate limits

### 1. Clone the Repo and Install the Project

**Windows (PowerShell)**

```powershell
git clone https://github.com/mcp-research-arvp/LayerMCP.git
cd LayerMCP
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

This installs the dependencies from `pyproject.toml` and registers:

- `layermcp-server`
- `layermcp-evaluate`

### 2. Start the MCP Server

Run the server directly:

```powershell
python mcp_server\server.py
```

Or use the installed entrypoint:

```powershell
layermcp-server
```

The server will usually appear to do nothing. That is expected. It is a stdio MCP server, so it waits silently for a client connection.

### 3. Run the Evaluation Harness

The evaluator starts the MCP server automatically. You do not need to start `mcp_server\server.py` first for evaluation runs.

Evaluate routing only:

```powershell
python evaluation\evaluate.py
```

Evaluate routing and execute the predicted MCP tool for each sample:

```powershell
python evaluation\evaluate.py --call-predicted-tools
```

Or use the installed entrypoint:

```powershell
layermcp-evaluate --call-predicted-tools
```

Choose a router backend explicitly:

```powershell
layermcp-evaluate --router qwen-hf
layermcp-evaluate --router gpt-oss-local
layermcp-evaluate --router phi-4-local
layermcp-evaluate --router llama-3.1-8b-local
layermcp-evaluate --router qwen-3.6-local
layermcp-evaluate --router gemma-4-local
```

Router naming:

- `qwen-hf` uses Hugging Face Transformers for both the architecture loader and Qwen weights.
- `gpt-oss-local` uses the local PyTorch GPT-OSS architecture in `models/architectures/gpt_oss_pytorch/` and local checkpoint files.
- `phi-4-local` uses the local PyTorch Phi-4 text-backbone architecture in `models/architectures/phi4_pytorch/` and local checkpoint files.
- `llama-3.1-8b-local` uses the local PyTorch Llama 3.1 8B Instruct architecture in `models/architectures/llama31_8b_pytorch/` and local checkpoint files.
- `qwen-3.6-local` uses the local PyTorch Qwen 3.6 text architecture in `models/architectures/qwen36_pytorch/` and a local Hugging Face-format checkpoint.
- `gemma-4-local` uses the local PyTorch Gemma 4 text architecture in `models/architectures/gemma4_pytorch/` and a local Hugging Face-format checkpoint.

### GPT-OSS Checkpoints

Downloaded weights should not be committed. By default, the GPT-OSS local router looks for:

```text
checkpoints/gpt-oss-20b/original/
```

You can download into the ignored `checkpoints/` directory:

```powershell
mkdir checkpoints
hf download openai/gpt-oss-20b --local-dir checkpoints/gpt-oss-20b
```

If your checkpoint lives somewhere else, set:

```powershell
$env:LAYERMCP_GPT_OSS_CHECKPOINT = "path\to\gpt-oss-20b\original"
```

### PHI-4 Checkpoints

By default, the PHI-4 local router looks for a Hugging Face-format checkpoint at:

```text
checkpoints/phi-4/
```

The directory should contain `config.json`, tokenizer files, and `.safetensors` shards. If your checkpoint lives somewhere else, set:

```powershell
$env:LAYERMCP_PHI4_CHECKPOINT = "path\to\phi-4"
```

### Llama 3.1 8B Instruct Checkpoints

By default, the Llama 3.1 8B Instruct local router looks for a Hugging Face-format checkpoint at:

```text
checkpoints/llama-3.1-8b-instruct/
```

The directory should contain tokenizer files and `.safetensors` shards. If your checkpoint lives somewhere else, set:

```powershell
$env:LAYERMCP_LLAMA31_8B_CHECKPOINT = "path\to\llama-3.1-8b-instruct"
```

### Qwen 3.6 Checkpoints

By default, the local Qwen 3.6 router looks for a Hugging Face-format checkpoint at:

```text
checkpoints/qwen-3.6/
```

The directory must contain `config.json`, tokenizer files, and `.safetensors` shards. To use another location:

```powershell
$env:LAYERMCP_QWEN36_CHECKPOINT = "path\to\qwen-3.6"
```

### Gemma 4 Checkpoints

By default, the local Gemma 4 router looks for a Hugging Face-format checkpoint at:

```text
checkpoints/gemma-4/
```

The directory must contain `config.json`, tokenizer files, and `.safetensors` shards. To use another location:

```powershell
$env:LAYERMCP_GEMMA4_CHECKPOINT = "path\to\gemma-4"
```

### 4. Available CLI Flags

- `--dataset <path>` -- use a different benchmark JSON file
- `--server <path>` -- use a different MCP server entrypoint
- `--router <name>` -- choose `qwen-hf`, `qwen-3.6-local`, `gemma-4-local`, `gpt-oss-local`, `phi-4-local`, or `llama-3.1-8b-local`
- `--call-predicted-tools` -- execute the predicted tool with arguments generated by the router
- `--help` -- show the built-in CLI help

### 5. Current MCP Tools

The server exposes deterministic offline tools across mathematics, enterprise,
Retail, coding, and finance domains. The coding tool catalog is:

- `code_list_files` — list bounded regular files by repository path and glob
- `code_read_file` — read a bounded UTF-8 line range
- `code_search_text` — fixed-string lexical search backed by ripgrep
- `git_log` — retrieve history reachable from the pinned fixture snapshot
- `git_show` — inspect one reachable commit and its patch
- `git_diff` — compare reachable commits or local branches, or inspect the worktree
- `git_status` — inspect bounded branch, index, worktree, and untracked state

The generated coding datasets use the allowlisted repository ID
`example/research-mcp`. That repository is created lazily from deterministic
files and three fixed commits.
Paths are repository-relative, `.git` access and symlinks are rejected, Git
revisions are restricted to the pinned history, and outputs are capped. These
seven tools are read-only.

A second allowlisted repository, `codesearchnet-public-v1`, contains a narrow
MIT-licensed adaptation of 15 exact CodeSearchNet human-evaluation queries and
their selected annotation records. It contains no target source code and is
explicitly a lexical tool-routing fixture rather than a reproduction of the
paper's semantic retrieval evaluation. Benchmark prompts wrap the exact source
queries in self-contained repository-search instructions and preserve the
verbatim text separately as `original_query`.

The older `github_search` and `read_code_file` fixtures remain registered for
backward compatibility with existing benchmark files.

The finance tool catalog is:

- `finance_lookup_company` — look up fixture companies by ticker, CIK, name, or alias
- `finance_search_filings` — filter bounded filing metadata by company, form, and year
- `finance_get_filing_section` — retrieve a bounded filing section
- `finance_get_company_facts` — retrieve normalized company facts
- `finance_get_financial_statement` — retrieve a normalized financial statement
- `finance_parse_xbrl` — parse facts from a server-owned XBRL instance
- `finance_query_table` — run bounded read-only SQL over an allowlisted table
- `finance_extract_pdf_tables` — retrieve pre-extracted tables for selected PDF pages
- `finance_get_market_quote` — retrieve the latest synthetic OHLCV quote
- `finance_get_market_time_series` — retrieve a bounded synthetic daily series

The main finance fixture uses fictional companies and synthetic filings, XBRL,
PDF tables, and market snapshots. It is offline and read-only. Two pinned
paper-dataset adaptations supply 30 executable public-derived table queries: 15
from FinQA and 15 from the CC BY 4.0 TAT-QA test-gold release. See
`benchmark/finance/README.md` for the exact runtime boundaries, attribution, and
provenance.

### 6. Benchmark Format

The default benchmark file is `benchmark/tool_routing.json`. The coding-specific
datasets are:

- `benchmark/coding/tool_routing_coding_smoke.json` — 7 direct examples, one per coding tool
- `benchmark/coding/tool_routing_coding_controlled.json` — 35 balanced controlled examples
- `benchmark/coding/tool_routing_coding_upstream_inspired.json` — 28 generated queries grounded in official upstream usage documentation
- `benchmark/coding/tool_routing_coding_codesearchnet_public_derived.json` — 15 self-contained lexical-search instructions preserving exact CodeSearchNet queries in `original_query`
- `benchmark/coding/tool_routing_coding_sweagent_multistep.json` — 1 exact SWE-bench issue with 3 ordered read-only actions from an official SWE-agent trajectory

See `benchmark/coding/README.md` for their scope, provenance, and run commands.
The finance-specific datasets are:

- `benchmark/finance/tool_routing_finance_smoke.json` — 10 direct examples, one per finance tool
- `benchmark/finance/tool_routing_finance_controlled.json` — 50 balanced controlled examples
- `benchmark/finance/tool_routing_finance_upstream_inspired.json` — 40 generated queries grounded in official upstream documentation
- `benchmark/finance/tool_routing_finance_public_derived.json` — 15 executable public-test adaptations from FinQA
- `benchmark/finance/tool_routing_finance_tatqa_public_derived.json` — 15 exact TAT-QA test-gold questions with executable SQL adaptations
- `benchmark/finance/tool_routing_finance_convfinqa_multistep.json` — 3 exact ConvFinQA conversations containing 12 ordered paper-authored turns

See `benchmark/finance/README.md` for their data boundaries, upstream mappings,
provenance, and run commands.
Each current-format benchmark item looks like:

```json
[
  {
    "id": "coding_smoke_code_list_files_001",
    "domain": "coding",
    "task_type": "single_tool_routing",
    "difficulty": "easy",
    "source": "controlled_synthetic",
    "query": "In example/research-mcp, list all repository files.",
    "available_tools": [
      "code_list_files",
      "code_read_file",
      "code_search_text",
      "git_log",
      "git_show",
      "git_diff",
      "git_status"
    ],
    "expected_tool": "code_list_files",
    "expected_args": {
      "repo_id": "example/research-mcp"
    },
    "expected_answer": {
      "count": 6,
      "truncated": false
    },
    "perturbation_type": "easy_direct",
    "notes": "Smoke coverage for bounded repository file listing."
  }
]
```

`expected_args` is the exact argument-generation label. With
`--call-predicted-tools`, the evaluator executes the router's predicted tool and
predicted arguments; it does not substitute the expected arguments.

### 7. Runtime Flow

1. `evaluation/evaluate.py` launches `mcp_server/server.py` as a child process.
2. The MCP client connects over stdio and calls `initialize`.
3. The evaluator calls `list_tools` to get the live tool catalog from the server.
4. The router predicts one tool name from that live catalog.
5. If `--call-predicted-tools` is enabled, the evaluator calls the predicted tool with the router's predicted arguments.

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
