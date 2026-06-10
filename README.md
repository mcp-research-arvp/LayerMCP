# LayerMCP# Layer-Aware Adaptation of Open-Source LLMs for MCP Tool Selection and Domain Expertization

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

## Getting Started

Phase 1 — the MCP benchmark harness (`benchmark/`) — is runnable now. Phases 2–4
land in `interpretability/`, `finetuning/`, and `experts/` as the project progresses.

```
# Repository structure
├── benchmark/        # Phase 1: MCP evaluation harness   (available)
├── interpretability/ # Phase 2: layer attribution tools  (planned)
├── finetuning/       # Phase 3: selective FT experiments (planned)
├── experts/          # Phase 4: domain specialist models (planned)
└── paper/            # LaTeX source for publications      (planned)
```

### Prerequisites

- **Git** and **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** — package/venv manager
  - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- A **model server with an OpenAI-compatible API**. The examples below use
  [llama.cpp](https://github.com/ggml-org/llama.cpp), but any OpenAI-compatible
  endpoint (vLLM, etc.) works — the harness only speaks HTTP.

### 1. Clone the repo and install the harness

**Linux / macOS**
```bash
git clone https://github.com/mcp-research-arvp/LayerMCP.git
cd LayerMCP
uv venv
source .venv/bin/activate
uv pip install -e ./benchmark
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/mcp-research-arvp/LayerMCP.git
cd LayerMCP
uv venv
.venv\Scripts\Activate.ps1
uv pip install -e .\benchmark
```

This installs the `mcpbench` CLI into the virtual environment.

### 2. Start a model server (llama.cpp)

> **`--jinja` is required** — without it, llama.cpp does not parse tool calls into
> the OpenAI `tool_calls` field and every task records an empty response.

**Linux / macOS — build from source**
```bash
sudo apt install -y cmake build-essential libcurl4-openssl-dev libopenblas-dev   # Debian/Ubuntu
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build ~/llama.cpp/build -j
```
Add `-DGGML_CUDA=ON` instead of the BLAS flags if you have an NVIDIA GPU.

**Windows — prebuilt binaries (easiest)**
Download the latest Windows build from the
[llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) — e.g.
`llama-*-bin-win-cpu-x64.zip` (CPU) or a `cuda` build (NVIDIA GPU) — and extract
it to, say, `C:\llama.cpp`.

**Download a model** (both OSes; `uv pip install huggingface_hub` first):
```bash
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir models
```

**Run the server:**

Linux / macOS
```bash
~/llama.cpp/build/bin/llama-server --model models/Qwen3-8B-Q4_K_M.gguf \
    --port 8080 --ctx-size 4096 --jinja
```
Windows (PowerShell)
```powershell
C:\llama.cpp\llama-server.exe --model models\Qwen3-8B-Q4_K_M.gguf `
    --port 8080 --ctx-size 4096 --jinja
```

### 3. Add tasks and run the harness

`benchmark/suites/` ships empty — drop in JSON task files (each file is an array of
tasks; the schema is the `Task` model in
[`benchmark/mcpbench/schema.py`](benchmark/mcpbench/schema.py)). Then run (with the
venv activated, on either OS):

```bash
mcpbench run --suite all --no-think                          # every suite in benchmark/suites/
mcpbench run --suite finance --no-think                      # just benchmark/suites/finance.json
mcpbench run --endpoint http://localhost:8080 --limit 5      # first 5 tasks only
```

Each run sends every task to the endpoint, records the model's tool choice and
telemetry (tokens, latency), prints a per-task line, and writes a JSON report to
`results/`. `--no-think` disables Qwen3's chain-of-thought for faster, cleaner
tool calls.

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
