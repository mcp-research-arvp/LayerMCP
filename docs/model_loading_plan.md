# Model Loading Plan

LayerMCP uses Hugging Face Transformers on top of PyTorch for Phase 2 because
the research questions depend on internal model state, not only final text
generation. Ollama is useful for convenient local serving, but its API is built
around prompting and responses. For layer attribution, the project needs direct
access to tensors and modules inside the network.

## Why PyTorch Access Matters

Phase 2 work needs access to:

- hidden states from each transformer layer
- output logits before decoding
- layer activations
- attention and MLP modules
- hooks for recording, patching, ablating, or comparing intermediate values

These signals are the raw material for logit-lens analysis, probing, causal
interventions, and later selective fine-tuning decisions. A text-serving
runtime can answer routing prompts, but it generally cannot expose the
per-layer tensors and module boundaries needed for mechanistic analysis.

## Current Default

The current default model is:

```text
Qwen/Qwen2.5-3B-Instruct
```

The default can be overridden with:

```text
LAYERMCP_MODEL_NAME
```

Both the router and analysis scripts should load models through
`models/model_loader.py` so device placement, dtype, quantization, and error
handling remain consistent.

## Future Model Targets

The shared loader is intended to support experiments across:

- GPT-OSS 20B
- Gemma
- Qwen
- Llama
- DeepSeek

The first implementation should prefer Hugging Face model IDs whenever they
provide complete PyTorch access. Model-specific PyTorch loading paths should be
added only when Hugging Face access is insufficient for the experiment.

## Reference Material

Tony's GPT-OSS and Gemma PyTorch links are useful as implementation references
for model-specific internals, especially if the Hugging Face wrappers hide
details that matter for activation capture or module-level interventions. They
should guide any future custom loading or architecture-specific inspection, not
replace the shared loader by default.

Karpathy's GPT-2 reproduction video is useful conceptually because it walks
through a transformer from raw PyTorch components to training behavior. For
LayerMCP, that helps the team reason about embeddings, residual streams,
attention blocks, MLPs, logits, and where hooks or probes should attach. It is
not a direct implementation dependency.

## Hardware And Memory Limits

The project should assume constrained local hardware. Small dense models should
come first because they make debugging and attribution runs fast enough to
iterate. Larger models, including 20B-scale targets, will likely require:

- `device_map="auto"` through Accelerate
- float16 or bfloat16 loading on CUDA
- 8-bit or 4-bit quantized loading when bitsandbytes is available
- careful batch sizes and short prompts for attribution runs
- explicit avoidance of generated model cache or result files in git

Quantization is a loading strategy for feasibility, not a replacement for
understanding the unquantized model. Whenever quantization changes the research
question, runs should be labeled clearly.

## Expected Loading Strategy

1. Start with small models that fit comfortably in local RAM or VRAM.
2. Use Hugging Face Transformers and PyTorch as the default path.
3. Use CUDA auto placement and reduced precision when available.
4. Use bitsandbytes quantization for larger models when installed.
5. Fall back safely to non-quantized loading if optional quantization libraries
   are missing.
6. Add model-specific PyTorch paths only when Hugging Face access does not
   expose the internals needed for the experiment.

This keeps Phase 1 routing simple while giving Phase 2 the instrumentation
surface needed for layer-aware research.
