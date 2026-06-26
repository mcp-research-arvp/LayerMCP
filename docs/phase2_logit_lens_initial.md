# Phase 2 Initial Logit Lens

This document describes the first Phase 2 mechanistic-interpretability experiment for LayerMCP: layer-wise tool-label tracking with a logit lens.

## What Logit Lens Measures

Logit lens projects intermediate hidden states through the model output head. For each layer, it asks: if the model had to predict the next token from this layer's representation, which candidate label would be favored?

In this project, each MCP tool is assigned a short label such as `A`, `B`, `C`, or `D`. For every benchmark example, the analysis records the layer-wise logits for those labels and tracks when the correct tool label becomes preferred.

## Observational, Not Causal

This analysis is observational. It shows whether information about the final tool choice is visible in intermediate states, but it does not prove that a layer causes the tool choice.

Causal claims require later experiments such as activation patching, ablation, attribution patching, or controlled interventions. Those are intentionally not implemented in this step.

## Why This Is Phase 2 Step 1

The Phase 1 scaffold now has deterministic tools, controlled benchmarks, and structured evaluation. The first Phase 2 step should be lightweight and diagnostic before more invasive methods are added.

Logit lens is useful here because it can:

- show where tool-choice evidence begins to separate by layer
- compare easy examples against same-domain confusions
- identify examples worth probing or patching later
- produce simple CSV/plot artifacts for repeated runs

## Why Short Labels Are Used

The analysis uses short labels instead of full tool names because full tool names can tokenize into multiple pieces. Multi-token tool names make layer-wise next-token comparisons harder to interpret.

The prompt maps tools to labels:

```text
A. calculator
B. unit_converter
C. stock_price_api
```

The script then compares logits for the single-token label choices. It checks that each label is represented as one token, preferring the leading-space form used after `Answer:`.

## Metrics Produced

For every sample and layer, the CSV includes:

- `correct_label_rank`: rank of the correct tool label among candidate labels
- `correct_minus_best_wrong_logit`: correct-label logit minus the strongest incorrect-label logit
- `predicted_label_at_layer`: highest-logit label at that layer
- `predicted_label_at_final_layer`: highest-logit label at the final layer
- `correct_label_logit`
- `best_wrong_label`
- `best_wrong_label_logit`

Positive `correct_minus_best_wrong_logit` means the correct label beats every wrong label at that layer. Negative values mean at least one distractor label is stronger.

## Connection To Later Methods

This step prepares later Phase 2 work:

- Linear probes can test whether tool labels are decodable from hidden states with a trained classifier.
- Activation patching can test whether swapping hidden states changes tool choice.
- Ablation can test whether suppressing layers or components harms correct routing.
- Selective fine-tuning can later target layers implicated by causal and observational evidence.

Logit lens alone should be treated as a guide for where to look next, not as the final evidence.

## How To Run

Run a small smoke analysis:

```powershell
python analysis/logit_lens.py --benchmark benchmark/tool_routing_phase2_seed.json --max-examples 4
```

Run with plotting:

```powershell
python analysis/logit_lens.py --benchmark benchmark/tool_routing_phase2_seed.json --max-examples 4 --plot
```

Override the model:

```powershell
$env:LAYERMCP_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
python analysis/logit_lens.py --benchmark benchmark/tool_routing_phase2_seed.json
```

Outputs are saved under `results/`:

- `logit_lens_<timestamp>.csv`
- `logit_lens_<timestamp>_summary.json`
- `logit_lens_<timestamp>.png` when `--plot` is passed

## Interpreting Easy And Confusable Examples

For easy-routing examples, the correct label should ideally become top-ranked earlier and maintain a positive margin through later layers.

For same-domain confusion examples, such as `calculator` vs `unit_converter`, `github_search` vs `read_code_file`, and `customer_lookup` vs `ticket_router`, the correct label may separate later or have a smaller margin.

For known-failure examples, inspect whether the wrong label dominates across many layers or only emerges near the final layers. Persistent wrong-label dominance suggests the representation may encode the wrong routing decision early; late reversals suggest the final decoding behavior may be unstable or prompt-sensitive.
