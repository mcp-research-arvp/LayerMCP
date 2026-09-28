# Phase 2: Llama activation observability

This is a deliberately small, read-only replay harness for one saved
`llama-3.1-8b-local` **direct** tool-routing result.  It reconstructs the
router's native chat-template prompt from the saved query, ordered tool names,
and the evaluator's live MCP catalog; it then makes one ordinary generation and
one teacher-forced observation pass.  The observation pass saves only selected
tool-name and argument key/value token positions from configured layers:

- the transformer block output (`residual_stream`);
- the self-attention block output (`attention_block`); and
- the MLP block output (`mlp_block`).

It never changes the model, evaluator, benchmark, or saved source artifact.
The hooks only copy selected tensors to CPU and are removed even if replay
raises an exception.  No full attention maps or all-token activation tensors
are written.
Set `observation_enabled: false` in a config to make the same ordinary replay
without installing any hooks or saving activation tensors.

Run from the repository root with a local Llama checkpoint available.  The
checked-in development config is deliberately portable: pass local paths on
the command line (or copy it locally and replace its placeholders):

```bash
python -m research.phase2.observe \
  --config research/phase2/configs/llama_direct_development.json \
  --source-run-dir /path/to/saved_llama_direct_run \
  --checkpoint /path/to/llama_checkpoint \
  --output-dir /path/to/new_observation_output
```

Use `LAYERMCP_LLAMA31_8B_CHECKPOINT`, `--checkpoint`, or the config's optional
`checkpoint` field to choose the custom-runtime checkpoint.  The development config points
to an older saved artifact deliberately marked `development_only`; its recorded
source commit is retained in the config and it is not headline benchmark
evidence.  Prompt tokenization uses the router's `encode_chat` method, including
its native-template fallback.  A prompt is exact only if its saved and live MCP
registry metadata match.  The generated `provenance.json` makes this visible
through `registry_exact_match`.

Each enabled-observation output directory contains `provenance.json`, selected
tensors in `activations.pt`, and `OBSERVATION_COMPLETE`. The provenance records
the exact parsed tool-call character span and, for every selected generated
token, its causal-LM input position. These are pre-token predictor states:
generated token zero is listed but intentionally not captured because its
predictor state is the final prompt token. A parse error, unknown tool, or
invalid arguments produces no observation directory or completion marker.
Disabled observation records provenance and `OBSERVATION_COMPLETE`, but
deliberately writes no activation tensor file. These outputs are inputs for
later **read-only probing** and, only after a separate approved design,
causal/intervention experiments.  This harness does not implement training,
LoRA/QLoRA, activation patching, or ablation.

## Temporary attention-parameter intervention

`research.phase2.intervention` provides a separate, in-memory context manager
for a one-layer attention experiment.  It does not alter the evaluator,
architecture, or checkpoint files.  It supports an identifiable decoder layer
only when it has one of these tested layouts: `model.layers[i].self_attn`,
`model.model.layers[i].self_attn` (or `.linear_attn` for the local Qwen hybrid),
or the local GPT-OSS `model.block[i].attn`.  The local Llama transformer is
instantiated in the unit tests without a checkpoint; the other repository
runtimes and Hugging Face causal-LM wrappers are accepted only when their
loaded instance presents one of those layouts.  Other layouts, ambiguous
attention attributes, quantized 4-bit/8-bit models, and non-floating-point or
cross-component-tied attention parameters fail before any parameter is changed.

The valid indices come from the supplied model rather than a fixed model-size
assumption.  `noise` sets every selected attention parameter `p` to
`p + strength * z`; `replace` sets it to `strength * z`; in both cases `z` is a
seeded standard-normal tensor with the same shape.  Generated values are cast
back to the parameter's original device and dtype.  The original tensors are
copied back exactly when the context exits, including after an exception.

For the local Llama runtime used by Phase 2, obtain the already-loaded model
through the same local checkpoint path as `observe.py`:

```python
from models.architectures.llama31_8b_pytorch.config import Config
from models.architectures.llama31_8b_pytorch.inference import TokenGenerator
from research.phase2.intervention import (
    available_attention_layers,
    temporary_attention_intervention,
)

checkpoint_dir = "/path/to/local/llama_checkpoint"
generator = TokenGenerator(checkpoint=checkpoint_dir, device=Config.device)
model = generator.model
print(available_attention_layers(model))

with temporary_attention_intervention(
    model,
    layer_index=12,
    method="noise",       # or "replace"
    strength=0.01,
    seed=1234,
    enabled=True,
):
    generated_token_ids = list(generator.generate(prompt_token_ids, max_tokens=8))
# model parameters are restored here
```

Use `enabled=False` with the same call site for a no-op baseline; it neither
inspects nor mutates the model.  This is intentionally unsuitable for the
loader's optional 4-bit/8-bit mode, which is rejected rather than modified in
place.  The local Llama runtime can be passed directly after it is loaded; it
uses the same `model.layers[i].self_attn` layout.

For an interactive, portable walk-through, open
`research/phase2/attention_intervention_demo.ipynb`. It defaults to a tiny
random CPU model and explicitly labels that demonstration as mechanics only,
not pretrained tool-selection behavior. It can also show Llama or GPT-OSS
layer IDs and expected attention shapes from repository/configuration metadata
without loading weights. A real checkpoint is loaded only after the notebook
user supplies a local directory and sets its explicit `LOAD_CHECKPOINT` flag.
The notebook separately reports checkpoint quantization metadata and actual
selected-attention dtypes before checking whether the current API accepts the
target.

For one bounded real-checkpoint functional check of the saved Phase 2 example,
run `python -m research.phase2.intervention_smoke` with the development config,
an explicit saved-run directory, checkpoint directory, and fresh output
directory.  It runs the reconstructed prompt once with intervention disabled
and once with seeded `noise` (`strength=0.01`, `seed=1234`), then writes
`intervention_smoke.json` and `INTERVENTION_SMOKE_COMPLETE`.  It is not an
accuracy evaluation and does not write model weights.
