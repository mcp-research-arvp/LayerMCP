# Phase 2: observability and attention intervention

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

The valid indices and safe target names come from the supplied model rather
than a fixed model-size assumption. `attention_all` changes every registered
floating-point parameter under the selected attention module. Individual
targets such as `q_proj`, `k_proj`, `v_proj`, `o_proj`, `qkv`, or `out` are
offered only when the loaded module stores that projection separately. A fused
QKV parameter is never sliced on an assumption: it is available only as `qkv`.
For example, the local GPT-OSS architecture exposes fused `qkv` and `out`,
while the local Llama layout exposes separate Q/K/V/O projections. `noise`
sets every selected parameter `p` to
`p + strength * z`; `replace` sets it to `strength * z`; in both cases `z` is a
seeded standard-normal tensor with the same shape.  Generated values are cast
back to the parameter's original device and dtype.  The original tensors are
copied back exactly when the context exits, including after an exception.

Contexts cannot be nested/re-entered or used concurrently on the **same model**,
even for different layers or a disabled control. They fail before mutation;
sequential reuse and separate models are allowed. This guard covers this API's
contexts, not arbitrary unwrapped inference/training in another thread: use a
loaded model sequentially. Backup tensors are released on exit/error.
If restoration itself fails, that model is rejected by future contexts; reload
a fresh model before continuing rather than trusting the altered instance.

`probe.perturbation` reports actual changed-element counts and, for each tensor,
the original/effective-delta RMS and L2 norms and their ratio **after dtype
rounding**. A zero original norm has a `null` ratio. Noise that rounds away is
explicitly `no_op`, not evidence of a perturbation. `noise, strength=0` is a
no-op; **`replace, strength=0` zeros the selected tensors**. Strengths remain
absolute; no RMS-relative mode is introduced by these integrity changes.

Memory needs include a native-dtype backup of all selected tensors, CPU FP32
random noise for the largest selected tensor, native-device noise/replacement
temporaries, and bounded diagnostic chunks. The API warns if its estimated
CUDA backup/workspace exceeds free memory; this is not a full generation/KV
cache budget or a guarantee against OOM. The notebook mechanics probe also
keeps its own independent comparison clones until that cell releases them.

For the local Llama runtime used by Phase 2, obtain the already-loaded model
through the same local checkpoint path as `observe.py`:

```python
from models.architectures.llama31_8b_pytorch.config import Config
from models.architectures.llama31_8b_pytorch.inference import TokenGenerator
from research.phase2.intervention import (
    available_attention_layers,
    available_attention_targets,
    temporary_attention_intervention,
)

checkpoint_dir = "/path/to/local/llama_checkpoint"
generator = TokenGenerator(checkpoint=checkpoint_dir, device=Config.device)
model = generator.model
print(available_attention_layers(model))
print(available_attention_targets(model, layer_index=12))

with temporary_attention_intervention(
    model,
    layer_index=12,
    target="q_proj",      # choose only from available_attention_targets(...)
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
not pretrained tool-selection behavior. Configuration-only views cover the
repository's Llama, GPT-OSS, Qwen, Gemma, and Phi implementations: they show
valid layer IDs, the selected attention-module path, projection shapes, and
the targets that the current API can safely select. This works without weights.

For a real inspection, select the matching `<family>_checkpoint` choice and
set `LOAD_CHECKPOINT = True` in an interactive GPU allocation. The notebook
gets `CHECKPOINT_DIR` from the matching variable in a local `.env`; it does
not contain a personal path and does not download weights. It separately
reports checkpoint quantization metadata and actual selected-attention dtypes
before checking whether the current API accepts the target.

For a visible, one-query GPT-OSS check, also set
`LAYERMCP_GPT_OSS_SOURCE_RUN` to a saved compatible GPT-OSS reasoning-low run.
After loading the checkpoint, set `RUN_PAIRED_SAMPLE_COMPARISON = True` and
choose `PAIRED_SAMPLE_ID` in the notebook settings cell. The notebook runs an
unchanged control and the selected intervention through the existing Harmony,
tool-execution, and final-outcome evaluator path, then displays both raw
outputs and outcome fields side by side. It is intentionally limited to one
sample; use the runner below for a panel or sweep.

### Notebook settings reference

The notebook has exactly one user-editable settings cell. Leave one
`MODEL_CHOICE` and one `ATTENTION_TARGET` line uncommented. Its normal
settings are deliberately small and explicit:

| Setting | What a user chooses | Effect |
| --- | --- | --- |
| `MODEL_CHOICE` | `tiny_cpu`, `<family>_config_only`, or `<family>_checkpoint` | Selects a random CPU mechanics demo, a no-weight architecture view, or a real locally loaded model. |
| `LOAD_CHECKPOINT` | `False` or `True` | Must be `True` only for a checkpoint choice; large real checkpoints require a GPU. |
| `LAYER_INDEX` | An ID printed by the inspector | Chooses the one decoder layer to alter. Never assume a layer count from another model. |
| `ATTENTION_TARGET` | A target printed by the inspector | Chooses all attention parameters or a safe component such as `qkv`, `out`, `q_proj`, or `o_proj`. Targets vary by model family. |
| `INTERVENTION_METHOD` | `noise` or `replace` | `noise` adds seeded noise to the selected tensors; `replace` temporarily replaces them with seeded noise. |
| `INTERVENTION_STRENGTH` | A non-negative number | Sets the absolute scale in `p + strength * z` for noise, or `strength * z` for replacement. It is not normalized to the parameter scale and should be calibrated experimentally. |
| `INTERVENTION_SEED` | An integer | Makes the temporary random perturbation reproducible. |
| `RUN_ZERO_STRENGTH_ACCEPTANCE_PROBE` | `True` or `False` | Checks whether the loaded target passes the API's safety/quantization checks without changing its values. |
| `RUN_REAL_INTERVENTION_PROBE` | `True` or `False` | A mechanics-only check: verifies selected tensors change in the enabled context and restore exactly afterward. It does not generate or score a benchmark answer. |
| `RUN_PAIRED_SAMPLE_COMPARISON` | `True` or `False` | Runs one real GPT-OSS saved example twice: unchanged control, then enabled intervention. It compares raw output, tool choice, arguments, execution, final outcome, and restoration. |
| `PAIRED_SAMPLE_ID` / `PAIRED_SOURCE_RUN_DIR` | A saved example ID and compatible run | Selects the one saved GPT-OSS reasoning-low example to compare. The run directory normally comes from ignored `.env`. |
| `RUN_PAIRED_SWEEP` | `True` or `False` | Runs a bounded grid of the same control/intervention comparison for GPT-OSS. It is disabled by default. |
| `SWEEP_SAMPLE_IDS`, `SWEEP_LAYERS`, `SWEEP_TARGETS` | Explicit tuples, or literal `"all"` for layers/targets | Selects saved examples and targets. `"all"` expands only to layers and attention targets the loaded model actually exposes. |
| `SWEEP_METHODS`, `SWEEP_STRENGTHS`, `SWEEP_SEEDS` | Explicit tuples | Selects the noise/replacement methods, absolute strengths, and reproducible perturbation seeds to compare. |
| `SWEEP_MAX_INTERVENTIONS` / `SWEEP_ALLOW_LARGE` | A positive pair limit / explicit override | Stops an accidental large interactive grid before it generates. The default maximum is 24 complete control/intervention pairs. |

The two probe flags answer different questions. Use the real-intervention probe
to establish that a target can be changed and restored safely. Use the paired
sample comparison to establish whether that change affects an actual answer.
They can be enabled independently; a normal one-sample experiment needs the
paired comparison, not the mechanics probe.

### Interactive GPU notebook workflow

This is for model inspection and short, manual intervention checks. Use batch
jobs for a complete benchmark or sweep: an interactive allocation has a fixed
Slurm end time even while it is active.

1. From the repository root, make a local configuration file. It is ignored
   by Git and must never be committed:

   ```bash
   cp .env.example .env
   ```

   Set `LAYERMCP_REPO_ROOT` to the current checkout and set only the checkpoint
   variables that are available locally, for example
   `LAYERMCP_GPT_OSS_CHECKPOINT`. For the optional one-sample GPT-OSS comparison,
   also set `LAYERMCP_GPT_OSS_SOURCE_RUN`. Set `TIKTOKEN_ENCODINGS_BASE` too
   when the local GPT-OSS/Harmony tokenizer assets live outside their usual cache.

2. Once per virtual environment, install/register a Jupyter kernel:

   ```bash
   python -m pip install ipykernel jupyterlab
   python -m ipykernel install --user --name layermcp --display-name "Python (layermcp)"
   ```

3. On the cluster login node, request an interactive GPU with a deliberately
   bounded duration. Replace the account and resources with values valid on
   your cluster:

   ```bash
   salloc --account=<gpu_account> --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=00:30:00
   srun --pty bash -l
   ```

   Some clusters place the `salloc` shell directly on a compute node. If
   `hostname` already shows the allocated compute hostname, do not start a
   second `srun` shell.

   On the allocated compute node, load the project environment and start a
   local-only Jupyter server. Leave this terminal running:

   ```bash
   cd /path/to/LayerMCP
   module load StdEnv/2023 python/3.11.5
   source /path/to/venv/bin/activate
   set -a; source .env; set +a
   jupyter lab --no-browser --ip=127.0.0.1 --port=8888
   ```

   The `set -a` line exports the local path/tokenizer settings for this session.
   The notebook also reads `.env` itself, so this is safe even when only some
   variables are populated.

4. In a second login-node terminal, forward the compute node's local Jupyter
   port; use the hostname printed by `hostname` on the allocated node (not the
   Slurm job ID):

   ```bash
   ssh -N -L 8888:127.0.0.1:8888 <compute_node_hostname>
   ```

   In VS Code, forward port 8888 in the **Ports** panel if necessary. Open the
   Jupyter URL containing its token, then select **Python (layermcp)** from the
   notebook's kernel picker (or choose **Existing Jupyter Server** and paste
   that token URL). Confirm the kernel is on the allocation before loading a
   checkpoint:

   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

5. In the notebook's one editable settings cell, leave exactly one model and
   target choice uncommented. Run the setup/import cells, the inspector, and
   the runtime type/quantization cell in order. `RUN_REAL_INTERVENTION_PROBE`
   is optional; when enabled it reports a disabled no-op, selected parameters
   changed inside the context, and exact restoration after it.

   For the first real one-sample GPT-OSS experiment, use the known successful
   calculator sample below. The final notebook cell shows unchanged and
   altered raw outputs, tool calls, execution, final outcome, and restoration:

   ```python
   MODEL_CHOICE = 'gpt_oss_checkpoint'
   LOAD_CHECKPOINT = True
   LAYER_INDEX = 0
   ATTENTION_TARGET = 'qkv'
   INTERVENTION_METHOD = 'noise'
   INTERVENTION_STRENGTH = 0.01
   INTERVENTION_SEED = 1234
   RUN_REAL_INTERVENTION_PROBE = False
   RUN_PAIRED_SAMPLE_COMPARISON = True
   PAIRED_SAMPLE_ID = 'math_v1_calculator_easy_001'
   ```

   This is a preservation check. The unchanged control should reproduce the
   saved baseline's correct calculator result. If the intervention also
   succeeds, this small perturbation did not visibly damage this sample. If it
   fails, the setting harmed a previously correct sample. It does not prove a
   general effect either way.

   To compare a second query with the **same** loaded model, layer, target,
   method, strength, and seed, change only `PAIRED_SAMPLE_ID`, rerun the
   settings cell, then rerun the final paired-comparison cell. Do not rerun the
   model-loading/inspector cell: the local checkpoint model remains in GPU
   memory for this kernel session, and reloading it only wastes allocation
   time. This reuse rule applies only while the kernel and Slurm allocation are
   still alive; a restarted kernel or expired allocation requires a new model
   load.

6. Keep local paths and GPU outputs out of the tracked notebook. Put paths in
   ignored `.env`; copy results you need elsewhere. Before closing a notebook
   used for a real run, choose **Discard/Don't Save** if prompted, then reopen
   it to return to the portable committed defaults. Interrupt Jupyter with
   `Ctrl-C`, exit the allocation shell, and let or cancel the interactive
   allocation. Do not rely on notebook activity to extend its Slurm deadline.

### Exploration versus sweeps

Use the notebook for architecture inspection, one-sample comparisons, and a
small visual paired sweep. Its sweep settings expose sample IDs, layers,
targets, strengths, methods, and seeds. The notebook prints its planned count
before generation and defaults to at most 24 complete control/intervention
pairs (48 generations). A request beyond that stops before generation unless
`SWEEP_ALLOW_LARGE = True` is set deliberately.

There is no model-architecture cap hidden in the notebook. For the current
GPT-OSS runtime, `SWEEP_LAYERS = 'all'` and `SWEEP_TARGETS = 'all'` expands to
24 layers times its three safe targets (`attention_all`, `qkv`, and `out`): 72
pairs for one sample, one method, one strength, and one seed. Adding two
methods, three strengths, two seeds, and two samples would request 1,728
pairs (3,456 generations). That is technically allowed only with the explicit
override, but it is unsuitable for an interactive allocation: use the paired
runner below for a full model-wide grid so the results have durable output and
completion markers.

For a larger study, keep three disjoint sample-ID lists:

1. **Exploration:** compare settings on a small panel of successes and failures.
2. **Validation:** test shortlisted settings on different queries; finalize the
   layer/target/method/strength/seed policy here.
3. **Held-out reporting:** freeze that policy, then evaluate untouched queries.
   Do not tune settings after looking at these results; if you do, those queries
   become exploration data and a fresh held-out panel is needed.

Record all three lists and the chosen settings before the reporting run;
manually check that the lists do not overlap. The current tools do not enforce
dataset splits. This is hyperparameter selection rather than model-training
cross-validation: a one-query sweep cannot establish general improvement.

The small CPU model is only a fast explanation of targeting and restoration.
Configuration-only mode reads model configuration (and, when supplied, local
`config.json`) without loading checkpoint weights. A `<family>_checkpoint`
mode is the authoritative view of real parameter names, shapes, dtypes,
devices, and intervention acceptance.

For one bounded real-checkpoint functional check of the saved Phase 2 example,
run `python -m research.phase2.intervention_smoke` with the development config,
an explicit saved-run directory, checkpoint directory, and fresh output
directory.  It runs the reconstructed prompt once with intervention disabled
and once with seeded `noise` (`strength=0.01`, `seed=1234`), then writes
`intervention_smoke.json` and `INTERVENTION_SMOKE_COMPLETE`.  It is not an
accuracy evaluation and does not write model weights.

## Paired GPT-OSS full evaluation

`research.phase2.gpt_oss_intervention_eval` runs an unchanged control and one
attention intervention for each selected saved GPT-OSS example, layer, and
seed. It reuses the local GPT-OSS Harmony generation/parser and the baseline
evaluator's tool execution, argument scoring, and final-outcome scoring. Its
`paired_records.jsonl` records both conditions' raw output and parser state,
expected/selected tool, expected/selected arguments, execution result, and
final-outcome fields, plus concise routing aliases. A run is valid only when
it has `RUN_COMPLETE`; interruption or failure leaves no summary or completion
marker, and weights must restore exactly after every intervention.

Unlike the earlier routing-only prototype, this runner executes predicted MCP
tools through the same baseline evaluator path. Use a small, deliberate saved
sample panel first. The evaluator retains its existing per-sample isolation
for stateful retail tools; additionally, each condition now executes in a
fresh MCP server session, including non-retail tools. A registry change between
sessions fails the run. Fresh processes reset process-local state, **not**
shared external files/databases/services; such tools require separately reset
fixtures before a causal comparison. This runner does not write model weights.

```bash
python -m research.phase2.gpt_oss_intervention_eval \
  --source-run-dir /path/to/saved_gpt_oss_single_step_run \
  --checkpoint /path/to/local_gpt_oss_checkpoint \
  --output-dir /path/to/fresh_output \
  --layers 0 --target qkv --seeds 1234 --method noise --strength 0.01 \
  --sample-ids math_v1_calculator_easy_001,math_v1_convert_units_easy_001
```

The runner requires an exact live-registry match by default, since changed tool
descriptions change the Harmony prompt. Use comma-separated `--layers`,
`--seeds`, and `--sample-ids` for a bounded experiment. Omit `--sample-ids`
and use `--example-limit` only when saved-run order is intentional. It never
writes model weights.

### Integrity fields and interpreting results

- Hardened runs use `phase2_gpt_oss_attention_intervention_eval_v2`, because
  fresh condition sessions change the experiment protocol (not scoring).
  Existing v1 outputs and queued jobs remain untouched. V2 reporting requires
  the companion combiner update on `research/phase2-paired-results-hardening`;
  the original combiner accepts v1 only. The updated combiner supports either
  version in a report but rejects mixed v1/v2 inputs and incompatible hardened
  fingerprints/runtime settings. Do not silently pool v1/v2 experiments.
- Durable runs stream SHA-256 hashes of the local `.safetensors` files,
  `config.json` and `TIKTOKEN_ENCODINGS_BASE/o200k_base.tiktoken` once before
  loading. Set that environment variable to the local Harmony asset directory.
  Missing inputs fail native-loader runs. Hashing reads the entire checkpoint
  and can add substantial cold-storage startup time; budget for it in Slurm.
  File signatures are checked again after loading and before completion.
- `run_config.json` includes the file manifest/fingerprint, source run ID,
  current Git commit/dirty status, Python/package/PyTorch/CUDA versions, GPU
  names/driver versions, model dtypes, generation settings and deterministic/TF32 flags.
  Driver information is explicitly `unavailable` if it cannot be read, rather
  than guessed from the CUDA runtime version.
  A fingerprint of the repository's evaluator/model/server/Phase 2 Python code
  also distinguishes different local edits sharing the same Git commit.
  Recorded settings are not a promise of bitwise deterministic GPU inference.
  The native loader rejects changed cached inputs or a switch to another
  checkpoint/tokenizer; restart the kernel/process when switching those.
- Pairs and conditions are explicitly labelled as intervention experiments,
  not ordinary baseline records. `registry_exact_match` remains mandatory by
  default; scoring and the baseline reporting protocol are unchanged.
- Each pair's `saved_control_comparison` compares available saved raw output,
  parsed tool, arguments and parser status against the newly generated control.
  A mismatch is labelled `differs_new_control_baseline`. Missing historical
  fingerprints yield `source_checkpoint_equivalence: unverifiable`; even
  matching outputs/fingerprints do not prove identical historical runtime.
  Notebook helpers verify the native loaded model/tokenizer binding and carry
  its checkpoint fingerprint into each pair. The first comparison hashes the
  full checkpoint once; subsequent comparisons reuse the cached identity with
  file-signature and complete file-set checks. Arbitrary caller-supplied models
  remain `caller_supplied_unverified`. This is not a digest of all live model
  tensors: do not train or edit the model outside the intervention API.
- `intervention_verification.perturbation` and the notebook show effective
  tensor changes and explicit no-ops alongside exact restoration. No-op pairs
  are counted separately in the summary and must not be treated as nonzero
  perturbation evidence.
- `summary.json` includes all four tool-choice transitions, tool changes,
  repairs and damage by layer/seed. Output categories are mutually exclusive:
  malformed output, valid no-call, valid wrong-tool call, valid correct-tool
  call. Final-outcome scoring stays separate and unchanged.
- Repeated controls are compared for output/argument/execution/outcome drift.
  `unique_control_examples` counts sample IDs once, while `paired_observations`
  includes layer/seed repetitions. If `control_drift_sample_ids` is nonempty,
  investigate before attributing differences to the intervention. A tiny panel
  with only one control per query labels repeatability `not_tested`, not stable.
  Per-sample observation counts identify which queries actually had repeats.
  Such a panel is a diagnostic, not a population-level effect estimate. Larger studies need
  sample-level uncertainty analysis and separate exploration/held-out panels;
  seeds are not independent benchmark questions.

The runner calls narrow public adapters in `evaluation.evaluate` and
`models.routers.gpt_oss_local_router`; those delegate to the existing native
prompt, generation, parser, registry and evaluator implementations. They do
not introduce a second scoring pipeline.

Focused CPU/mock validation in the project environment (no weights/GPU needed):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest \
  tests.test_phase2_intervention tests.test_phase2_experiment_integrity \
  tests.test_phase2_gpt_oss_intervention_eval tests.test_gpt_oss_cache_integrity \
  tests.test_phase2_intervention_notebook tests.test_phase2_paired_sweep \
  tests.test_evaluate_metrics tests.test_minimal_scorecard tests.test_gpt_oss_api
python -m unittest tests.test_evaluation_tool_catalog
python -m compileall analysis benchmark evaluation mcp_server models research tests
git diff --check
```

These checks do not replace a real GPU smoke run or report a GitHub CI result.

### GPU review smoke check (manual; no new notebook)

Use the hardened worktree, a live interactive allocation, the registered
project kernel and your ignored `.env` paths. Load GPT-OSS once as usual. First
use the existing sweep cell for a repeated **unchanged/sham** check:

```python
RUN_PAIRED_SAMPLE_COMPARISON = False
RUN_REAL_INTERVENTION_PROBE = False
RUN_PAIRED_SWEEP = True
SWEEP_SAMPLE_IDS = ('math_v1_calculator_easy_001',)
SWEEP_LAYERS = (0,)
SWEEP_TARGETS = ('qkv',)
SWEEP_METHODS = ('noise',)
SWEEP_STRENGTHS = (0.0,)  # Deliberate no-op probe, not a perturbation finding.
SWEEP_SEEDS = (1234, 5678, 9012)
SWEEP_MAX_INTERVENTIONS = 3
SWEEP_ALLOW_LARGE = False
SWEEP_SHOW_RAW_OUTPUTS = True
```

Run the settings cell, then the existing sweep cell. Expect three no-op pairs,
zero changed values, exact restoration, a populated checkpoint fingerprint,
and `control_repeatability_status: stable_repeats` with no drift IDs. Check raw
outputs across both conditions, not just the scores. The control should
reproduce the saved calculator success; inspect any saved-output mismatch.

Then change **only** `SWEEP_STRENGTHS = (0.01,)`, rerun settings and sweep, and
verify nonzero changed-value counts/norms and exact restoration for all three
pairs. Do **not** reload the model. The intervention need not cause a wrong
answer for this safety check to pass. If controls drift, a fingerprint is
unavailable or restoration fails, investigate before drawing causal conclusions.
This check must actually run on a GPU before we claim GPU validation of v2.

The display-only cell immediately after the sweep shows per-tensor changed
values, original RMS, effective change RMS, relative change and restoration
from the latest `sweep_records`. It does not generate, reload or modify weights;
measurements describe changes made inside the earlier context, not the current
restored tensors. Before any sweep it prints a harmless notice. The ratio is
the effective delta norm divided by the original norm, not a new noise-scaling
mode. Rerunning a sweep replaces `sweep_records`; export records outside the
repository before comparing another configuration or closing the kernel.
Interactive exports are evidence, not batch `RUN_COMPLETE` folders for the
completed-output combiner. Once evidence is preserved, restore portable
defaults and clear notebook outputs before committing.

### Minimal GPT-OSS starter panel

These saved IDs are a quick five-query panel from the frozen GPT-OSS primary
run. They are intentionally not a scorecard or a layer sweep; they check that
the paired path can preserve good behavior and expose different failure modes.

| Saved sample ID | What it checks in the frozen GPT-OSS baseline |
| --- | --- |
| `math_v1_calculator_easy_001` | Correct control: tool, arguments, execution, and final outcome all succeed. |
| `math_v1_convert_units_easy_001` | Shared routing boundary: expected `convert_units`, selected `unit_converter`. |
| `math_v1_calculator_difficult_001` | Reference-argument mismatch but correct final outcome (`4 * 11 + 6` versus `4*11+6`). |
| `math_public_v2_calculator_005` | Invalid/no-call boundary: GPT-OSS produced a parse error. |
| `finance_controlled_finance_parse_xbrl_003` | Finance routing boundary: expected `finance_parse_xbrl`, selected `finance_get_company_facts`. |

Use a fresh output directory and the exact saved GPT-OSS primary run that
contains these IDs. This runner currently supports the tested local GPT-OSS
Harmony **reasoning-low** path. It is the first adapter to the reusable
baseline single-step evaluator; other model/runtime adapters must be added
only after their corresponding baseline path is checked.
