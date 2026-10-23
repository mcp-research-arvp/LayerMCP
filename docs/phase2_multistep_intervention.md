# Paired GPT-OSS multi-step attention intervention

`research.phase2.gpt_oss_multistep_intervention_eval` evaluates one active
multi-step dataset group per run. It loads the native GPT-OSS checkpoint once,
then runs the evaluator's `guided_predicted_rollout_v1` protocol twice: a
control rollout and an intervention rollout. Each condition starts a fresh MCP
evaluator session. The intervention applies the configured attention parameter
change during each generation, restores the original parameters afterwards,
and verifies exact restoration. Subsequent prompts use that condition's own
predicted calls and tool results.

The runner uses the active dataset groups in `analysis.multi_step_run.DATASET_GROUPS`.
These include grounded execution, offline replay, controlled, and public or
source-derived datasets. Report those categories separately. It does not use
saved single-step examples or archived/fixture benchmarks. The existing
multi-step evaluator supplies tool execution, step scoring, workflow scoring,
and per-benchmark-mode summaries. The runner's paired transitions are
additional diagnostics; execution success is not a task-success metric.

## Submit complete Phase 2 datasets

`scripts/slurm/submit_phase2.sh` loads the repository `.env` itself. Run it on
the login node from any directory after the short test below has completed:

```bash
scripts/slurm/submit_phase2.sh both --dry-run
scripts/slurm/submit_phase2.sh both
```

Set `LAYERMCP_VENV` in `.env` to use an external virtual environment. The
default is `$LAYERMCP_REPO_ROOT/.venv`; relative paths resolve from the checkout.
The submitter checks for a readable `bin/activate` and an executable, working
`bin/python` before submitting any jobs, including in dry-run mode. Both
launchers use the same setting and repeat the check on the compute node.
Optional `LAYERMCP_MODULES` and `LAYERMCP_SLURM_ACCOUNT`, `_PARTITION`,
`_GRES`, `_CPUS_PER_TASK`, `_MEM`, and `_TIME` values in `.env` provide
site-specific deployment settings; no account, GPU type, module stack, or
resource request is hard-coded by these Phase 2 launchers.

Use `single` for the complete Phase 1 primary single-step benchmarks, or `multi` for all
eight active multi-step groups alone. `both` is the default and submits nine
jobs: one single-step job and eight multi-step jobs. Like Phase 1, the single-step
job reads the 13 primary benchmark JSON files directly through the canonical
evaluator loader; it does not use an older saved run. Every multi-step group has its own job and runs all its
workflows. `--primary-only` omits the optional controlled math multi-step
group, leaving seven multi-step jobs. The script defaults to layer 0, `qkv`,
`noise`, strength `0.01`, and seed `1234`; these can be changed with
`--layer`, `--target`, `--method`, `--strength`, and `--seed`.
`--strength-mode tensor_rms` and `--plan-manifest` apply only to direct
single-step jobs; multi-step intervention remains absolute-strength.

Use `--domains math,finance` or `--single-datasets math_public,finance_finqa_test_single`
to restrict the single-step job. These selectors use the same IDs as Phase 1.
`LAYERMCP_GPT_OSS_SOURCE_RUN` is no longer needed for `submit_phase2.sh`;
the saved-run CLI remains available for notebook and historical comparisons.
Direct single-step runs save SHA-256 content hashes in `run_config.json` under
`benchmark_sha256`, alongside `selected_sample_manifest` with ordered sample
IDs and counts per benchmark, including files with zero selected rows after a
limit. Benchmark contents are checked after loading and before completion;
changed or missing inputs fail the run without `RUN_COMPLETE`. Per-row evidence
categories use the standard scorecard's classification rules, including
`diagnostic/adapted` for `public_adapted` sources.

A full job uses
the same control and intervention pairing as the short test. Keep replay,
public/source-derived, and controlled results separate in reporting.

## Submit a short test

From the repository root on the login node:

```bash
set -a
source .env
set +a
export DATASET_GROUP=math_controlled RUN_KIND=short_test WORKFLOW_LIMIT=2
export LAYER_INDEX=0 ATTENTION_TARGET=qkv METHOD=noise STRENGTH=0.01 SEED=1234
sbatch --export=ALL,LAYERMCP_REPO_ROOT,LAYERMCP_GPT_OSS_CHECKPOINT,TIKTOKEN_ENCODINGS_BASE,DATASET_GROUP,RUN_KIND,WORKFLOW_LIMIT,LAYER_INDEX,ATTENTION_TARGET,METHOD,STRENGTH,SEED \
  scripts/slurm/run_phase2_multistep.sbatch
```

The Slurm stdout path is `slurm-gpt-oss-phase2-multistep-JOBID.out`. The
run directory is printed on completion and begins with
`results/runs/phase2/JOBID_gpt-oss_multistep_...`. Inspect `RUN_COMPLETE`
before using the result; `RUN_FAILED.json` records an interrupted or failed
run. `control/` and `intervention/` each contain canonical multi-step
`samples.jsonl` and `summary.json`. `paired_records.jsonl` compares matched
workflows and steps; the top-level `summary.json` contains both condition
summaries and paired correctness transitions. `run_config.json` records the
checkpoint fingerprint, code/runtime identity, dataset hash, selected IDs,
raw field counts, and evaluator-applied default counts.

## Submit a full group

After the short test succeeds, run one group per job. For example, full FinQA:

```bash
export DATASET_GROUP=finance_finqa RUN_KIND=full
unset WORKFLOW_LIMIT
sbatch --export=ALL,LAYERMCP_REPO_ROOT,LAYERMCP_GPT_OSS_CHECKPOINT,TIKTOKEN_ENCODINGS_BASE,DATASET_GROUP,RUN_KIND,LAYER_INDEX,ATTENTION_TARGET,METHOD,STRENGTH,SEED \
  scripts/slurm/run_phase2_multistep.sbatch
```

Keep the intervention configuration fixed across groups. `RUN_KIND=full`
rejects a workflow limit. Each group takes two full model rollouts, so schedule
large groups independently. The current job cannot resume a partial condition
after its wall time expires; the partial directory remains marked incomplete.
