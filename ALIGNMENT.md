# VibeSim instrumentation on upstream vLLM

This checkout maintains `glm53-dflash2-alignment` on official vLLM
`upstream/main`. The current upstream base is
`8369affa5428ca29a30f378d660fbffbad12e240`.

The instrumentation source is `glm52-spec5-alignment` at
`556aaf7309348fa292d76aa0f49533937eaf48bb`. Its original checkout remains at
`../wt-glm52-spec5/alignment/profiler/vllm`; do not switch, install into, or edit
that checkout when maintaining this branch.

## Scope

The eight source commits, in order, are `7c025011f`, `24b6c79a8`, `08fe031f0`,
`15092c51b`, `4492fff9a`, `91547cfa0`, `0b8edfb56`, and `556aaf730`.
They preserve request TTFT/TPOT, API timing, EngineCore cadence and speculative
progress, bulk token/routing dumps, and target/draft EPLB provenance.

Migration adaptations preserve upstream's `capture_iteration_details` and
scheduler prefill throttling. Token-in/token-out serving now lives in
`vllm/entrypoints/scale_out/token_in_token_out/`.

Model Runner V2 additionally emits indexed `vllm_iteration(N): <phase>` scopes
for preprocess, target forward, postprocess, sampling, draft, bookkeeping, and
EPLB. The draft scope covers the complete speculator proposal, including the
DFlash2 selector. Warmup/dummy calls do not acquire real iteration IDs. The
index travels with `ExecuteModelState` from target execution to sampling.
Bookkeeping may have multiple disjoint scopes in one iteration.

`VLLM_NVTX_SCOPES_FOR_PROFILING=1` enables the scopes. Enable
`--enable-logging-iteration-details` for schema-v4
`VibeSimAlignmentIteration` records. Existing
`VLLM_VIBESIM_TOKEN_TRACE_*` and `VLLM_VIBESIM_ROUTING_TRACE_*` switches apply
to both model runners. Dumps can synchronize/copy GPU data; collect them in
the separate evidence pass, not in timing captures. The token dump uses actual
query boundaries, including when adaptive verification trims scheduled tokens.

## Environment

Use this checkout's `.venv`; never copy another checkout's editable environment
or native extensions. The installation used:

```bash
uv venv --python 3.12 .venv
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=8369affa5428ca29a30f378d660fbffbad12e240 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python -r requirements/lint.txt \
  pytest pytest-asyncio nvtx ruff tblib
uv pip check --python .venv/bin/python
```

If the upstream base changes, update the wheel commit explicitly. Import and
GPU validation must resolve Python and native extensions under this worktree.
An import check alone does not qualify the driver or a serving run.

## Frozen models

| Role | Repository | Revision |
| --- | --- | --- |
| Target | `zai-org/GLM-5.3` | `aca966e4e02791568aa6a4ced368624b3d897f42` |
| Draft | `incoai/GLM-5.3-DFlash2` | `425aa615ce320caac34400208b30808c8f14f76c` |

The snapshots live under `/raid/hf/hub/models--<owner>--<model>/snapshots/<revision>`.
The target is the original roughly 756 GB checkpoint, not GLM-5.3-Flash or an
NVFP4 conversion. The roughly 4.92 GB BF16 draft has six layers, hidden size
6144, block size 8, selector rank 256, and selector top-k 16. It proposes seven
draft tokens per block.

Download logs and completion manifests live in
`../glm53-dflash2-artifacts/`. A model's manifest is written only after every
repository file exists and its size matches Hub metadata. The download script
can resume the fixed revisions using the shared cache.

For a future serving validation, select Model Runner V2 and pass the draft
snapshot with `--speculative-config`:

```json
{"method":"dflash","model":"/raid/hf/hub/models--incoai--GLM-5.3-DFlash2/snapshots/425aa615ce320caac34400208b30808c8f14f76c","num_speculative_tokens":7}
```

Use `VLLM_USE_V2_MODEL_RUNNER=1`. Select topology, memory budget, and attention
backend against the actual available GPUs. The model card demonstrates SGLang;
checkpoint compatibility in a full vLLM run still requires validation.

## Maintenance and validation

Fetch official upstream, record its SHA, and replay this branch's instrumentation
in an isolated worktree. Keep the wheel base and Python base identical. Preserve
upstream fixes when resolving conflicts; audit both runner implementations.

Focused tests include:

```bash
.venv/bin/python -m pytest -q \
  tests/v1/worker/test_alignment_trace.py \
  tests/v1/worker/test_gpu_model_runner_v2_eplb.py \
  tests/v1/engine/test_iteration_logging.py \
  tests/entrypoints/openai/completion/test_alignment_api_timing.py \
  tests/entrypoints/scale_out/token_in_token_out/test_generate_stream.py
.venv/bin/python -m pytest -q tests/v1/engine/test_output_processor.py \
  -k alignment_request_timing
```

The final migration result and commands belong in the task artifacts. Do not
claim GPU/NSYS or full GLM-5.3 serving qualification from unit-test results.
