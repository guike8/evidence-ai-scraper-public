# Local model experiments

This optional workflow compares an unchanged small model, a gradient-trained
QLoRA adapter, and a separately diagnosed direct weight-space edit. It does not
change the scraper's default model or start/stop its inference service. Training
and synthetic validation have run on Apple Silicon; see [measured results](RESULTS.md).
Representative real-web accuracy remains unverified.

The frozen [experimental MLX adapter](https://huggingface.co/guike8/evidence-scraper-qwen3-1.7b-mlx-lora)
is distributed separately with its model card, measurements and checksums.
It is MLX-only and has not qualified for general production use; read the
[held-out evaluation](RESULTS.md#frozen-held-out-comparison) before using it.

The baseline is public, ungated `mlx-community/Qwen3-1.7B-4bit`, pinned at
`3b1b1768f8f8cf8351c712464f906e86c2b8269e`. Its `model.safetensors` is
968,080,210 bytes, SHA256
`0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f`.
The MLX conversion's historical upstream revision is not documented by its
publisher. The model is Apache-2.0; repository source code remains MIT.

## Prepare on an Apple Silicon Mac

Use Python 3.12 and a private experiment directory outside the source checkout.
Commands below run from the source checkout; replace `/private/scraper-lab`
with your private directory. No API credential is required for these public
downloads, training, local inference or scoring.

```sh
uv venv --python 3.12 /private/scraper-lab/env
uv pip install --python /private/scraper-lab/env/bin/python -r training/requirements.lock -e .
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 /private/scraper-lab/env/bin/hf download \
  mlx-community/Qwen3-1.7B-4bit \
  --revision 3b1b1768f8f8cf8351c712464f906e86c2b8269e \
  --local-dir /private/scraper-lab/base
/private/scraper-lab/env/bin/python -m training.dataset \
  --output-dir /private/scraper-lab/data --target-order schema
```

Installing the scraper into this separate environment supplies validation and
evaluation dependencies. Keep the training lock's MLX-LM commit and resolved
dependencies pinned. Preserve the serving environment.

The [data notes](README-data.md) describe the synthetic 720/72/108 split across
eight presets and a custom schema. Gold facts are authored independently of
model outputs. Splits use different DOM/lexical families, entities, domains and
source hashes; shared simple task patterns remain. No private snapshots or
existing benchmark cases enter training. The dataset is automatically checked,
not independently reviewed real-web data.

Schema order matches the tested server grammar's forced JSON property order.
The canonical mode remains available to reproduce the original pilot bytes;
its alphabetical targets differ from the serving grammar. The source messages,
gold facts and split identities are identical across these serialization modes.

With the pinned tokenizer, all 900 pilot rows passed full/prompt prefix checks;
the longest row was 2,379 tokens and every row had supervised answer tokens.
These are tokenization measurements, not training or inference measurements.

## Train without depending on the SSH session

```sh
/private/scraper-lab/env/bin/python -m training.job start \
  --python /private/scraper-lab/env/bin/python \
  --model /private/scraper-lab/base --data /private/scraper-lab/data \
  --run-dir /private/scraper-lab/run-001 --max-seconds 3600 --iters 720
/private/scraper-lab/env/bin/python -m training.job status \
  --run-dir /private/scraper-lab/run-001
```

The detached supervisor holds a process lock, persists private state and runs
preflight, a disposable one-update sanity check, then training. Starting the
same run again reports existing state. It never silently starts duplicate work
or restarts a failed run. Its deadline terminates only its own child process
group. A failed/interrupted run requires a new run directory and an explicit
decision about saved weights. Upstream checkpoints contain adapter weights,
not complete optimizer/RNG state, so they cannot provide an exact continuation.

The initial settings are rank 8, the last 8 layers' attention q/v projections,
batch 1, gradient accumulation 4, learning rate 1e-5, 4,096-token maximum,
gradient checkpointing and an 8 GiB MLX allocator limit. The example's 720
iterations are microbatches: 180 optimizer updates and one pass over the pilot
training split. The CLI defaults to 200 microbatches for a shorter pilot.
Check available memory alongside any
other running models before launch; no service is stopped automatically.

Preflight forces `enable_thinking=False`, checks actual MLX ChatDataset masking
against the generation prefix, and rejects overlong rows instead of truncating
JSON targets. Sanity checks finite nonzero gradients, a changed adapter and
exact save/reload tensor equality. The original base's hashes are rechecked.
Only data/evidence/warnings are training targets; application code assigns
provenance and status. No test loss or test accuracy is used during training.
Upstream's last validation callback precedes the final update, so evaluate the
saved final adapter separately.

## Compare models

`training.infer` runs the isolated MLX model with the production system/source
messages, thinking disabled and fixed greedy sampling. It supports `base`,
`lora` and `wsp` arms. Its unconstrained JSON generation measures the model
itself; it does not establish schema grammar compatibility. Freeze a candidate
after validation before opening the held-out test split. See its `--help` for
artifact and frozen-candidate arguments.

`training.evaluate` scores saved outputs or calls a separate loopback schema
server through the actual LocalExtractor. The local server must support the
app's structured output contract. Test the actual serving artifact and grammar
before switching the application's model.

```sh
python -m training.evaluate score /private/scraper-lab/data/valid.cases.jsonl \
  /private/scraper-lab/base-outputs.jsonl \
  --model mlx-community/Qwen3-1.7B-4bit \
  --revision 3b1b1768f8f8cf8351c712464f906e86c2b8269e \
  --response-format none --output /private/scraper-lab/base-metrics.json
```

Compare identical cases, prompts, budgets and inference modes. Report raw and
accepted field precision/recall, required-field recall, null abstention,
unsupported/wrong facts, evidence coverage, schema/completeness, failures,
truncation, repetition and latency. Missing, rejected and timed-out outcomes
remain in recall. Exact quote occurrence is checked separately from matching
independent labels. Expected critical-missing cases can reject otherwise
correct raw facts; read raw and accepted scores separately. A perfect authored
108-case replay checks 676 fact leaves and 113 absent-value controls; it is a
scorer plumbing check, not a language-model result.

## Weight-space experiment

The [weight-space programming article](https://rikiyat.github.io/blog/weight-space-programming/)
motivates an independent experiment. Its language-model edit addressed thinking
loops on Qwen3.5-0.8B. This workflow uses a different architecture, thinking off
and quantized weights. Neither its layer indices, direction nor edit strength
is transferred automatically.

First reproduce an extraction failure on training/calibration cases. Use a
bounded set of forward passes and independent control cases to determine
whether a localized repeated-generation behavior exists. Inspect the target
weights and matching activations before choosing a direction and columns.
Keep the final test split sealed during diagnosis. Record all probes and
operator decisions; do not describe random edits as a diagnosed fix.

`training.weight_space.write_bundle` requires exact base/file/tensor hashes,
the observed failure, operator rationale, separate calibration/control source
hashes, explicit module paths, a unit direction, columns and edit strength.
It creates the non-gradient rank-one edit
`W_new = W - alpha * outer(u, mask * (u @ W))` with a declared relative norm
bound. `apply_bundle` installs frozen additive wrappers, preserving every
original quantized base tensor and avoiding requantization. The wrapper uses
the same arithmetic shape as rank-one LoRA but its values are written directly;
no optimizer trains this arm. Arithmetic controls include an alpha-zero no-op,
explicit forward equivalence, frozen parameters and unchanged base hashes.
These controls do not demonstrate a useful edit on a language model.

Keep base, zero-edit control, direct edit and QLoRA results separate. An edit
that fails controls, increases unsupported facts, or does not improve the
validation criterion is not promoted. If no comparable localized failure is
observed, report that result rather than copying the paper's patch.

## Publish only a verified artifact

No trained adapter or edited language-model artifact is included in this
repository. After training, reload and score the exact saved candidate, freeze
its settings and run held-out evaluation plus representative reviewed web
fixtures. Compare against the same base, retaining all failures. A synthetic
pilot alone cannot establish universal extraction or superiority.

Keep generated data, checkpoints, raw outputs, runtime configuration and
credentials private. Prepare a new model-only staging directory containing the
chosen adapter tensors/config, the upstream Apache-2.0 license and attribution,
a model card, SHA256SUMS and sanitized measured results. Remove local paths
from the MLX adapter config. MLX adapters are not PEFT-compatible; do not claim
Transformers or GGUF support. If fusing, reevaluate the exact fused artifact:
default quantized fusion requantizes adapted matrices.

The model card must give the exact base pin, data fingerprint/provenance,
training/runtime versions and settings, sample counts, before/after metrics,
known limitations and no-thinking usage. Use `library_name: mlx`,
`license: apache-2.0`, `base_model: mlx-community/Qwen3-1.7B-4bit`, and
`base_model_relation: adapter` for the QLoRA adapter. Preserve separate
attribution if incorporating any published third-party edit artifacts.

Publishing to a Hugging Face account requires write authentication, even
though downloading and running the public model is credentialless. Authenticate
with `hf auth login` in a terminal; never put tokens in source, model cards,
training logs or chat. Upload only the audited model staging directory.
