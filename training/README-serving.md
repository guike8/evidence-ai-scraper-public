# Experimental local adapter serving

`training.serve_omlx` exposes the verified, unfused MLX LoRA through a separate
loopback oMLX server. It uses unsupported process-local loader hooks and requires
`--experimental-omlx-adapter`. It does not change installed packages, train,
fuse or requantize weights, update the scraper's private profile, or stop another
service. Model downloads and local inference need no API credential.

Use a dedicated Apple Silicon serving environment. The tested versions are:

| Package | Version |
|---|---|
| oMLX | `0.7.3rc1` |
| MLX | `0.32.4` |
| MLX-LM | `0.32.3.dev8+g0d4d3d04f` |
| XGrammar | `0.2.8` |

The MLX-LM serving revision is
`0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`, distinct from the training lock's
revision. Do not replace an existing serving environment to run an experiment.
For example, prepare a new environment from verified package artifacts or the
pinned VCS revision. Replace these paths with a writable directory outside the
source checkout. The measured serving interpreter was Python 3.12.16 on arm64:

```sh
uv venv --python 3.12.16 /private/scraper-serving/env
uv pip install --python /private/scraper-serving/env/bin/python \
  'omlx==0.7.3rc1' 'mlx==0.32.4' 'xgrammar==0.2.8' \
  'mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git@0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d' \
  -e .
```

This is an installation example, not a claim that every historical artifact is
available from every package index. Version strings alone are insufficient:
the helper also refuses changed installed package trees, including native
libraries and resources. The measured Apple Silicon package identities are:

| Package | Regular files | Bytes | Canonical tree SHA256 |
|---|---:|---:|---|
| oMLX | 641 | 105358682 | `4a2955e933c7f5efe26c8282bd0747a4969477d032d4627729005b6dc0408709` |
| MLX | 452 | 243821514 | `fbea43bc9098d1c93c44b7d6ad3eda22e8fd2e13684c6a6cb0c8078c57bf9ce2` |
| MLX-LM | 209 | 2036132 | `5d099ec02f044540cb32b8721bec4b0f0fced01168fa17592bc1508882a9f7f4` |
| XGrammar | 40 | 4059583 | `06be1e91fd5b9260e5c60c6f0c88d67def919d75bded0cab81bfe37f4c2b86e5` |

These identities include platform and Python ABI dependent native content;
another legitimate build of the same version can fail verification. Keep the
verified package artifacts in an offline wheelhouse. Do not replace the pins
merely to bypass a mismatch. MLX-LM must additionally have `direct_url.json`
identifying the exact GitHub URL and full commit above.

Canonical hashing sorts every relative POSIX file path, then hashes the
concatenation of UTF-8 path, NUL, lowercase file SHA256 hex and newline. It
excludes `__pycache__` directories, `.DS_Store`, `.pyc` and `.pyo`; symbolic links
are rejected. Reads are bounded to 20,000 files, 2 GiB per file and 4 GiB total.
All package checks run before model/backend imports.
Import specs must resolve to those same verified package roots, so an earlier
`PYTHONPATH` or `sys.path` entry cannot supply a different implementation.
Already-loaded backend modules receive the same origin checks. MLX may use a
namespace package only when its single search root is the verified directory.
Local absolute package paths are used internally and omitted from receipts.
The serving environment does not use `training/requirements.lock`.

Download the pinned public base and a released adapter separately. The base is
`mlx-community/Qwen3-1.7B-4bit`, revision
`3b1b1768f8f8cf8351c712464f906e86c2b8269e`. Keep the adapter's minimal
`adapter_config.json`, `adapters.safetensors` and portable `training.json`
together. Its base/tokenizer hashes must match the local base. A complete
original training run can instead supply its parent `train.json`.

Run from the source checkout with a new private state directory:

```sh
/private/scraper-serving/env/bin/python -m training.serve_omlx \
  --experimental-omlx-adapter \
  --base /private/scraper-serving/base \
  --adapter-dir /private/scraper-serving/adapter \
  --adapter-sha256 VERIFIED_ADAPTER_SHA256 \
  --state-dir /private/scraper-serving/run \
  --model-name evidence-scraper-local --port 8001
```

Keep that terminal open. Ctrl-C stops this foreground server; no stored PID is
used to stop or restart anything. A lifetime file lock prevents duplicate work
in the same state directory. A busy port fails explicitly. State is private and
bound to the exact model, adapter, backend versions and port; choose a new state
directory for a changed profile. The server binds only `127.0.0.1`, loads one
whitelisted base, uses an empty MCP configuration and inherits no API tokens.
`LocalExtractor` also sends `tool_choice="none"` to disable server-side MCP tool
merging for extraction requests.
Only this process receives an 8 GiB MLX allocator limit. This is not an OS
network sandbox or a total-machine memory limit.

Before serving, the helper checks the adapter checksum, compatible configuration,
all 32 finite float32 tensors and their exact shapes. Weights, configuration and
the receipt use bounded regular-file reads through no-follow descriptors, with
inode/size/timestamp checks before and after reading. Replaced or growing inputs
fail verification; receipt parsing uses the captured bytes. These are local
file-integrity checks rather than a sandbox against another process continually
modifying the environment. After oMLX startup it
checks every original base tensor, exact saved LoRA values, the 16 additive
wrappers and the original Qwen forward methods. `integrity.json` records portable
hashes and counters. A successful startup proves these integration checks, not
extraction accuracy. The helper has CPU mock coverage; run an actual Studio
startup and request before treating a new installation as verified.

In another terminal, select this model explicitly:

```sh
export SCRAPER_RETRIEVAL_BACKEND=direct
export SCRAPER_INFERENCE_BACKEND=local
export SCRAPER_ALLOWED_DOMAINS=example.org
export LOCAL_BASE_URL=http://127.0.0.1:8001/v1
export LOCAL_MODEL=evidence-scraper-local
uv run ai-scraper extract https://example.org/ --output page.json
```

The regular exclusive-reference grammar is the default. Source-quote mode is a
separate experimental opt-in and should not be inferred from the model filename.
It can admit literal but semantically incorrect field assignments; evaluate the
exact model/policy pair on labeled sources. Never change validators to obtain a
higher acceptance count. Use non-streaming `LocalExtractor` requests for this
tested interface. The server is an experimental adapter integration, not an
official oMLX adapter configuration feature.
