# Citation training and local inference

The citation curriculum and optional local grammar address separate problems.
Training teaches the model to copy the correct location, quote and field path.
The grammar limits references to locations present in the supplied source. Run
comparisons with the same grammar to measure a weight change; report grammar
changes separately. Neither mechanism establishes arbitrary semantic accuracy.
See [measured results](../training/RESULTS.md) for evaluated artifacts.

## Restrict references to the saved source

Local inference can build a citation catalog from the exact serialized source
view. Text citations may use only included section IDs. JSON-LD citations may
use only existing, non-null scalar RFC 6901 pointers. The two reference forms
remain mutually exclusive. Excluded sections cannot become catalog entries.

This is an explicit opt-in; the default remains disabled:

```sh
export LOCAL_SOURCE_BOUND_EVIDENCE=true
uv run ai-scraper probe --schema page_v1
uv run ai-scraper extract https://example.org/ --source-bound-evidence \
  --output page.json
```

Configure the direct/local profile and start a compatible loopback server as
described in [model selection](model-selection.md). The CLI flag applies to
`extract`, `resume`, `replay` and `probe`. `--no-source-bound-evidence` explicitly
overrides an enabled environment setting. For saved records, semantic replay
also requires `--local-inference`. Environment boolean values are `true`,
`false`, `1` or `0`; an invalid value fails explicitly.

Library callers can use the same policy:

```python
from ai_scraper.extraction.local import LocalExtractor

extractor = LocalExtractor(
    model="your-local-model-id",
    base_url="http://127.0.0.1:8000/v1",
    source_bound_evidence=True,
)
```

Each snapshot gets a fresh catalog and grammar. The policy version
`source-bound-evidence-v1` separates extraction caches from the earlier grammar.
Catalog construction has limits for locators, UTF-8 bytes, traversal nodes,
depth and locator length. Exceeding a limit fails with `CONTENT_TOO_LARGE`;
the adapter does not silently omit references or fall back to weaker grammar.
An empty catalog permits no evidence items.

This mode does not rewrite model values, quotes or returned evidence. The
unchanged production validators still check exact occurrence, supported values,
field meaning, conflicts and completeness. A permitted section can still be
the wrong section for a quote. A literal quote can still refer to another
field or entity. Measure accepted factual precision alongside record acceptance.
Hosted Hugging Face inference retains its separate provider schema.

## Reproduce the citation curriculum

Use an Apple Silicon Mac, Python 3.12 and the pinned training dependencies in
an isolated environment. Keep generated data and runs outside the source
checkout. The example commands run from the repository root and use a new
private experiment directory:

```sh
uv venv --python 3.12 /private/scraper-lab-v2/env
uv pip install --python /private/scraper-lab-v2/env/bin/python \
  -r training/requirements.lock -e .
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 /private/scraper-lab-v2/env/bin/hf download \
  mlx-community/Qwen3-1.7B-4bit \
  --revision 3b1b1768f8f8cf8351c712464f906e86c2b8269e \
  --local-dir /private/scraper-lab-v2/base
/private/scraper-lab-v2/env/bin/python -m training.citation_dataset \
  --output-dir /private/scraper-lab-v2/data
```

The default seed is `2026110603`. The corpus contains 1,800 training, 144
validation and 108 test cases across eight presets and custom fields. Seventy
percent of training cases omit JSON-LD; the manifest records each split's exact
counts. Half of training messages shuffle
section presentation while retaining the actual normalized IDs and exact text;
validation and test messages retain production order. Labels cover nested
leaves, numeric zero and false, relative links, separate short/long descriptions,
and missing start dates with present end dates. Critical-null narratives remove
explicit identity mentions that would contradict an absent-identity target.

Gold values are authored before rendering and do not come from model outputs.
Gold citations refer only to locations supplied in the actual source view.
Snapshots retain the normalizer's IDs and order, so cases can be replayed
against their immutable raw records. Assistant targets use Pydantic property
order and contain only business data, evidence and warnings. Application code
continues to assign provenance and status.

The generator saves split counts, file hashes, its source hash, data provenance
and group-disjoint checks. It does not consume model responses, private pages
or the real-web evaluation corpus. The original `training.dataset` generator
and its reproducible v1 bytes remain unchanged. V2 reuses related synthetic
template and task families with new entities and a new seed; this is not an
independently reviewed or representative web dataset.

## Run bounded training and evaluation

```sh
/private/scraper-lab-v2/env/bin/python -m training.job start \
  --python /private/scraper-lab-v2/env/bin/python \
  --model /private/scraper-lab-v2/base --data /private/scraper-lab-v2/data \
  --run-dir /private/scraper-lab-v2/run-001 \
  --iters 1800 --max-seconds 2400 --memory-gib 8
/private/scraper-lab-v2/env/bin/python -m training.job status \
  --run-dir /private/scraper-lab-v2/run-001
```

The driver uses rank 8 on the last eight layers' q/v projections, batch one,
gradient accumulation four, learning rate 1e-5 and a 4,096-token sequence limit.
The example requests 1,800 microbatches: 450 optimizer updates and one pass over
the default training split. The 8 GiB limit applies to this process's MLX
allocator, not total machine or Python memory. Preflight checks token masking
and rejects overlong rows; a disposable update verifies gradients and exact
adapter reload before training. The detached job has a fixed deadline and
does not stop existing serving processes.

Evaluate the exact saved adapter on validation sources before selecting it.
Freeze weights, source/grammar versions, prompts, decoding, output/input limits
and deadlines before opening fresh synthetic tests or real-web outcomes.
Previously inspected tests serve as regression checks. Retain retrieval,
context-limit, schema, evidence, timeout and truncation failures in their
reported denominators. Report raw predictions separately from accepted facts;
an increase in accepted records is useful only alongside factual precision
and required-field coverage. See the [training guide](../training/README.md)
for scoring, checkpoint limits and artifact publication requirements.
