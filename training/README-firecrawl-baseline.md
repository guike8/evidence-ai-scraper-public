# Freeze a Firecrawl-content model baseline

This optional workflow tests **our extraction contract** on saved Firecrawl
content. It does not train, retrieve URLs, call Firecrawl's native JSON
extraction or establish an advantage over Spark 2. Keep the model, input
preparation and full URL-to-result measurements separate.

Prepare a private labeled case JSONL with `training.evaluate`'s immutable
request/snapshot/gold format. Gold values and evidence must be independently
authored and reviewed before inference, then pass the unchanged production
validators. Snapshots declare retrieval method `firecrawl_self_hosted` for real
adapter retrieval, or `firecrawl_preparation_replay` for separately recorded
preparation-component runs on captured HTML. Never label component replay as
live Firecrawl API retrieval. Raw pages and responses belong outside the source
checkout, with private permissions.

The explicit source controls are:

- `full_record`: exactly the existing local extractor's input policy, including
  its normal source planning and input limits.
- `markdown`: every normalized Firecrawl Markdown section, retaining original
  IDs, with HTML and JSON-LD omitted from the model input. The complete saved
  snapshot remains available to the scorer. Metadata declares the selected
  representation and every omitted section; it never claims full-page coverage.

No view clips source text to fit. Oversized input remains an explicit failure
in the common denominator. A field omitted by preparation remains a recall
opportunity rather than disappearing from the evaluation.

## Freeze before model calls

Use a separately running, verified loopback model server and its exact model ID,
revision and weight checksum. The checksum is an operator declaration; the
supervisor must independently verify the loaded weights. The example below uses
placeholders for those installation-specific values:

```sh
python -m training.firecrawl_benchmark /private/firecrawl-bench/cases.jsonl \
  --model YOUR_LOCAL_MODEL_ID --revision YOUR_IMMUTABLE_REVISION \
  --weights-sha256 YOUR_VERIFIED_WEIGHT_SHA256 \
  --base-url http://127.0.0.1:8004/v1 --source-policy markdown \
  --freeze-candidate /private/firecrawl-bench/run/freeze.json
```

The offline command creates no HTTP client. It binds actual case-file bytes,
case order, gold/snapshot identity, all application/training Python code, exact
schema/prompt/input hashes, view descriptors, endpoint identity and decoding/time
budgets. It prints `freeze_sha256`. Freeze **every arm** of a comparison before
starting any inference. Use separate private directories for models and source
policies. There is no automatic model or policy selection.

```sh
python -m training.firecrawl_benchmark /private/firecrawl-bench/cases.jsonl \
  --model YOUR_LOCAL_MODEL_ID --revision YOUR_IMMUTABLE_REVISION \
  --weights-sha256 YOUR_VERIFIED_WEIGHT_SHA256 \
  --base-url http://127.0.0.1:8004/v1 --source-policy markdown \
  --freeze /private/firecrawl-bench/run/freeze.json --freeze-sha FROZEN_FILE_SHA256 \
  --checkpoint /private/firecrawl-bench/run/raw.json \
  --output /private/firecrawl-bench/run/receipt.json
```

Defaults are temperature zero, thinking disabled, 4,096 output tokens,
48,000 input characters, 45 seconds per request, a 1,200-second hard CLI deadline,
one completion and no repair. The original exclusive-reference grammar remains
unchanged. Source-bound and quote-catalog policies cannot be mixed into this
control. Changed code, cases, model/configuration or inputs fail before inference.

A lifetime lock protects the private checkpoint. Every call is marked in-flight
before sending. Completed cases are skipped on restart; an interrupted ambiguous
call becomes a retained failed case rather than being sent twice. The checkpoint
contains private raw candidates. Public receipts contain counts, trusted field
names, IDs and hashes, without source passages or endpoint paths.

## Read the result

All cases remain in the denominator. Reports separate raw facts, returned
accepted facts, required/optional precision and recall, absent-fact handling,
schema validity, source support, truncation, timeouts, tokens and latency.
Both source controls additionally reject accepted evidence references outside
the **actual model input**, while retaining the raw prediction for factual
scoring. The scorer still uses the unchanged business/evidence validators.

The first corpus is development data: 12 narrow custom-field tasks on previously
captured public pages, 29 positive facts and three optional absence controls.
It uses related inspected sources, not an independently human-reviewed fresh
holdout. Its gold review corrected a browser-title/visible-heading mix-up and
three citation contexts before inference. All 29 final gold facts are supported
by the actual selected Markdown. Changed tasks and input preparation prevent
comparing its score directly with the earlier 13-task real-page replay.

The actual preparation run used the official Firecrawl Rust HTML cleaner and
`parseMarkdown` from a pinned image, with full-page cleaning, no browser, no API
retrieval and no network. It does not exercise the complete API HTML wrapper or
OMCE path. A successful component result does not verify service startup,
asynchronous API behavior or scraping coverage. Full native Firecrawl JSON
extraction and representative unseen-site evaluation remain separate work.
