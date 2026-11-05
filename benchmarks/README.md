# Credentialless comparison protocol

This harness compares structured predictions against reviewed scalar labels for
the registered page, article, organization, person, job, event, listing, product
and request-specific custom contracts. It also runs the real application
pipeline with a fixture retrieval adapter, fresh temporary state and **no model,
DNS lookup or network connection**. Environment credentials are not loaded.

The 19-case `fixtures.json` corpus is authored synthetic data, not a production
accuracy benchmark. It includes nested tables, links, JSON-LD, missing facts,
entity conflicts, long navigation, challenges, a JavaScript shell, zero/false
values and two existing product regression examples. A declared positive field
does not disappear from recall when the operation fails. In particular, the
custom scalar case needs semantic extraction: the deterministic runner exposes
those misses. A JavaScript shell remains a retrieval failure in this track.

## Run and score

From the repository root with its development environment installed:

```sh
uv run python -m ai_scraper.competition run benchmarks/fixtures.json \
  --candidate evidence-current --candidate-version YOUR_COMMIT \
  --output /tmp/evidence-current.json

uv run python -m ai_scraper.competition score benchmarks/fixtures.json \
  /tmp/evidence-current.json --output /tmp/comparison.json
```

`run` measures cold application initialization, extraction, validation and local
state cleanup per case. Preparing the common manifest/source identities is
outside each case clock. Total receipt wall time is also recorded. The
`pipeline_reported` component comes from the application's usage field; it is
separate from the runner's wall clock. No vendor, browser, live-page or model
performance is implied by this mode.

To compare an earlier code revision, execute the same harness file with
`PYTHONPATH` pointing at that revision's `src` directory and use the same Python
environment. The harness uses absolute `ai_scraper` imports for this purpose:

The historical Python baseline in the saved receipts can be reconstructed from
the public source patch without importing any earlier repository history:

```sh
uv run python benchmarks/reconstruct_baseline.py /tmp/evidence-baseline
```

This creates a new local directory outside the checkout, copies Python source
and applies the reviewed reversal patch in an empty local Git repository. It
does not configure a remote or access an account. The reconstructed Python
implementation hash and all 19 predictions were checked against the baseline
receipt. Use `/tmp/evidence-baseline/src` as the baseline `PYTHONPATH` below.

```sh
PYTHONPATH=/path/to/baseline/src python /path/to/current/src/ai_scraper/competition.py \
  run /path/to/current/benchmarks/fixtures.json \
  --candidate evidence-baseline --candidate-version BASELINE_COMMIT \
  --output /tmp/evidence-baseline.json

python -m ai_scraper.competition score benchmarks/fixtures.json \
  /tmp/evidence-baseline.json /tmp/evidence-current.json \
  --output /tmp/comparison.json
```

Keep the measurement harness, manifest, interpreter, hardware and state policy
constant. Receipts include implementation and harness hashes. The runner's
deadline uses a cooperative async timeout plus a post-completion elapsed-time
check for synchronous CPU work. Late work is retained in latency and counted as
a failure; this is not an OS-level preemptive deadline.

## Independent tool receipts

Collectors for another tool must measure that tool independently and export
`snapshot-receipt-v1`. The scorer never runs another tool or invents results.
Required top-level fields are `format`, `corpus_id`, `manifest_hash`, `metadata`
and `results`. Candidate metadata must explicitly include:

- `candidate`, exact `version`, `mode`, `runtime` and `hardware.system/machine`;
- `track: "identical_snapshot"` and `cache_state` (`cold`, `warm`, `disabled`,
  `mixed` or `unknown`);
- finite positive `deadline_seconds`;
- `model` and `provider` (JSON null when no model was used or not measured);
- `network_calls` (a measured nonnegative integer, or JSON null if unknown).

Each result supplies `case_id`, `source_hash`, `request_hash`, `status` and
business `data`. `case_context(case)` returns the canonical request, trusted
business contract, saved source and request hash; `snapshot.source_hash` is the
common raw-record hash. The manifest hash is SHA-256 over UTF-8 JSON using sorted
keys, compact separators, non-ASCII text preserved and non-finite values
rejected. Changing a task/schema/source invalidates a receipt. Changing an
implementation's model, prompt or source planner is an eligible comparison and
belongs in candidate metadata, not in the common request identity.

Results may also provide:

- `wall_latency_ms`, or JSON null for unknown;
- `component_latency_ms`, containing only actually measured components;
- `retrieval_usable`, explicitly true/false when measured;
- `usage.input_tokens`, `output_tokens`, `inference_attempts` and
  `retrieval_attempts`, with nonnegative integer counts or null/omission for
  unknown;
- `evidence` in the application's candidate reference format, as a list or a
  map of field paths to references.

Unknown cases, duplicate outcomes and mismatched input hashes are rejected.
Missing outcomes become explicit failed misses; failed and pending payloads
cannot supply accepted business data. Unknown timing, usage and retrieval
measurements remain unknown rather than becoming zero or a successful scrape.

## Read the scores

`expected_fields` contains complete manually specified scalar labels using
paths such as `title`, `items.0.name` and `tables.0.rows.1.2`. Null and structural
empty-string labels are not positive recall opportunities. **Zero and false
remain positive facts.** Values compare exactly, including Python/JSON scalar
type: false, zero, `0.0` and `"0"` are distinct predictions. Lists retain their
declared order.

The report separates all, required and optional fields, plus schema/domain
groups. Required roots come from the trusted registry contract unless the case
explicitly names `required_paths`. Failed, pending, late and missing outcomes
retain every present gold field in recall. Extra or incorrect populated values
count as label mismatches (`unsupported_populated_facts`); this name describes
the reviewed corpus, not a universal factual guarantee.

Source evidence is assessed independently using the production schema,
completeness, identity, conflict and exact-source validator. Coverage counts
fields accepted by that whole-record gate; partial per-field successes within a
rejected record are not counted. A tool without references can still score
business accuracy while receiving zero verified-evidence coverage. Conversely,
an exact quote can pass source verification yet answer the wrong question and
fail the labels. A test explicitly preserves this distinction.

Latency uses nearest-rank p50 and p95 over measured wall samples, including
failed and late attempts. Component samples and token/attempt counts have their
own measured-case counts. Do not compare parser microseconds with network or
model end-to-end latency, rank different cache policies together, or turn this
small authored corpus into a claim of overall superiority.

URL-to-result/browser coverage needs a separate live track with reviewed
snapshots, redirects, resource policy and deadlines. It is intentionally rejected
by this identical-snapshot scorer. External human-reviewed datasets should be
reserved as held-out evaluations, with their revision/license recorded and
without tuning on their test labels.

## Context and selector component experiments

`context_fit.py` measures serialized source fit without retrieving a URL or
calling a model. Run the same script/interpreter/manifest against both source
trees through `PYTHONPATH`, as above:

```sh
uv run python benchmarks/context_fit.py benchmarks/fixtures.json \
  --candidate context-current --limit 48000 --output /tmp/context-current.json
```

Characters describe the source JSON payload; they are not a tokenizer result or
an inference speed measurement. Selection scope, original evidence IDs and
source identity remain inspectable in the receipt and application provenance.

`selector_components.py` measures only parser/selector components on six authored
HTML cases with the same predefined CSS selectors. Use separate environments:
BeautifulSoup 4.15.0, Scrapling 0.5.1, and Crawl4AI 0.9.5. Invoke each environment's
Python with `beautifulsoup`, `scrapling` or `crawl4ai`, respectively:

```sh
python benchmarks/selector_components.py scrapling \
  --state-dir /tmp/selector-state --warmups 3 --iterations 25 \
  --output /tmp/selector-scrapling.json
```

The script rejects incorrect output before ranking speed, records hashes and
timing samples, and separates import time and whole-process RSS. A Python socket
guard blocks and counts attempted networking; it is not OS isolation. It does
not fetch, render, infer schemas, run a model or verify evidence. Compare those
component results separately from the application's end-to-end operation. See
the [competitive review](../docs/competitive-review.md) for measured limits and
the next comparison gates.
