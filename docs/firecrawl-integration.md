# Firecrawl integration and comparison plan

Firecrawl is a practical retrieval backend and comparison target. This project
has not demonstrated an accuracy or coverage win over it. Reusing its crawler
can let this project focus on local structured extraction and independently
verified evidence instead of duplicating crawling infrastructure.

## Use the self-hosted retrieval adapter

The library, CLI, crawler and configured MCP interface now support backend
`firecrawl`. It calls a separately operated local service and requests only
raw HTML, cleaned HTML and Markdown. Semantic extraction remains in the selected
local model. This does not copy Firecrawl core into this MIT repository.

```sh
export SCRAPER_RETRIEVAL_BACKEND=firecrawl
export FIRECRAWL_BASE_URL=http://127.0.0.1:3002
export FIRECRAWL_RUNTIME_ID=sha256:YOUR_VERIFIED_IMAGE_DIGEST
export FIRECRAWL_TRUST_SERVICE_EGRESS=true
export SCRAPER_ALLOWED_DOMAINS=example.org
uv run ai-scraper extract https://example.org/ --output page.json
```

Replace the digest placeholder with the 64 lowercase hexadecimal characters
of the actual service image digest (or use its exact 40-hex source commit).
The service needs to be running first. No API key or paid-price configuration
is required by this adapter. For semantic gaps, configure the separate loopback
model endpoint as described in [model selection](model-selection.md).

The endpoint must be a literal loopback HTTP(S) address with an explicit port
from 1024 through 65535 and no path, query, fragment or credentials. It sends
no authorization, cookie or proxy-authorization headers and follows no API
redirects or vendor-supplied job URLs. It requests no JSON extraction, action,
webhook, document parser or hosted model format. Country selection and browser
action plans are unsupported in this mode and fail before submission.

`FIRECRAWL_TRUST_SERVICE_EGRESS` is an explicit service boundary: Firecrawl
controls its website connections, redirects and subresources. The application
checks target and reported destination URLs, but cannot pin those remote
connections. The returned warning and provenance describe the destination as
service-reported. Configure the service's public-target egress restrictions
and keep it private. The runtime ID binds caches and resume state; it does not
attest a running container. Pin and verify the actual deployment separately.

Each URL uses a one-URL asynchronous batch. The adapter persists submission
state before POST and the batch ID before polling. Restart polls the same ID.
An ambiguous POST response stays `RECONCILIATION_REQUIRED`; recover its ID with
`resume JOB_ID --snapshot-id BATCH_UUID` rather than submitting again. Only
explicit submission rejection, such as 429, permits a bounded retry. A changed
endpoint, runtime or service-policy identity cannot resume an existing job or
crawl silently. Existing non-Firecrawl resume identities remain unchanged.

## Verification scope

The new adapter is covered by fixture tests for source mapping, asynchronous
completion, restart, per-record errors, response limits, rejected destinations,
credential stripping, reconciliation and cache/resume identity. The actual
full self-host API was attempted in a separate bounded Docker lab, but could
not start within its spare memory. No scrape jobs were submitted there.

The official Firecrawl Rust HTML cleaner and Markdown conversion component
were run successfully on 12 previously saved public pages in a networkless,
read-only 384 MiB container. This is actual component execution, not a complete
API, browser or URL-retrieval test. The official API image was pinned by ARM64
digest `sha256:6157ce24e6867f013704c1cf2b752ec1ea7a3f730da09014b511d1049c8651af`,
with build SHA `7666c1f9ae8720a6bba271e0f60b6a217f8a5210` (tag 2.11.162).
That runtime differs from the source-inspection commit below.

The exact request body also passed the actual image's bundled batch-request
schema in a networkless test. The adapter explicitly sends
`skipTlsVerification: false` and `parsers: []`: the tested image otherwise
defaults to skipping certificate verification and enabling PDF parsing.
This verifies request-schema compatibility, not a working HTTP API.

A [frozen model/source benchmark](../training/README-firecrawl-baseline.md)
compares the unmodified model and prior checkpoint over the same saved inputs,
using the full record and selected Firecrawl Markdown separately. It is our
extraction contract on a development corpus, with factual and citation scoring.
It is not a comparison against Firecrawl's native JSON extraction or Spark 2.

The [completed baseline](../training/RESULTS-firecrawl.md) returned 0/12
accepted cited records in all four model/source arms. A separately frozen
business-only diagnostic yielded 6/12 exact records and 22/29 correct facts
for the base model, versus 5/12 and 20/29 for the existing citation fine-tune.
It produced no citations and changes multiple input/output factors; it is
exploratory development evidence, not a production result or a causal test
of citation overhead. No further weights were trained for these measurements.

## What the current model result means

The citation-v3 adapter accepted 61/108 cases on the previously inspected
synthetic regression set, compared with 41/108 for the old adapter under the
same exclusive-reference grammar and validators. On the newly authored
108-case synthetic set, it accepted 29/108. Both new-model runs had zero wrong
accepted facts against the authored labels. The old model accepted 16/108 on
that same fresh set, with two wrong accepted facts.

Both weights also accepted 0/13 eligible real-page replays (0/18 attempted
including five pre-inference blockers). The new model exhausted its output
budget on nine tasks. This is an extraction and instruction-following failure,
not evidence of competitiveness with Firecrawl.

The regression gain measures improvement over our old weights. The fresh-set
result exposes substantial remaining coverage gaps. Neither result compares
against Firecrawl, represents arbitrary web accuracy, or qualifies a general
production claim. See [complete experiment results](../training/RESULTS.md).

## What can be reused

Source inspection is pinned to Firecrawl commit
[`edf02642c8f002576e9f26b9bbbb7b327313558c`](https://github.com/firecrawl/firecrawl/tree/edf02642c8f002576e9f26b9bbbb7b327313558c).
Its [self-hosted stack](https://github.com/firecrawl/firecrawl/blob/edf02642c8f002576e9f26b9bbbb7b327313558c/SELF_HOST.md)
includes fetch/Playwright processing, API workers and supporting queues.
The [official deployment comparison](https://docs.firecrawl.dev/contributing/open-source-or-cloud)
distinguishes these core capabilities from Cloud-only product surfaces and
separately configured advanced services. Forking the repository does not
provide Firecrawl Cloud's managed proxy infrastructure or the hosted Spark-2
model weights. The repository's
[Agent examples](https://github.com/firecrawl/firecrawl/blob/edf02642c8f002576e9f26b9bbbb7b327313558c/README.md#agent)
describe a hosted model selection, not a downloadable training checkpoint.

The recommended first integration is a separately operated self-hosted
Firecrawl service behind a retrieval adapter:

`Firecrawl retrieval → saved source snapshot → local model → our validation → JSON`

Keep returned HTML/text, final URL and retrieval metadata in the immutable
snapshot. Derive evidence identifiers and provenance in application code.
Markdown cleanup alone must not become proof that an extracted value has the
correct field or entity meaning. Replay should use the saved bytes and retain
source-selection details so citations can be checked independently.

Keep the independently written MIT library and model service separate from
Firecrawl's core. Its core uses
[AGPL-3.0](https://github.com/firecrawl/firecrawl/blob/edf02642c8f002576e9f26b9bbbb7b327313558c/LICENSE);
the SDKs have separate licensing. An AGPL fork can hold necessary upstream
changes with its notices and source obligations preserved. Do not copy core
code into this MIT repository and relabel it. Start with the service boundary;
fork only when a specific integration change requires it.

## Local model routing needs a real test

Firecrawl already supports configurable model providers. Its
[provider implementation](https://github.com/firecrawl/firecrawl/blob/edf02642c8f002576e9f26b9bbbb7b327313558c/apps/api/src/lib/generic-ai.ts#L28)
has `OPENAI_BASE_URL`, `MODEL_NAME`, `OLLAMA_BASE_URL` and a dedicated
`EXTRACTION_PROVIDER_*` Chat Completions path. The dedicated path does not
require a real provider key for a local endpoint.

There is an important routing distinction: ordinary JSON extraction explicitly
selects the OpenAI provider, choosing `gpt-4o-mini` for simple schemas and
`gpt-4.1` for schemas containing `$ref`/`$defs`. The dedicated extraction
provider can fall back to the default provider when it fails or returns empty
lists. See the
[selection and fallback code](https://github.com/firecrawl/firecrawl/blob/edf02642c8f002576e9f26b9bbbb7b327313558c/apps/api/src/scraper/scrapeURL/transformers/llmExtract.ts#L1061).
Setting `OLLAMA_BASE_URL` alone does not prove that every extraction request
runs locally. Test the exact route, prohibit external inference fallback, and
verify the actual schema before selecting our model. The pinned
[Compose environment](https://github.com/firecrawl/firecrawl/blob/edf02642c8f002576e9f26b9bbbb7b327313558c/docker-compose.yaml#L24)
passes the older local-model settings but needs an explicit override for the
new dedicated extraction-provider variables.

The reviewed
[source tracker](https://github.com/firecrawl/firecrawl/blob/edf02642c8f002576e9f26b9bbbb7b327313558c/apps/api/src/lib/extract/helpers/source-tracker.ts#L103)
maps extracted/merged array items to source URLs. That is useful provenance,
but this path does not impose this project's compulsory per-leaf literal
excerpt and saved-location contract. This observation is limited to the
reviewed path; it is not a claim that Firecrawl lacks all citation features.

## What the next comparison must establish

The target is accurate extraction for previously unseen public websites and
user-specified schemas, with correct abstention and verifiable citations.
Arbitrary nested-schema support remains a gap in our current custom interface;
unsupported schemas must count as unsupported, not disappear from evaluation.

1. **Extraction on identical saved sources.** Freeze independently reviewed
   pages, labels and schema requests before training. Run Firecrawl's extraction
   pipeline and ours with the same local weights, hardware, source bytes,
   output budget and deadline. Record their respective prompts and processing;
   use a separate fixed-prompt control when attributing gains to weights.
   Compare raw and returned business accuracy before applying each evidence
   contract. Report citation verification and evidence coverage separately so
   a tool without our reference format can still receive business-fact credit.
2. **Whole URL-to-result operation.** Compare retrieval/rendering coverage,
   long-document handling, structured extraction, retries and restart recovery
   from the same public URL tasks. Record versions, cache policy, cold/warm
   state, resource use, latency p50/p95 and model tokens. A retrieval failure
   must remain in the end-to-end denominator.

Both tracks must report required/all-field precision and recall, wrong accepted
facts, duplicate entities, absent-fact abstention and accepted record coverage.
Retain timeouts, refusals, malformed output, unsupported schemas and source-limit
failures in their applicable denominators. Separate model changes from grammar,
prompt, source planning and retrieval changes. A small model trained on our
synthetic templates cannot be called better than Firecrawl's hosted models
without a matched, disclosed comparison.

The adapter and preparation-component verification above are completed work;
full API retrieval, native Firecrawl JSON extraction and the broader comparison
remain unverified. This work makes no Firecrawl Cloud calls, requires no paid
provider credentials and changes no Studio network settings. The Studio's existing VPN protection remains intact;
live retrieval there still requires permitted outbound connectivity. Public
captures replayed locally must be labeled as replay, not live Studio fetching.
