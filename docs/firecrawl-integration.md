# Firecrawl integration and comparison plan

Firecrawl is a practical retrieval backend and comparison target. This project
has not demonstrated an accuracy or coverage win over it. Reusing its crawler
can let this project focus on local structured extraction and independently
verified evidence instead of duplicating crawling infrastructure.

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

This document describes the next integration and evaluation work. It does not
claim that Firecrawl has been deployed or benchmarked here. It makes no
Firecrawl Cloud calls, requires no paid provider credentials and changes no
Studio network settings. The Studio's existing VPN protection remains intact;
live retrieval there still requires permitted outbound connectivity. Public
captures replayed locally must be labeled as replay, not live Studio fetching.
