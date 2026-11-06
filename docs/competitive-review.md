# Competitive review — November 5, 2026

The target is fast local extraction of correct, verifiable business facts, with
inspectable recovery and explicit failure behavior. The reviewed implementations already cover many of our
features. A broad superiority claim needs paired measurements on representative
pages and schemas.

The current priority is [reusing and directly comparing self-hosted Firecrawl](firecrawl-integration.md),
with our local model and independent evidence contract. The updated model's
synthetic gains do not establish an overall win; its eligible real-page replay
fails all 13 tasks. See the [model measurements and failures](../training/RESULTS.md).
The component and fixture measurements below remain separate historical tracks.

| Implementation checked | Established strengths | Implication for this project |
| --- | --- | --- |
| [Firecrawl](https://docs.firecrawl.dev/contributing/open-source-or-cloud), published release 2.11.0; self-host guide pins 2.11.162 | Browser-backed scraping, local-model self-hosting, schema extraction; current Cloud examples allow starting without an API key. The release also advertises reusable deterministic JSON extractors. | Credentials and deterministic-first extraction are shared capabilities. Cloud and self-hosted features require separate comparisons. |
| [Crawl4AI 0.9.5](https://github.com/unclecode/crawl4ai/releases/tag/v0.9.5) | Playwright, local LLMs, CSS/XPath/regex, content filtering, chunking and crawl checkpoints. | Browser coverage, large-document handling and reusable selectors are important gaps. Its hardened deployment protections must be distinguished from bare library defaults. |
| [Scrapling 0.5.1](https://github.com/D4Vinci/Scrapling/releases/tag/v0.5.1) | Fast parsing, adaptive selectors, HTTP/browser sessions, crawling, checkpoints and MCP tools. | Reuse efficient parsing rather than compete with native parsers through Python DOM work. |
| [ScrapeGraphAI 2.3.1](https://github.com/ScrapeGraphAI/Scrapegraph-ai/releases/tag/v2.3.1) | Local models, browser-backed graphs, typed extraction and published extraction-specific model/data work. | Benchmark small task models and general nested schemas. Disable optional telemetry for local-only evaluation. |

These are official-source observations. Firecrawl's release notes describe
[deterministicJson](https://github.com/firecrawl/firecrawl/releases/tag/v2.11.0),
and its [JSON extraction documentation](https://docs.firecrawl.dev/features/llm-extract)
describes keyless Cloud entry. Anonymous Cloud behavior and limits were not
exercised here. None of those documented features establishes a compulsory
per-leaf immutable-source validation contract equivalent to ours; that contract
is a useful evaluation lane, rather than proof of an overall lead.

Published headline numbers use different tasks. Firecrawl's publisher-run
[benchmark](https://www.firecrawl.dev/benchmarks) reports 96% retrieval coverage,
but coverage requires only 10% of expected core text; reported extraction F1 is
0.638. Scrapling's published milliseconds concern a
[parser microbenchmark](https://github.com/D4Vinci/Scrapling/blob/main/benchmarks.py).
Neither number measures our complete schema, evidence and recovery operation.

The first implemented improvements address reproduced defects:

1. Text limits count main-content candidates after scope filtering. Previously,
   2,100 navigation paragraphs could consume the budget before a useful main
   paragraph, producing a critical missing-text failure.
2. General inference uses an immutable source view with original evidence IDs.
   Duplicate text representations collapse losslessly. Oversized page/article
   tasks can use one explicit main/article scope. All JSON-LD, metadata,
   attributes and other retained representations remain available. Custom tasks
   retain full-page coverage so footer facts cannot disappear silently.
3. Application code persists the selection descriptor before a model call,
   attaches a compact provenance summary, versions extraction-cache identity,
   validates cached output against the full source and removes descriptors under
   the existing retention policy. Product behavior remains compatible.

The new [comparison harness](../benchmarks/README.md) accepts independent tool
receipts, binds them to identical source/request identities, and separates
business correctness from evidence acceptance. Failures, pending results and
missing predictions remain in recall. Its 19-case corpus is authored synthetic
data; it is not a held-out live-web accuracy benchmark.

| Paired deterministic fixture measure | Historical baseline | Context update |
| --- | ---: | ---: |
| Correct labelled scalar leaves | 74 / 80 | 76 / 80 |
| Required-field recall | 34 / 39 (87.2%) | 35 / 39 (89.7%) |
| Optional-field recall | 40 / 41 | 41 / 41 |
| Unsupported accepted facts | 0 | 0 |
| Accepted facts passing production evidence validation | 74 / 74 | 76 / 76 |
| Source payloads fitting a 48,000-character limit | 17 / 19 | 19 / 19 |

The 2,100-navigation source payload changed from 104,099 characters, which was
rejected, to 352 characters with recorded main scope. Planning took a median
14.17 ms across seven local runs. This measures source-payload fit; it is not a
token count or model speedup. Four labelled leaves remain missed: a critical
article-conflict case and a custom zero/false case run without any model. The
JavaScript shell and challenge controls remain explicit content failures.

The [paired receipts](../benchmarks/results/paired-context-results.json) and
[source-fit measurements](../benchmarks/results/context-context-update.json)
retain input hashes and comparison scope. Historical version labels in those
receipts identify a measured implementation; the
[comparison protocol](../benchmarks/README.md) explains baseline reconstruction
without importing deployment or private repository history. The updated application
passed 519 local tests, Ruff checks and formatting; the source-context diff
received a static/fixture security review without actionable findings. These
fixture checks do not establish performance on a representative live-web corpus.

A separate local component experiment used identical authored HTML and explicit
CSS selectors. All three components returned the expected data on all six cases.
Runs were sequential, with three warmups and 25 measured iterations per case:

| Parser/selector component | 2,100-navigation p50 | 20 × 10 table p50 |
| --- | ---: | ---: |
| BeautifulSoup 4.15.0, html.parser | 11.444 ms | 1.583 ms |
| Scrapling 0.5.1 | 0.412 ms | 0.405 ms |
| Crawl4AI 0.9.5, JsonLxmlExtractionStrategy | 0.376 ms | 0.516 ms |

This compares components with predefined selectors. It excludes fetching,
browser rendering, automatic schema inference, model calls, evidence validation
and durable jobs. The [reproduction script](../benchmarks/selector_components.py)
records input/schema/output hashes, import time and process RSS separately.
Python socket guards observed no network attempts; they are not OS isolation.
The [component receipts](../benchmarks/results/selector-summary.json) preserve
the samples, configurations and measurement limits.

The next measurable improvements are:

1. **Smaller local inference.** The public Apache-2.0
   [sgai-qwen3-1.7b](https://huggingface.co/scrapegraphai/sgai-qwen3-1.7b)
   candidate is pinned at `fadf46f3e11cf650f281019fa39211acf058681c`.
   A candidate listing, model-card result or download does not establish
   compatibility with this application's business-and-evidence schema. Require
   successful production-validator probes and paired task measurements before
   recommending a model; no small-model speed or accuracy result is asserted here.
2. **Native parsing and safe rendering.** Evaluate a versioned native-parser
   backend while preserving old snapshot normalization and evidence IDs. Add
   browser coverage through an adapter that enforces equivalent URL, redirect,
   DNS and subresource policy. Current Playwright competitors cover cases our
   direct-only profile cannot retrieve.
3. **Broader schema and holdout coverage.** Our custom interface has at most
   32 scalar/string-list fields; competitors accept broader nested schemas.
   Count unsupported schemas explicitly. The external
   [ScrapeGraphAI evaluation work](https://github.com/ScrapeGraphAI/scrapegraph-100k-paper)
   provides human-reviewed cases to reserve for a final holdout. Teacher-generated
   labels and schema compliance alone cannot establish factual superiority.

Before a public performance claim, run the same model and hardware, frozen
reviewed sources, declared schemas, equal deadlines and token budgets, cold/warm
trials and identical evidence rules. Report retrieval coverage, required-field
precision/recall, unsupported values, duplicate answers, end-to-end p50/p95,
model tokens, resource use and restart recovery separately. No current result
establishes an overall win over these implementations.
