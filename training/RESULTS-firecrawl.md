# Firecrawl preparation and local-model baseline

All four extraction arms accepted **0/12 records**. The existing citation-v3
adapter did not improve factual extraction over the stock model on these tasks.
This phase added integration and measurement; it performed **no new training or
weight upload**. These results do not establish superiority over Firecrawl.
Machine-readable counts and pinned identities are in
[firecrawl-baseline-metrics.json](firecrawl-baseline-metrics.json).

## What was measured

Twelve narrow custom-schema tasks use 12 previously captured public URLs across
10 hosts: documentation, organizations, people, a release, an event, a product
and a job. Gold contains 29 populated scalar leaves and three optional null
controls. This is a **development corpus**, not fresh retrieval, an unseen-site
holdout or a public benchmark dataset. Captures, task labels and raw model
completions are not redistributed.

Gold was authored and reviewed before inference. Review corrected one personal
page's visible-heading value instead of its document title and moved three
citations to the intended mission, founding-year and foundation-heading context
rather than navigation or unrelated text. All 12 gold candidates passed the
unchanged application validators; all 29 positive facts occur in the actual
converted Markdown. Independent review checked all positive bindings and absent
controls. No inference output was consulted before the gold freeze.

The pinned Firecrawl image's actual `transformHtml` and `parseMarkdown`
components converted **12/12 saved HTML inputs**, with zero parser warnings or
errors. Conversion used a read-only, network-disabled container limited to
384 MiB, native HTML cleaning enabled and `onlyMainContent=false`. It did not
exercise URL retrieval, browser rendering, the full HTML wrapper or OMCE.

Full Firecrawl API startup failed within the safe 2 GiB test budget. The test
stack and its empty volumes were removed, other services were preserved, and
**zero API scrape jobs were submitted**. Studio VPN and egress settings remained
unchanged. Separately, the actual pinned image's Zod `batchScrapeRequestSchema`
accepted the final adapter's exact nine-field request, including explicit
`skipTlsVerification=false` and `parsers=[]`; an unknown field was rejected.
This networkless schema check establishes request compatibility, not a live
endpoint, async-job or URL-retrieval success.

## Frozen four-arm matrix

The stock model is
[mlx-community/Qwen3-1.7B-4bit](https://huggingface.co/mlx-community/Qwen3-1.7B-4bit/tree/3b1b1768f8f8cf8351c712464f906e86c2b8269e).
The existing adapter is
[citation-v3 MLX LoRA](https://huggingface.co/guike8/evidence-scraper-qwen3-1.7b-citation-v3-mlx-lora/tree/b0b97c7dbfda946e982073984888fe5759351836),
loaded over that same base. The isolated serving supervisor verified 704
unchanged base tensors, exact reload of 32 adapter tensors and invocation of all
16 adapted modules. The serving profile used an internal adapter hook; existing
model services were not changed, and experiment servers were stopped afterward.

All four arms were frozen before model calls. Each used one completion,
temperature zero, thinking disabled, a 4,096-token output ceiling, 48,000 source
characters, 45 seconds per request and 1,200 seconds per arm. There were no
retries or repairs. Full-record input kept the existing representation;
Markdown input kept original numbered Markdown sections and declared excluded
HTML/JSON-LD content. Neither policy clipped oversized sources. Both used the
original exclusive citation grammar and rejected evidence locators outside the
actual supplied input.

| Model | Input | Source fits | Schema-valid candidates | Raw correct / predicted facts | Wrong or unlabeled | Truncated | Accepted records |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock | Full record | 10/12 | 5/12 | 10/14 | 4 | 5 | 0/12 |
| Stock | Numbered Markdown | 12/12 | 5/12 | 8/14 | 6 | 7 | 0/12 |
| Citation-v3 | Full record | 10/12 | 6/12 | 2/2 | 0 | 4 | 0/12 |
| Citation-v3 | Numbered Markdown | 12/12 | 8/12 | 3/8 | 5 | 4 | 0/12 |

Every arm retains the same 12-case, 29-positive-fact denominator, including the
two oversized full-record cases. The citation-v3 Markdown arm also had one
`EVIDENCE_OUTSIDE_INPUT` failure. Raw factual counts measure typed exact leaf
agreement in the unmodified candidate, including candidates rejected by the
application. Schema validity alone does not establish factual or citation
correctness. All four arms returned zero accepted facts and zero correctly
source-supported facts. A small raw prediction count with no wrong values is
not useful coverage: citation-v3 full-record found only 2/29 labeled facts.

## Separate exploratory business-only diagnostic

After inspecting the matrix failures, a separate diagnostic was declared and
both model arms frozen before its calls. It reused the same 12 development
cases and budgets, supplied **raw unnumbered Markdown**, and requested only the
business data schema. Trusted task descriptions had citation clauses removed.
No citations were produced, no evidence was assigned automatically and every
application-contract result was deliberately rejected for missing evidence.

| Model | Exact business records | Business-schema valid | Typed correct / predicted facts | Wrong or unlabeled | Correct null controls | Accepted contracts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock | 6/12 | 12/12 | 22/30 | 8 | 2/3 | 0/12 |
| Citation-v3 | 5/12 | 12/12 | 20/29 | 9 | 3/3 | 0/12 |

Stock factual precision/recall were 73.33%/75.86%; citation-v3 was
68.97%/68.97%. An exact record must match the complete raw business object,
including types, keys and nulls. Both arms completed without token truncation.
These are exploratory raw-data diagnostics, not usable cited extraction.
The prompt, schema and source representation changed together after matrix
inspection, so the difference from wrapped extraction does **not isolate a
causal effect of citations**. Nor does it demonstrate an improved model.

## Reproduction and remaining work

The frozen application commit was
[`468bab67cfe7cf87c45402ece4ab7c6285ac8c66`](https://github.com/guike8/evidence-ai-scraper-public/tree/468bab67cfe7cf87c45402ece4ab7c6285ac8c66),
with 44 Python files bound by tree SHA256
`87257f3ba57831f0c70eee2bcaf18f74acd21936d6f8bca4456e77ba23f55dc7`.
The final release adapter has subsequent TLS/parser hardening; the model runs
retain their original frozen source identity.

Key immutable hashes:

- Case-file SHA256: `68cb32b7b42365dfcb585a64337faf9f4cf5be864ef28c7531b6f41fd4e88859`.
- Reviewed label-plan SHA256: `88fa869739c2beb032da8d5dbf532be84ac7f9b3f2d6e05a37f47e86f5d8b729`.
- Four-arm matrix-plan SHA256: `3b896d5bdf55097ab7a90bc9cf143e3fef06d82c4ead88f50fd93ab4668bf1c7`.
- Business-plan SHA256: `96bc60e29c5b4e103efec57dae5596fe18dce8bc8cdb38a5e2f30e8c729c5cb4`.
- Unchanged published diagnostic SHA256: `439c1aa26c1a7a7cbcc54bbad0c0b3ad92d9491cdd8b0bca95c840b7f39b204b`.

Actual preparation image: `ghcr.io/firecrawl/firecrawl:2.11.162`, ARM64 digest
`sha256:6157ce24e6867f013704c1cf2b752ec1ea7a3f730da09014b511d1049c8651af`,
reporting build `7666c1f9ae8720a6bba271e0f60b6a217f8a5210`. This image build is
separate from the application commit above. Component output SHA256 is
`b65f0811f0c0075760955d6d347ba8ce72f1a504dc51303cc1b2f52d9b568d9a`.
Per-arm freezes, receipt hashes, model artifact hashes, backend versions and
request-schema proof identities are preserved in the metrics JSON.

The [baseline guide](README-firecrawl-baseline.md) describes the public wrapped
runner. The historical business-only runner is available as
`python -m benchmarks.firecrawl_business_diagnostic --help`; it requires an
offline freeze and its checksum before inference. Public mock tests use newly
authored synthetic pages and test mechanics, not these historical scores.
Exact historical score reproduction additionally requires the original private
corpus and isolated serving profile.

The remaining live check is full API retrieval and restart recovery on a host
with enough safe memory and authorized outbound access. Model work first needs
a larger reviewed business-extraction corpus with separate training,
validation and unseen-site evaluation, plus independently validated evidence.
The earlier 13-page replay and these changed 12 tasks cannot be compared as a
quality trend. This baseline does not compare Firecrawl's native JSON
extraction pipeline or establish a competitive accuracy win.
