# Real-page local-model results

V6 did not meet the preregistered validation gate. The reserved site test is reported separately. This small pilot does not establish production or universal-web accuracy.

The pipeline extracts business JSON, then selects application-assigned source-evidence IDs and runs unchanged validators. Raw facts, complete exact business records, accepted facts and wrong accepted facts measure different things.

There are 48 saved public pages across 18 site families: 32 TRAIN, eight validation and eight reserved site-test pages. The 96 original requests contain 220 positive scalar leaves and 22 optional absence controls. Source-reviewed agent gold predates V4 inference;12 prior DEV pages are TRAIN-only. Captures and raw completions stay private.

## Validation history

Validation was reused for curriculum design and selection. V4 and V5 generated no reserved site-test outputs.

| Run | Raw correct / gold | Raw wrong | Exact business | Accepted cases | Accepted correct / predicted | Wrong accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 base | 28/36 | 7 | 9/16 | 11/16 | 23/26 | 3 |
| V4 candidate | 27/36 | 9 | 9/16 | 14/16 | 26/32 | 6 |
| V5 base | 28/36 | 7 | 9/16 | 11/16 | 23/26 | 3 |
| V5 candidate | 28/36 | 8 | 10/16 | 3/16 | 5/7 | 2 |
| V6 base | 28/36 | 7 | 9/16 | 11/16 | 23/26 | 3 |
| V6 candidate | 26/36 | 8 | 8/16 | 13/16 | 24/30 | 6 |

An accepted record can still contain a wrong fact. V5’s 13 grounding rejections included 10 unnecessary nulls on correct values with reviewed evidence available, rejecting eight entirely correct business records. Its one-null pattern supports an over-abstention hypothesis; it does not prove a training-balance cause.

## Final reserved site test

| Model | Raw correct / gold | Raw precision / recall | Exact business | Accepted cases | Accepted correct / predicted | Accepted precision / recall | Wrong accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 27/38 | 67.5% / 71.1% | 9/16 | 14/16 | 27/36 | 75.0% / 71.1% | 9 |
| candidate | 25/38 | 71.4% / 65.8% | 8/16 | 14/16 | 25/31 | 80.6% / 65.8% | 6 |

All 16 cases and 38 full-page test facts remain in the denominator, including four reviewed anchors outside the fixed input windows. Only the frozen V6 candidate and base generated final-site outputs; no post-test tuning followed.

## Citation mechanics and runtime

| Arm | Valid IDs / catalogued fields | Grounding nulls | Invalid IDs | Anchor proxy | Gold anchors in input | Raw null controls | Tokens in / out | p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test-base | 37/40 | 3 | 0 | 23 | 34/38 | 4/4 | 51384 / 1152 | 897 ms / 3911 ms |
| test-candidate | 33/35 | 2 | 0 | 20 | 34/38 | 4/4 | 51274 / 785 | 1996 ms / 6402 ms |
| valid-base | 31/35 | 4 | 0 | 20 | 33/36 | 4/4 | 48385 / 950 | 908 ms / 5781 ms |
| valid-candidate | 33/34 | 1 | 0 | 18 | 33/36 | 4/4 | 48972 / 755 | 1940 ms / 6274 ms |

Literal reference validity does not establish field meaning, entity identity or ordered-list correctness. Anchor agreement is a proxy; valid alternative citations exist. Latency includes rejected cases and warm-up, so it is not accepted-correct throughput.

## Training and lineage

| Candidate | Rows: business / grounding | Microbatches / updates | Learning rate | Elapsed | Peak MLX allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| V4 | 64 / 76 | 512 / 128 | 3e-05 | 191.4 s | 5.65 GiB |
| V5 | 274 / 127 | 1024 / 256 | 2e-05 | 386.6 s | 5.62 GiB |
| V6 | 274 / 286 | 1536 / 384 | 1e-05 | 529.8 s | 5.65 GiB |

V6 starts from the original V4 parent: 274 business rows paired with 274 positive grounding targets, plus 12 original reviewed counterfactual controls (560 rows). The positive pairs include 32 empty targets arising from declared source omissions. Projections/aliases add no pages or independent gold facts.

V5 history alone includes a tokenizer-preflight failure on 402 rows: one 12,579-token counterfactual exceeded 8,192 tokens and was removed whole before gradients, leaving 401 rows. V6 did not filter that V5 curriculum; it deliberately rebuilt from V4 and does not use the supplemental 57+7 annotation file set. Twelve original controls remain. This is a curriculum choice, not a V6 training-budget exclusion.

Each run initialized a fresh rank-8 adapter over the same pinned four-bit Qwen3-1.7B base, adapting q/v projections in the last eight layers. Full tokenization/mask preflight reports zero truncation; reserved rows had no loss or generation during optimization. V6 changes data balance, duration and learning rate together, so improvements cannot be attributed to one factor.

Inference uses temperature zero, thinking disabled, up to two calls per case, 768 output tokens per stage, 45 seconds per stage, 8,000 source characters and no repairs. Serving receipts verify 704 base tensors plus 32 adapter tensors / 16 wrappers. The validation gate requires raw correct ≥ 28, exact business records ≥ 9, accepted correct ≥ 23 and wrong accepted ≤ 3.

Source provenance exception: V6 includes commit `4cd4689d5d17542cb0f6e870ead8d1e38d3fb599`, which only adds `SCRAPER_MAX_SOURCE_CHARS` and `SCRAPER_MAX_OUTPUT_TOKENS` environment mappings in `ai_scraper/config.py`. V4/V5 use config SHA `4c313bd487d52a6c3757b67f96ba5d09745c355c7e63bdb11d78d9948cc5e411`; V6 uses `97722ebeea01f8316e19b311d87618cc7695e221ec13e3bfb2ea7beefc0101b7`. Every other application file and the evaluation source builders have identical hashes. All actual arm freezes independently retain 8,000 source characters and 768 output tokens. The report permits only these exact config hashes and this sole changed application path; it does not claim all production bytes are unchanged.

The actual pinned Firecrawl cleaning/Markdown component prepared 48/48 saved HTML inputs. This does not test live URL retrieval, the full API or Firecrawl native JSON extraction. Studio VPN/blocked egress and default services stayed unchanged.

## Pinned artifacts

- V6 dataset: `83a0444ac32cf955925894f6e023086feda57c9a4b92838f34ce676894aee992`.
- V6 source commit: `2912ccc3aad74fd2ff2eba9bae2fcad25e49a3e7`.
- V6 Python tree: `b0c269479ae0b4f3789f6bd860e357d4349cef6a46277c677f2bbcc4738e05b6`.
- V6 adapter: `99a5c04ed5714b938e7232718870fe80248f839227dfe59902de0abd8085637d`.
- V6 comparison plan: `f57ab6e0cb25fe4ad9d64d25103570b6b7224be5f848d44f476cb1f65a270c6a`.

The JSON summary retains whitelisted data/source/weight/freeze/receipt identities and stage measurements. Raw source passages, task prompts, completions, private paths and credentials are excluded. There is no matched benchmark establishing superiority over Firecrawl or hosted extraction systems.

## Packaged Studio CLI and MCP checks

The portable adapter was also tested through a separate installed 0.8 CLI and
MCP research profile. A caller-authored observatory example passed with two
local model calls, one saved snapshot, two evidence bindings, matching stdout/JSON
export and six discovered MCP tools. Its `partial` status records source-view
omissions. Loaded-weight receipts confirmed 704 unchanged base tensors, 32 exact
adapter tensors and 16 invoked wrappers. The owned loopback server stopped after
the check; the existing default service stayed unchanged.

A second functional check used the first IANA TRAIN case in frozen file order,
selected before site-test outcomes. It failed after the bounded repair: two
attempts / four local calls, with `GROUNDING_REJECTED` reporting missing critical
business fields. Independent CPU reconstruction confirmed identical requests,
saved source, source windows and initial business prompts, with all three reviewed
facts visible. This is a model failure on a TRAIN page, not a new generalization
score or an intentionally omitted training target. It is retained as a practical
limitation; no labels, test scores or weights were changed in response.

These packaged checks are separate from the frozen no-repair comparison above.
They reinforce the experimental, non-default status of this adapter.
