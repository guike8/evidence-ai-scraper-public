# Real-page v6: business extraction and citation selection

This experimental MLX adapter targets the scraper's two-stage local extraction
policy. It interprets saved page content. Retrieval, crawling, browser rendering,
source selection, validation, recovery and provenance remain application code.
It does not change the default model. Read the [paired results and failures](RESULTS-realpage.md)
and [machine-readable measurements](realpage-v6-metrics.json) before selecting
the separately distributed model weights.

V6 failed its preregistered validation gate and is not recommended as the default
extractor. The weights are an experimental research artifact: raw validation
accuracy declined and wrong facts passing validation increased relative to base.

The intended model release is
[`guike8/evidence-scraper-qwen3-1.7b-realpage-v6-mlx-lora`](https://huggingface.co/guike8/evidence-scraper-qwen3-1.7b-realpage-v6-mlx-lora).
It is an unfused MLX LoRA over `mlx-community/Qwen3-1.7B-4bit`, pinned at
`3b1b1768f8f8cf8351c712464f906e86c2b8269e`. It requires that base; it is not
a standalone model, GGUF or generic PEFT adapter. The model retains Apache-2.0
terms and upstream attribution; the scraper source is MIT. The conversion's
historical upstream revision is not documented by its publisher.

## How grounding works

The `local-two-stage-grounded-v1` policy supplies trusted field definitions,
the user's instruction and a declared source view. The first completion returns
bare business JSON. Application code enumerates eligible literal occurrences of
each predicted value in that exact source. A second completion selects a
candidate ID or null for every populated leaf. Application code maps those IDs
to original sections and exact excerpts, then runs the existing schema, evidence
and completeness validators. The model never assigns provenance or status.

An eligible occurrence can belong to the wrong entity, field, unit or navigation
item. Valid IDs and verified quotations therefore do not guarantee semantic
correctness. Business accuracy, wrong accepted facts, citation mechanics and
reviewed-anchor agreement are separate measurements. A null selection is an
abstention, not automatic success.

Both stages share `grounded-keyword-window-v1`: whole original sections ranked
using trusted request terms and restored to page order, with an 8,000-character
serialized limit. It prefers actual saved Markdown, retains original evidence
IDs and declares omitted sections/representations. It never clips a paragraph
or uses gold, predicted values or page instructions to choose the query. The
saved full snapshot remains intact. Operational results declare incomplete
source coverage rather than claim full-page support.

## Run the measured profile

Install and verify the separate [experimental serving integration](README-serving.md).
Download the pinned base and released adapter separately, verify the release's
checksums, then use the verified serving interpreter:

```sh
python -m training.serve_omlx \
  --experimental-omlx-adapter \
  --base /private/scraper-serving/base \
  --adapter-dir /private/scraper-serving/realpage-v6 \
  --adapter-sha256 VERIFIED_ADAPTER_SHA256 \
  --state-dir /private/scraper-serving/realpage-v6-run \
  --model-name evidence-scraper-realpage-v6 --port 8001
```

Replace paths and the checksum with your verified artifacts. The helper checks
base, adapter and backend identities. An arbitrary OpenAI-compatible endpoint
is not sufficient compatibility evidence; run the actual schema probe against
the selected artifact/runtime. No model is downloaded or service switched
automatically.

In a second terminal, from the source checkout:

```sh
export SCRAPER_INFERENCE_BACKEND=local
export LOCAL_BASE_URL=http://127.0.0.1:8001/v1
export LOCAL_MODEL=evidence-scraper-realpage-v6
export LOCAL_TWO_STAGE_GROUNDING=true
export SCRAPER_MAX_SOURCE_CHARS=8000
export SCRAPER_MAX_OUTPUT_TOKENS=768
export HF_TIMEOUT_SECONDS=45
uv run ai-scraper replay examples/page-record.json \
  --url https://example.test/field-notes --local-inference \
  --fields 'document_title,purpose' \
  --ask 'Extract the exact document title and the stated purpose.' --output result.json
```

This authored replay uses local inference without outbound retrieval. For a
live page, set `SCRAPER_RETRIEVAL_BACKEND=direct` and exact
`SCRAPER_ALLOWED_DOMAINS`, then run `extract URL` with your fields/instruction.
Direct retrieval needs permitted outbound internet. The CLI reads exported
variables and does not load `.env` automatically. The same settings work with
the library and stdio MCP server. Earlier citation modes must remain false:
they cannot be combined with two-stage grounding.

The operational pipeline allows one repair: at most two composite attempts and
four model calls on the same saved source. The recorded evaluator allows two
calls per case, greedy decoding, thinking disabled, 768 output tokens per stage,
a 45-second stage timeout and no repair. This CLI example is not the frozen
evaluation protocol.

## Corpus and review

The private corpus contains 48 saved public pages across 18 site families,
96 original flat custom-field requests, 220 positive scalar leaves and 22
optional absence controls:

| Split | Pages | Original requests | Positive facts | Absence controls |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 32 | 64 | 146 | 14 |
| Validation | 8 | 16 | 36 | 4 |
| Reserved site test | 8 | 16 | 38 | 4 |

Families and saved-source identities are disjoint. Twelve previous development
pages are TRAIN-only. Site holdout means unseen in this fine-tune, not necessarily
unseen in the base model's pretraining. Agents authored gold and another agent
independently checked full saved sources and exact evidence contexts before v4
inference. This is not external human annotation.

Initial URL capture used direct HTTP on a connected machine. The pinned official
Firecrawl cleaning/Markdown functions prepared all 48 saved HTML inputs in a
network-disabled container. Full Firecrawl API retrieval and native extraction
were not evaluated. The Studio's VPN protection and blocked public egress were
preserved. Raw captures, labels and completions are private, not redistributed
or relicensed with this source.

The automatic window retains reviewed anchors for 133/146 original TRAIN facts,
33/36 validation facts and 34/38 test facts. Supervised fields absent from the
shown view become null/empty. Evaluation retains every full-page gold fact in
the denominator, including source-window omissions and rejected/failed cases.

## Why a paired curriculum

Validation is selection data. The completed earlier trials are retained:

| Validation model | Raw correct /36 facts | Exact business records /16 | Accepted correct facts | Wrong accepted facts | Accepted cases /16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 28 | 9 | 23 | 3 | 11 |
| V4 | 27 | 9 | 26 | 6 | 14 |
| V5 | 28 | 10 | 5 | 2 | 3 |
| V6 | 26 | 8 | 24 | 6 | 13 |

V4 increased acceptance while admitting more wrong facts. V5 kept raw fact
accuracy but rejected 13 cases, including eight entirely correct business
records. Audit found unnecessary null selections on visible correct values.
The high proportion of counterfactual grounding rows was a motivation for the
v6 balance change; it is a hypothesis, not an isolated causal result. No v4 or
v5 site-test inference was performed.

V6 accepted more cases and abstained less, but raw correct facts fell from 28
to 26 and wrong accepted facts rose from three to six. It failed the gate
requiring at least 28 raw correct facts, nine exact business records, 23 accepted
correct facts and at most three wrong accepted facts. It was not promoted.
Literal ID validity is a separate outcome: 33 valid IDs, zero invalid/missing
IDs and one abstention still allowed semantic mistakes through validation.

V6 contains 560 conversations: 274 business rows and 286 grounding rows. Business
rows comprise 64 originals, 146 single-field projections and 64 field-name aliases
over unchanged reviewed facts. Every business task has one corresponding
grounding task with the exact source, schema and visible target: 274 ordinary
grounding rows, of which 32 have empty selections. Another 12 rows retain the
original reviewed wrong-context controls. The later 57+7 negative curriculum
used by v5 is not added. These are derived examples, not new independent pages
or facts.

Positive grounding targets must match reviewed original section IDs and exact
excerpts; they are not assigned to the first occurrence of a value. Projections
and aliases remap those anchors and preserve original list indices when omitted
elements are reindexed. Both stages reconstruct the exact production prompts
and schemas. No arbitrary schema, provenance or status target is accepted.
All five reserved row/case files remain byte-identical. Original business gold
is unchanged. Validation informed design and selection; no validation/test
labels or predictions became training targets. The site test was opened only
for the frozen final candidate, with no post-test tuning. Its results show a
precision/coverage tradeoff rather than an overall win; validation failure
remains part of the release decision.

## Compile and train

The [initial compiler guide](README-realpage-v4.md#compile-and-train) describes
reviewed captures and exact-hash receipts. Starting with the original v4 dataset
containing its 12 reviewed controls, the public paired augmenter is explicit:

```sh
python -m training.realpage_augment \
  --base-dir /private/realpage-v4-data \
  --pair-grounding --output-dir /private/realpage-v6-data
```

Omitting `--pair-grounding` preserves the earlier business-only augmentation.
Do not supply supplemental negative files to reproduce this v6 recipe. The
manifest binds parent files, transformations, source workflow and review
identities. Reserved rows are copied and hashed without parsing their content.
Reproducing the measured fingerprint requires the private reviewed corpus and
exact pinned source/environment, not just this checkout.

After the full pinned-tokenizer/masking preflight, use the separate MLX training
environment described in the [training guide](README.md):

```sh
python -m training.job start --python /private/mlx-training/bin/python \
  --model /private/models/qwen3-base --data /private/realpage-v6-data \
  --run-dir /private/realpage-v6-run --target-format realpage_grounded_v1 \
  --iters 1536 --learning-rate 0.00001 --max-seq-length 8192 \
  --memory-gib 8 --max-seconds 5400
```

The completed preregistered run started a fresh rank-eight adapter over the original
four-bit base, without a v3/v4/v5 warm start. It uses the last eight layers' q/v
projections, scale 20, batch one, accumulation four, 1,536 microsteps and 384
optimizer updates. The maximum sequence is 8,192 tokens and MLX cap is 8 GiB.
Overlong rows fail preflight; targets are not truncated. One final checkpoint
was preregistered with no checkpoint search. Test rows received token/mask checks
only, without test loss or model generation during optimization.

The final receipt verifies 1,536 microsteps and 384 optimizer updates. Training
took 529.75 seconds, or 541.67 seconds including supervisor stages, with peak
MLX allocation of 6,062,763,256 bytes. All 560 TRAIN rows passed exact masking
and zero truncation; the longest was 6,112 tokens. The adapter checksum is
`99a5c04ed5714b938e7232718870fe80248f839227dfe59902de0abd8085637d`.

Training/evaluation source commit:
`2912ccc3aad74fd2ff2eba9bae2fcad25e49a3e7`.
Dataset fingerprint:
`83a0444ac32cf955925894f6e023086feda57c9a4b92838f34ce676894aee992`.
Full results retain all validation history, failures, source coverage, literal
citation mechanics, raw/accepted fact accuracy, hashes, tokens and latency.

## Scope

This is an English, technical-documentation-heavy pilot with flat custom schemas,
limited ecommerce/event/job pages and narrow contact-based absence controls.
The reserved site test has only 16 requests. Semantic errors can pass literal
evidence checks. The experiment does not establish universal web accuracy,
multilingual/nested-schema performance, production reliability or superiority
over Firecrawl or hosted extraction models. Live Studio public retrieval remains
unverified while VPN egress is blocked. Existing default services remain separate
from this opt-in profile.
