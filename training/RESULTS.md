# Local model experiment results

## Citation-v3 continuation: improvement with a real-page failure

The updated adapter improves our previous weights on two synthetic sets, but
**fails every eligible real-page task** in this small replay test. It is an
experimental checkpoint, not a production recommendation or a Firecrawl win.
The next priority is a matched Firecrawl baseline and reviewed real-page data,
as described in the [integration and comparison plan](../docs/firecrawl-integration.md).

| Track | Old accepted | New accepted | Correct accepted facts, old / new | Wrong accepted facts, old / new |
|---|---:|---:|---:|---:|
| Original synthetic regression | 41/108 | 61/108 | 231 / 407 | 0 / 0 |
| Fresh synthetic families | 16/108 | 29/108 | 87 / 169 | 2 / 0 |
| Eligible real-page replays | 0/13 | 0/13 | 0 / 0 | 0 / 0 |

The 108 original cases were inspected in previous work and are explicitly a
regression set. The 61/108 gain is +20 records (+18.52 percentage points).
The paired run retained 39 old accepted records,
gained 22 and lost 2.
Accepted required recall rose from 82/252
(32.54%) to
154/252
(61.11%).
Records with evidence rejection fell from 56 to 36; validation codes can overlap.
All 17 intentional critical-missing controls were rejected in both arms.
The new raw candidates still contain 21 wrong or unlabeled values; their
validated accepted business objects contain 407 correct and zero wrong facts.

The newly authored synthetic test uses disjoint DOM/entity/domain/custom-name
families, with related extraction tasks. New accepted required recall is
56/207
(27.05%); raw all-field precision/recall
is 87.50%/93.33%.
It retains its timeout and truncated completion in the 108-case denominator.
These automatically checked, agent-authored labels have no independent human
review. Neither synthetic score establishes general web accuracy.

### Real-page replay scope and failure

The frozen corpus attempted 18 tasks on 17 distinct public URLs across 12 hosts.
Five tasks could not reach model evaluation: three authored targets fail the
current production contract, and two exceed the fixed source-planning budget.
The eligible 13 tasks contain 31 positive facts, 25 required facts and one
intentional no-product negative. They are English developer-oriented tasks,
not a representative sample; no positive product or job task is eligible.
Labels and source hashes were reviewed before inference, with a second agent
review but no independent human annotation. Raw sources are not redistributed.

New acceptance is 0/13 eligible and
0/18 attempted. Its raw correct fact count is
0/31, with
9 output-budget failures.
The remaining failures include schema, completeness and evidence rejection.
This is broader failure than citation formatting: synthetic training has not
taught adequate instruction following and extraction on these natural pages.
The old model's paired eligible acceptance is 0/13;
accepted required-fact recall across all 18 attempts is
0/48 old and
0/48 new.
Zero wrong accepted facts when nothing is accepted is not an accuracy success.

Public pages were captured on a connected Mac and replayed through inference
on Studio. The protected Studio VPN egress stayed blocked. This is neither
live Studio retrieval nor a comparison against Firecrawl Cloud. Retrieval,
deterministic extraction and bounded repair are outside these model-only scores.

### Selection, training and reproducibility

All six old/new runs use the same original exclusive-reference grammar,
unchanged business/evidence/completeness validators, greedy decoding, thinking
disabled, 4,096 output tokens, 48,000 source characters, 45 seconds per request,
one completion and no repair. Weights, source, case hashes and policy were
frozen before testing. There was no post-test tuning or concurrent training.
Read [portable paired metrics](citation-v3-metrics.json) for denominators,
raw/accepted/required/optional facts, per-schema scores, errors, latency,
tokens, negative controls, source hashes and paired record changes.

Training continued from the old adapter with a fresh Adam optimizer and no
optimizer/RNG resume: 1,800 microbatches, 450 updates, learning rate 5e-6,
rank 8 on the last eight q/v layers, scale 20, dropout zero, batch one,
accumulation four and a 4,096-token sequence limit. The blend contains 720
original **training-only** replay rows and 1,080 balanced citation-v3 rows;
validation is 72 new rows and the fresh test 108 rows. Plaintext, JSON-LD and
mixed representations cross ordinary/negative modes independently. V2 had
confounded representation with task mode and was not promoted.

The [fixed public replay recipe](README-citation-v3.md#reproduce-the-measured-training-replay)
reproduces all six measured JSONL files byte for byte, fingerprint
`410a7f708a512ca305c1cbd92db55fd1a3938d06a7804484c3932e0b821ee47e`.
Training including preflight took 1,026.8 seconds with 3.44 GB peak MLX
allocation. All 1,800 examples fit without truncation. The saved adapter has
32 finite F32 tensors / 458,752 parameters; original base tensors remain
unchanged. New adapter SHA256:
`64e66dab7b04a7c8a3e90af136477856ba57de52023b45ab4dabb985deb49f7b`.

The experimental quote/locator/field-path grammar is available separately but
was **not selected** for these results. New weights plus that grammar accepted
44/72 original validation records with no wrong accepted facts, then admitted
two semantically wrong facts on a second validation set. Literal quote
occurrence cannot establish full identity or ownership by an array item.
The selected original grammar had zero wrong accepted facts on both validation
sets. No production validator was relaxed or target relabeled to raise a score.

The [experimental citation-v3 MLX adapter](https://huggingface.co/guike8/evidence-scraper-qwen3-1.7b-citation-v3-mlx-lora)
has a separate model card and checksums; the earlier checkpoint remains intact.
The [foreground serving helper](README-serving.md) verifies exact backend
package identities/import origins, base/adapter tensors and active wrappers.
The public helper and minimal published adapter configuration passed actual
Studio startup and tensor checks. All nine probe completions were schema-valid
and reference-exclusive, but only the listing probe passed the full exact-fact
and evidence contract (1/9). Local fixtures passed 1,124 tests (38 optional
skips); Studio fixtures including browser execution passed 1,130 tests (32 skips).
Ruff checks, distribution audits and the serving security review passed. No API inference fees were incurred; hardware and energy
costs are unmeasured. The historical experiments below remain for comparison.

## Fixed baseline and data

The base is `mlx-community/Qwen3-1.7B-4bit` at
`3b1b1768f8f8cf8351c712464f906e86c2b8269e`. MLX-LM training code is pinned at
`736ab0d8c1d0edc79c3d798044fe6904e2fa23be`. Training uses rank 8, the final eight
layers' q/v attention projections, scale 20, dropout zero, batch one,
accumulation four and learning rate 1e-5. The adapter has 458,752 parameters
across 32 float32 tensors, occupying 1,838,449 bytes.

The corpus contains 720 training, 72 validation and 108 held-out test cases over
eight presets and a custom schema. Template families, domains, entities and
source hashes are disjoint. Facts are independently authored before rendering
HTML; no private pages, public benchmark cases or model-generated labels are
used. Repeated simple task families remain, so this is a pilot rather than a
generalization guarantee.

Canonical data fingerprint:
`a714afb1a8274e97e2ddac65737a1cfe0ee76b0ce3693b5ea26d802523881d91`.
Schema-ordered data fingerprint:
`2f3639f41a4bf6a5d41eaf58a959067abeb61f61abf1c9cad5d222ae80be2499`.
All 900 source messages, gold values and case-file bytes match between modes;
only assistant JSON key serialization differs. Actual serving forces schema
property order, which motivated the second mode.

## Original-grammar validation

All rows below use the same 72 cases, greedy decoding, thinking disabled,
4,096 output tokens, the original local JSON grammar and identical business/
source validators. Failed and truncated outcomes remain in denominators.
“Accepted” means the application returned success or partial; it does not
independently certify arbitrary field meaning.

| Model | Training microbatches | Required precision | Required recall | Schema-valid cases | Accepted cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0 | 80.54% | 95.51% | 68/72 | 0/72 |
| Canonical-target QLoRA | 200 | 86.78% | 96.79% | 65/72 | 0/72 |
| Schema-ordered QLoRA | 720 | 96.89% | 100.00% | 72/72 | 25/72 |

The final row has 156 correct required facts out of 161 predictions and 156
present gold facts. Across all fields it has 423 correct values out of 438
predictions and 424 present gold values. Source-supported leaves rose from
zero in the base run to 317; 312 of those also match the independent labels.
Remaining citation and unsupported-value failures prevent production claims.

Both target serialization and training duration changed between the two
adapters. Their difference cannot be attributed to serialization alone. The
200-microbatch run took 97.0 seconds including preflight/sanity; the 720 run took
268.7 seconds. Peak MLX allocation was approximately 3.38 GB. The latter run's
teacher-forced validation loss fell from 0.714 to 0.0229; the upstream last loss
is measured before its final update. Accuracy above is from reloaded weights,
not inferred from loss.

The serving experiment used a private process-local oMLX loader hook to load
the unfused adapter. It checked 704 original base tensors unchanged, all 32
adapter tensors exact, and all 16 wrappers active after server transforms.
It did not modify installed server files. This hook is experimental integration,
not an official oMLX adapter configuration feature. No fusion or requantization
was used in these comparisons.

The valid split includes intended critical-missing negative controls. Even
perfect authored gold has accepted required recall of 136/156 (87.18%) when
whole rejected records remain in the denominator. Keep raw fact accuracy,
accepted accuracy and negative-control handling separate; do not claim a 95%
accepted-recall gate passed on this full corpus.

## Evidence schema control

The original provider schema permitted two non-null reference fields although
application validation requires exactly one. The base populated both in all
488 evidence items. A deterministic path-only control rewrote 321 unambiguous
`data.` prefixes without changing facts or quotes; it still accepted zero
records because other reference/binding failures remained.

A fixed-object `anyOf` grammar now expresses exclusive text versus JSON-LD
references in the local adapter. With the installed XGrammar 0.2.8, all nine
schema families compiled, accepted gold and rejected both-reference and
neither-reference examples. Actual server requests also enforced exclusivity.
Guarded model measurements are recorded separately so grammar changes are not
attributed to training.

On the same 72 validation cases with this exclusive-reference grammar, the base
accepted 0 records and the schema-ordered adapter accepted 27 (26 success and
one partial), with all 72 adapter completions schema-valid. Required raw
precision/recall were 82.35%/98.72% for the base and 96.89%/100% for the adapter.
The adapter verified source support for 327 of 438 populated fields, of which
322 also matched independent gold values. This remains an experimental result.
All nine schema envelopes were supported, but none of the nine separate exact
fact-and-evidence capability probes passed the full contract.

## Frozen held-out comparison

The selected adapter, grammar, 4,096-token output budget and inference policy
were frozen before either 108-case test run. The snapshot/case hash, original
base tensors, both adapter files and grammar fingerprints were verified for
both arms. No weights or policy were tuned after opening the test split.

| Metric | Base | Schema-ordered QLoRA720 |
| --- | ---: | ---: |
| Required raw precision | 81.58% (248/304) | 95.44% (251/263) |
| Required raw recall | 98.41% (248/252) | 99.60% (251/252) |
| All raw precision / recall | 82.86% / 95.12% | 96.07% / 97.63% |
| Schema-valid completions | 104/108 | 107/108 |
| Accepted records | 0/108 | 41/108 |
| Accepted required precision / recall | unavailable / 0% | 100% / 32.54% |
| Source-supported leaves | 7/776 | 500/687 |
| Truncated completions | 4 | 1 |
| Request latency p50 / p95 | 1.88 s / 6.72 s | 1.42 s / 2.50 s |

All 231 populated facts in the adapter's accepted records matched the authored
gold, including all 82 accepted required facts. Its raw candidates still
contained 27 wrong/unlabeled values, and citation failures rejected many
otherwise accurate records. There were no repetition flags. Failures and the
one truncated adapter outcome remain in the 108-case denominator.

These are warm-loaded-model request timings after capability probes, with a
fresh source task per case. Generated length and failure patterns differ;
this is not a universal throughput claim. Both isolated servers stopped after
evaluation. The separate exact fact-and-evidence capability probes remained
0/9, so the result does not qualify the model for general production use.

Adapter SHA256:
`a39abab04fec46859f2931f741b377a714cda6d7c2086ac2b29faa3a1d49ca09`.
The evaluated training workflow matches source commit
`e9d1458` byte for byte; subsequent process-supervision fixes do not change the
frozen weights. The base and adapter test-receipt SHA256s are respectively
`b2c034c793372d49363d2265462bb18cb722539f1df15b5ea6ef92349991908d` and
`9a30bdb36e3c1fc1ae207a4b5454add1cc3f223f0d246067203284dfa8c0910e`.

## Weight-space experiment

Inspired by [weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/),
a separate non-gradient experiment targeted the incorrect `data.` citation
prefix. Calibration and controls used 12 training cases, never held-out test
examples. Actual weight/activation inspection selected layer 23's down
projection, column 3657; its projected activation was selective for the
citation context relative to wrapper and business-string controls.

The fixed primary dose changed 2,048 effective values in that one projection,
with a relative Frobenius change of 0.00132859 (0.133% of that matrix's norm).
The original packed quantized tensors remained byte-identical; frozen additive
wrappers represented the edit. Predetermined half-dose and norm-matched
orthogonal controls were also measured.

The actual-model zero-dose control preserved selected logits bit for bit and
preserved dtypes. Primary citation margins shifted slightly, but citation
argmax was unchanged on all 12 cases. Base and edited native outputs were
byte-identical on both fixed generation cases; neither passed the extraction
schema. No extraction improvement was demonstrated, and the patch was not
promoted. This bounded result does not establish that other weight-space
methods cannot help.

The budget was 46 teacher-forced sequences plus four native generations,
including 694 actual full-transformer calls. Diagnostic/probe input totaled
62,379 teacher-forced tokens and 4,828 generation-prefill tokens; generation
produced 640 tokens. Full unseen grammar-assisted WSP accuracy was not measured.

## Scope and publication

The frozen [experimental MLX adapter](https://huggingface.co/guike8/evidence-scraper-qwen3-1.7b-mlx-lora)
is available separately with its model card and checksums. It is MLX-only;
Transformers/PEFT and GGUF compatibility are not established.

The held-out result remains below the full accepted-recall release target.
No API inference fees were incurred; hardware,
electricity and download costs were not measured. The citation-v3 real-page replay above adds limited coverage; representative
human-reviewed data, multilingual behavior, unfamiliar custom schemas and fused
artifacts still require separate evaluation.

Source code is MIT. Model derivatives retain Apache-2.0 attribution. Public
model packages contain only verified adapter tensors, a minimal MLX loader
configuration, licenses, sanitized measurements and checksums; raw run state,
datasets, credentials and base weights are excluded.
