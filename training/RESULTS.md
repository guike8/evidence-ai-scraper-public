# Local model experiment results

These November 2026 experiments use authored synthetic snapshots. They do not
establish representative web accuracy or a lead over other scrapers. Retrieval
and deterministic extraction are excluded: each model receives the same saved
source and extraction contract, with one completion and no repair.

## Fixed baseline and data

The base is `mlx-community/Qwen3-1.7B-4bit` at
`3b1b1768f8f8cf8351c712464f906e86c2b8269e`. MLX-LM training code is pinned at
`736ab0d8c1d0edc79c3d798044fe6904e2fa23be`. Training uses rank 8, the final eight
layers' q/v attention projections, scale 20, dropout zero, batch one,
accumulation four and learning rate 1e-5. The adapter has 458,752 parameters
across 32 float32 tensors, occupying 1,838,449 bytes.

The corpus contains 720 training, 72 validation and 108 sealed test cases over
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

The final model selection and held-out results must be frozen and reported
before a production decision. No API inference fees were incurred; hardware,
electricity and download costs were not measured. Real-page retrieval,
representative human-reviewed data, multilingual behavior, unfamiliar custom
schemas and fused artifacts still require separate evaluation.

Source code is MIT. Model derivatives retain Apache-2.0 attribution. Public
model packages contain only verified adapter tensors, a minimal MLX loader
configuration, licenses, sanitized measurements and checksums; raw run state,
datasets, credentials and base weights are excluded.
