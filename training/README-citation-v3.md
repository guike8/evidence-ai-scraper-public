# Citation curriculum v3

This opt-in, authored synthetic curriculum separates source representation from
task mode. It fixes a v2 design confound: v2 ordinary positive examples omitted
JSON-LD while its injection and ambiguity examples retained it. This is a data
design correction, not a demonstrated model improvement.

```sh
python -m training.citation_dataset_v3 --output-dir ../private-data/citation-v3
```

Defaults produce 1,800 training, 180 validation and 108 newly authored test rows.
The generator reads no existing dataset, benchmark, model completion, private
page or remote source. Generated files stay outside the checkout with private
permissions. Existing pilot and v2 datasets are unchanged.

Each schema uses a 24-case cycle: all five semantic modes cross all three source
representations, followed by nine additional complete positives. Complete
examples constitute half of a full cycle. The five modes are complete,
missing noncritical, missing critical, injection and ambiguity. Plaintext
examples contain no JSON-LD; JSON-LD examples place business values in structured
data; mixed examples contain both. JSON-LD-only products expose their offer's
raw numeric price string as `price_text`, keep the explicit currency separate,
and do not invent a visible short description. Ambiguous page-title examples
retain their conflicting HTML metadata.

DOM family, nuisance section offsets and mixed-source citation preference are
sampled independently of this cycle. Most train rows retain production section
order; one quarter shuffle presentation while preserving real IDs and text.
Validation and test presentation exactly match the production inference view.
Train, validation and test use distinct DOM families, entity/domain identities,
source hashes and custom field-name vocabularies. Custom schemas vary typed
field-name tuples instead of repeating one fixed tuple per split.

Facts are authored before rendering. Every candidate then passes Pydantic and
the unchanged source/evidence/business validators. Critical-null negative
controls may fail completeness only. Evidence is bound to the exact included
inference view, not excluded duplicate sections. The manifest records hashes,
joint representation/mode counts, citation-form counts, custom schema variety
and split-disjointness checks. Actual tokenizer preflight remains required
before training; the generator does not claim a token-length guarantee.

For a training-only blend, 120 v3 rows per schema span five full cycles and yield
1,080 rows. A caller may combine those with the original 720 training rows to
retain prior task coverage. Keep validation and test rows out of that blend,
freeze candidate selection on validation, and reserve held-out outcomes for the
final comparison. Reused business fact templates and synthetic task families
still limit generalization claims; this is not human-reviewed web accuracy.

## Reproduce the measured training replay

The fixed replay recipe used for the citation-v3 continuation combines only
720 original **training** rows with 1,080 balanced v3 training rows. It does not
consume original validation/test rows as training examples. Generate both
schema-ordered inputs outside the checkout, then build a new private output:

```sh
python -m training.dataset --output-dir ../private-data/original-schema \
  --seed 20261105 --target-order schema \
  --train-per-schema 80 --valid-per-schema 8 --test-per-schema 12
python -m training.citation_dataset_v3 --output-dir ../private-data/citation-v3-pool \
  --seed 2026110701 --train-per-schema 120 --valid-per-schema 20 --test-per-schema 12
python -m training.citation_replay \
  --original-dir ../private-data/original-schema \
  --v3-dir ../private-data/citation-v3-pool \
  --output-dir ../private-data/citation-v3-replay
```

The mixer fixes the shuffle seed at `2026110704`. It selects validation indices
`0, 2, 6, 8, 10, 14, 16, 19` for each v3 schema and copies the 108 fresh test
rows unchanged. Final counts are **1,800 train / 72 validation / 108 fresh test**.
The six JSONL files reproduce dataset fingerprint
`410a7f708a512ca305c1cbd92db55fd1a3938d06a7804484c3932e0b821ee47e`.

This is an intentionally fixed recipe. It requires the measured source dataset
identities, verifies file hashes/counts, validates every source case and target
against the unchanged production contracts, checks split-disjoint groups, and
refuses overlapping or existing output directories. The manifest records
portable identities rather than local paths. Its source-code hash differs from
the original private mixer, while all six dataset files remain byte-identical.
Generating or mixing these files runs no inference, scoring or training.
