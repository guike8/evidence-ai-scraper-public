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
