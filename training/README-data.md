# Synthetic pilot data

This optional offline generator authors a small schema and evidence curriculum.
It is **synthetic, automatically validated data**, not a human-reviewed sample
of real websites. Its scores cannot establish real-web accuracy, universal
coverage, or an advantage over another scraper.

Run from this checkout after installing the scraper:

```sh
python -m training.dataset --output-dir /absolute/private/path/scraper-pilot
```

The output path is required and must be outside the public source checkout.
The defaults produce 720 training, 72 validation, and 108 held-out test records,
covering all eight registered presets and a custom contract. Adjust counts with
`--train-per-schema`, `--valid-per-schema`, and `--test-per-schema`; pin `--seed`
to reproduce bytes and fingerprints. A nonempty directory can be reused only
when its generator version and configuration match.

For a new run using grammar-constrained serving, add `--target-order schema`
and choose a new output directory. This serializes assistant JSON in the
Pydantic model's declared property order, including nested business objects and
each evidence object's `field_path`, `section_id`, `excerpt`, `json_pointer`.
That order matches the compiled provider schema used by serving. Source prompts,
gold values, evidence, sidecar bytes, split groups, and completeness labels stay
identical. The schema-order manifest records the serialization mode and obtains
new message-file checksums and a new dataset fingerprint. A directory from one
mode cannot be reused in the other mode.

The default `--target-order canonical` preserves the original pilot's sorted-key
targets and complete file bytes for reproducibility. Its frozen 900-case
fingerprint is `a714afb1a8274e97e2ddac65737a1cfe0ee76b0ce3693b5ea26d802523881d91`.
Legacy manifests omit `target_order`; absence means canonical. This serialization
change requires a separate training run and comparison; it does not itself
establish a model-quality improvement.

Each split has an MLX-LM-compatible `*.jsonl` file with three chat messages:
system, user, and assistant. The system instruction and untrusted source JSON
use the production adapter's exact rendering. The assistant is plain JSON with
only `data`, `evidence`, and `warnings`; application status and provenance stay
outside the model target. Custom requests explicitly describe field names and
types in their trusted instruction, as a serving caller can do. Production
serving additionally supplies the schema as a grammar, which these chat records
do not simulate.

`*.cases.jsonl` sidecars hold the original request, synthetic saved snapshot,
authored expected candidate, expected completeness status, flattened expected
field values, JSON schema, source-view metadata, and grouping identifiers. All
populated business leaves must pass the actual Pydantic and source-evidence
validators. Deliberate null critical fields are labeled as expected failed
completeness assessments; other validation errors abort generation.

Facts are authored before HTML rendering or validation, without using model
answers or deterministic extraction output as gold. Sources contain strict
numbers, integers, booleans, zero, false, strings, arrays, tables, relative URL
attributes, missing fields, competing entities, and page-level prompt injection.
Some examples cite saved section IDs and others cite JSON-LD array-root pointers.
All entities and `.test` domains are invented. No private snapshots, external
training datasets, existing public benchmark rows, capability probes, or remote
pages are read by the generator.

The train, validation, and test splits use different actual DOM skeletons,
lexical families, domains, and entity identifiers. Generation verifies disjoint
template IDs/fingerprints, domains, entities, raw source hashes, and serialized
source-view hashes. Shared schema vocabulary and simple fact patterns remain;
this is a basic grouped leakage check, not a guarantee of semantic novelty.

`manifest.json` records provenance, counts by schema/mode/status, checksums,
quality checks, and a dataset fingerprint. The files are written with private
permissions. Preserve this manifest alongside any trained adapter so its data
origin can be audited. Keep all generated datasets and checkpoints outside the
public checkout; publish only sanitized model artifacts and their provenance.

Before training, render chat with the model's own pinned tokenizer and
`enable_thinking=False` for both the full and prompt-only sequences. Assert that
the prompt token sequence is an exact prefix of the full token sequence and
measure every record against the configured sequence limit. Reject overlong
records instead of silently truncating a completion. The generator records
character counts only; those are not token counts. Preserve the test split for
one frozen before/after comparison and do not use it for training or checkpoint
selection. A meaningful release still needs an independently reviewed real-web
evaluation and checks for regressions on unmodified public benchmark fixtures.
