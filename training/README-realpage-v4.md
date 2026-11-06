# Reviewed real-page, two-stage local extraction

This experimental path separates business extraction from evidence selection.
It is an opt-in local policy, not the default model or evidence format. Use
`LOCAL_TWO_STAGE_GROUNDING=true` with `SCRAPER_INFERENCE_BACKEND=local`, or pass
`--two-stage-grounding` to the CLI. It cannot be combined with the earlier
source-bound or source-quote citation modes.

1. The model receives trusted field definitions, task guidance and a declared
   source window, then returns only business JSON.
2. Application code enumerates eligible occurrences of each predicted value in
   that exact source. The model selects a per-field candidate ID or abstains.
3. Application code maps the chosen IDs to saved locations and exact excerpts.
   The existing schema, evidence and completeness validators assess the result.
   Application code assigns provenance, status and usage.

An occurrence is eligibility for a citation, not proof of semantic meaning.
The second model must distinguish the requested field, entity, units and
context. Custom-field factual accuracy still needs reviewed examples. No first
literal match is automatically turned into evidence. Unknown, cross-field and
out-of-view IDs are rejected. Candidate count/byte limits fail closed.

Each operational attempt makes at most two local calls. The pipeline allows
one repair on the same saved snapshot: at most two composite attempts and four
calls, reserved persistently before each attempt. `usage.inference_attempts`
counts composite attempts; `usage.model_calls` counts observed submitted calls.
No hosted inference or scraping credentials are needed for local replay.

## Source selection and coverage

The shared `grounded-keyword-window-v1` policy selects whole original source
sections using only the trusted request's terms. It prefers saved Markdown,
retains original IDs and ordering, and records omitted representations and
sections. It never uses gold answers, predicted values or source instructions
to choose the relevance query, and never clips a selected paragraph. This
pilot uses an 8,000-character serialized source limit.

A selected window is not proof of full-page coverage. The saved snapshot stays
complete, the descriptor records omissions, and the operational pipeline marks
an incomplete-source result `partial`. Source selection can miss facts; those
remain in evaluation recall. Copying or locating text does not prove that it
has the requested meaning.

## Private dataset and independent review

The pilot retains full captures from 48 public pages across 18 site families:
32 training pages, eight validation pages and eight site-holdout pages. The
training split includes 12 explicitly retired development pages; the other 36
captures are new for this corpus. Site families and source hashes do not cross
splits. The site holdout is unseen relative to this fine-tune, not a claim
about the base model's pretraining data.

Agent-authored gold was independently checked by another agent against saved
HTML, structured data and actual Markdown. This is not human-reviewed gold.
The 96 distinct custom-field requests contain 220 positive scalar leaves and
22 optional absence controls (14/4/4 across splits). Corrections to answer
hints, literal source forms and citation contexts preceded model calls.

Actual content preparation used the pinned official Firecrawl HTML-cleaning
and Markdown functions in a network-disabled container. All 48 conversions
succeeded. URL capture used the independent direct HTTP retriever on a
connected machine; it was not live Firecrawl API retrieval. The Studio's VPN
and blocked public egress were preserved. Raw captures, labels and completions
are private and are not relicensed or redistributed with this MIT source.

There are 140 training conversations: 64 business tasks, 64 corresponding
citation-selection tasks and 12 reviewed counterfactual grounding controls.
The controls give one field a literal value belonging to the wrong context or
entity; its selection target is null despite eligible occurrences. Validation
and test each contain 16 business requests and 32 token-preflight conversations.
Only training rows receive gradient updates. Test tokenization checks do not
compute test loss or accuracy.

The automatic window retains reviewed anchors for 133/146 training facts,
33/36 validation facts and 34/38 test facts. Supervised fields absent from the
shown view become null/empty with logged losses. Evaluation keeps the original
full-page gold and denominator for every split. This distinction prevents
teaching unseen values or removing source-selection failures from the score.

## Compile and train

Keep captures and generated data outside the checkout. The compiler requires
an exact-hash independent review receipt and fails on source identity, schema,
source-family overlap or evidence mismatches:

```sh
python -m training.realpage_dataset \
  --labels /private/realpage-corpus/tasks.json \
  --snapshots /private/realpage-corpus/prepared \
  --review-receipt /private/realpage-corpus/independent-review.json \
  --grounding-negatives /private/realpage-corpus/grounding-negatives.json \
  --negative-review-receipt /private/realpage-corpus/negative-review.json \
  --output-dir /private/realpage-data
```

The optional negative-file pair adds only training citation rows. All business
case files and validation/test conversations stay unchanged. The manifest
binds every row/case file, policy, label and review identity. The row validator
reconstructs the exact production prompts and schemas; a row cannot supply an
arbitrary schema or train application provenance/status fields.

Use the separately installed pinned MLX training environment described in the
[training guide](README.md). The new target format is explicit:

```sh
python -m training.job start --python /private/mlx-training/bin/python \
  --model /private/models/qwen3-base --data /private/realpage-data \
  --run-dir /private/realpage-run --target-format realpage_grounded_v1 \
  --iters 512 --learning-rate 0.00003 --max-seq-length 8192 \
  --memory-gib 8 --max-seconds 5400
```

The measured pilot started a fresh adapter from the pinned base, without the
previous citation-v3 adapter. It used rank eight, the last eight layers' q/v
projections, batch one and accumulation four: 512 microsteps, 128 optimizer
updates. No example was truncated; the longest training conversation was
6,102 tokens. One final checkpoint was preregistered, with no checkpoint search.
The job supervises preflight, a disposable gradient/save/reload sanity check,
and actual training under a persistent lock and finite deadline.

## Frozen comparison

Use a separately verified loopback model server. Freeze the cases, model weight
checksum, source, prompts, business schemas, code and interpreter before calls:

```sh
python -m training.realpage_evaluate /private/realpage-data/valid.cases.jsonl \
  --model YOUR_VERIFIED_MODEL_ID --revision YOUR_IMMUTABLE_REVISION \
  --weights-sha256 VERIFIED_SHA256 --base-url http://127.0.0.1:8004/v1 \
  --freeze-candidate /private/realpage-eval/freeze.json
```

Then replace `--freeze-candidate` with `--freeze`, add `--freeze-sha` and
`--output-dir`. The comparison uses greedy decoding, thinking disabled,
768 output tokens per stage, 45 seconds per stage, at most two calls per case
and no repair. Grounding catalogs are derived from predictions, never gold.
Large complete snapshots may occupy up to 8 MB per case row; the model source
window remains 8,000 characters.

Public receipts separate original business JSON correctness, schema validity,
exact records, abstention, accepted facts, wrong accepted facts, citation
selection, reviewed-anchor agreement, source coverage, tokens and latency.
Anchor agreement is a conservative reviewed-location proxy: a different valid
citation may disagree. Interrupted cases retain an unknown-call reservation
and are not silently resubmitted. Raw responses stay in private checkpoints.

This English technical-site pilot, flat custom schemas and contact-based null
controls do not establish universal web accuracy, broader nested-schema
support or an advantage over Firecrawl's native extraction models.
