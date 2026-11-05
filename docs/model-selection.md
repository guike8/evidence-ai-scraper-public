# Check model capability before selecting it

Choose an explicit server/provider and model identifier, then test the exact registered schema. A model catalog, valid JSON response or successful type check does not establish factual extraction quality.

## Local inference

Start a separately installed local server and model, then export the direct/local profile:

```sh
export SCRAPER_RETRIEVAL_BACKEND=direct
export SCRAPER_ALLOWED_DOMAINS=example.org
export SCRAPER_INFERENCE_BACKEND=local
export LOCAL_BASE_URL=http://127.0.0.1:8000/v1
export LOCAL_MODEL=your-local-model-id
export HF_TIMEOUT_SECONDS=300
uv run ai-scraper probe --schema page_v1
uv run ai-scraper probe --schema custom_v1 \
  --fields 'title,count:integer,amount:number,active:boolean,tags:string_list'
```

The local adapter targets [oMLX](https://github.com/jundot/omlx). It sends both the standard JSON-schema response format and oMLX's `structured_outputs` configuration, disables tools and thinking, and rejects a reported downgrade to unenforced output. `LOCAL_BASE_URL` must use loopback HTTP(S). Model weights are not bundled, downloaded or selected by the library. Other endpoints need independent compatibility testing even if they implement an OpenAI-style API.

The default `probe` exercises the nested product contract. `--schema` selects a different preset; custom descriptors select a request-specific contract. The synthetic probe checks grammar-compatible output, known expected values and the production evidence validator. Save provider/model identity, schema hash, runtime versions, token counts and latency with each result.

A passing probe establishes one request at one configuration. Evaluate representative pages, absent facts, conflicting entities, long input, malformed evidence, duplicate answers and embedded instructions before choosing a model for a task. Measure required-field precision/recall and unsupported populated values as well as speed; failures and timeouts remain in recall denominators.

## Hosted Hugging Face inference

For the optional hosted path, set `SCRAPER_INFERENCE_BACKEND=huggingface`, `HF_TOKEN`, an explicit `HF_PROVIDER` and `HF_MODEL`, plus actual input/output prices and a cumulative `SCRAPER_MAX_SPEND_USD`. Configure retrieval separately. The probe requires paid-call settings when hosted inference is selected. Do not infer account eligibility or zero pricing from advertised allowances.

The [HF structured-output guide](https://huggingface.co/docs/inference-providers/en/guides/structured-output) describes the response-format envelope. Provider compatibility varies. The schema compiler closes objects, requires keys with nullable values, preserves supported local references and nested collections, and omits some provider-incompatible constraints. Rich Pydantic, business and evidence checks remain enforced locally. Unsupported or recursive/external schema references fail before inference.

## Compare configurations

Use the same frozen source, task, schema, deadlines, source budget, output limit and hardware for each local candidate. Separate cold/warm runs, source planning, retrieval/rendering and model latency. Record exact model revision, quantization and runtime configuration. A small model can reduce resource use while failing evidence or semantic checks; do not promote it from speed alone.

The [credentialless comparison protocol](../benchmarks/README.md) supplies synthetic deterministic fixtures and an independent receipt scorer. The [competitive review](competitive-review.md) distinguishes those measurements from parser components, model tests and live-web coverage. None of the included authored fixture scores establishes general production accuracy.
