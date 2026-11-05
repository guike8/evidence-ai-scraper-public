# Evidence AI Scraper

Turn a public webpage into structured JSON with saved source evidence:

`URL → immutable snapshot → deterministic extraction → optional local model → validation → JSON`

The library, CLI and MCP server share retrieval, schema, evidence and recovery logic. The credentialless profile uses direct public HTTP and a separately configured local model. Bright Data and hosted Hugging Face adapters are optional.

## Quickstart: no model or network call

Use Python 3.12 and [uv](https://docs.astral.sh/uv/), then run from the repository root:

```sh
uv sync --locked --python 3.12
uv run ai-scraper replay examples/page-record.json \
  --url https://example.test/field-notes --output page.json
```

The authored example contains a title, headings, paragraphs, links and a table. Plain `replay` reads that saved record deterministically; it does not fetch a URL or call a model. The `.test` domain is reserved for fixtures. The output includes `data`, `evidence`, `provenance`, completeness and usage.

```sh
uv run ai-scraper schemas
uv run ai-scraper schema page --provider-compatible
```

## Retrieve a public page

Export settings in your shell or configure them through a process supervisor. The CLI does **not** automatically load `.env`; [.env.example](.env.example) documents the supported variables.

```sh
export SCRAPER_RETRIEVAL_BACKEND=direct
export SCRAPER_ALLOWED_DOMAINS=example.org
uv run ai-scraper extract https://example.org/ --output page.json
```

Literal metadata and page structure can be extracted without a model. To fill semantic gaps or use custom fields, configure a local inference service:

```sh
export SCRAPER_INFERENCE_BACKEND=local
export LOCAL_BASE_URL=http://127.0.0.1:8000/v1
export LOCAL_MODEL=your-local-model-id
export HF_TIMEOUT_SECONDS=300
uv run ai-scraper probe --schema page_v1
```

Install and start the model server separately. The local adapter targets [oMLX](https://github.com/jundot/omlx), sends `response_format` and `structured_outputs`, disables tools and thinking, and rejects a reported schema downgrade. A generic OpenAI-compatible endpoint is not sufficient evidence of compatibility: run the actual schema probe against the selected server/model. No model weights are bundled or downloaded automatically. Hardware, model licensing and energy requirements depend on your chosen model.

`LOCAL_BASE_URL` must be loopback HTTP(S). No cloud inference account or scraping API key is needed for the direct/local profile. Set exact comma-separated target hosts; `SCRAPER_ALLOWED_DOMAINS='*'` permits public hosts more broadly while retaining IP, port and redirect checks.

## Choose the output

CLI and MCP default to `page_v1`: literal text blocks, metadata, headings, links and tables. Presets include `article`, `org`, `person`, `job`, `event`, `listing` and `product`.

```sh
uv run ai-scraper extract https://example.org/ --schema article --output article.json
uv run ai-scraper extract https://example.org/ \
  --fields 'document_title,purpose' \
  --ask 'Extract the exact document title and the stated purpose.' --output custom.json
uv run ai-scraper extract https://example.org/ \
  --ask 'Find the stated purpose of this domain.' --output passages.json
```

Custom contracts accept at most 32 fields of type `string`, `number`, `integer`, `boolean` or `string_list`. Use `--fields-file examples/universal-fields.json` for descriptions and required/critical flags. Arbitrary nested custom JSON schemas are not supported. With the default general/page selection and no fields, an instruction requests a `matches` array of exact source passages.

These live URLs illustrate syntax. Missing facts stay missing; source occurrence alone does not prove that a passage answers the task. Evaluate semantic meaning and duplicate answers on reviewed examples. See [contracts, custom fields and MCP](docs/universal.md).

## Evidence and recovery

Every populated business leaf requires a verified excerpt in a saved section or a verified JSON-LD pointer. Application code assigns source hashes, retrieval details and model identifiers. Models provide business fields and evidence candidates, not provenance.

General inference saves a source-view descriptor with original evidence IDs. Identical text representations can collapse; oversized page/article tasks can use a declared main/article scope. Structured data, metadata and the complete snapshot remain retained. Custom tasks keep full-page coverage and fail explicitly when they cannot fit the configured context budget.

| Status | Meaning | CLI exit |
| --- | --- | ---: |
| `success` | Schema, required fields and evidence checks passed. | 0 |
| `partial` | Supported data has allowed completeness gaps or declared incomplete coverage. | 2 |
| `pending` | Durable work is waiting and can be resumed. | 3 |
| `failed` | A critical fact, source, schema, evidence or execution check failed; data is null. | 1 |

SQLite stores jobs and caches, while snapshots, descriptors and exports use private atomic files. Repairs reuse the saved source: one initial semantic call and at most one repair per source/configuration. Pending vendor snapshot IDs persist before polling, and ambiguous submissions require reconciliation rather than automatic duplicate work.

```sh
uv run ai-scraper resume LOCAL_JOB_ID --output result.json
uv run ai-scraper prune
```

Different questions can reuse one fresh retrieval while schema and instruction identities isolate their extraction results. Replay has a separate source identity from live retrieval. Default retention is 30 days; pruning preserves pending jobs and the spending ledger.

## Library and MCP

With the exported environment above, a standalone script can call the library:

```python
import asyncio

from ai_scraper.pipeline import create_pipeline
from ai_scraper.schemas import ExtractionRequest


async def main():
    pipeline = create_pipeline()
    try:
        result = await pipeline.extract_url(
            ExtractionRequest(url="https://example.org/", schema_id="page_v1")
        )
        print(result.model_dump_json(indent=2))
    finally:
        await pipeline.close()


asyncio.run(main())
```

Select the schema explicitly in new library integrations; the library request constructor retains its product default for compatibility. Retrieval, inference and validation are independently replaceable.

Start the stdio MCP server with `uv run ai-scraper-mcp`. It exposes `list_schemas`, `extract_url` and `resume_extraction`. See [MCP configuration](docs/universal.md#mcp) and the [portable deployment guide](docs/deployment.md).

## Scope, testing and comparison

The initial scope is known public HTML/text URLs. JavaScript/browser interaction, login, challenge handling, recursive discovery and PDF/OCR need additional adapters. Direct retrieval rejects credential-bearing URLs, private/local destinations, unsafe ports and redirects to disallowed destinations. Native connections pin an approved public IP and preserve TLS hostname verification.

```sh
uv run pytest -q
uv run ruff check src tests deploy benchmarks
uv run ruff format --check src tests deploy benchmarks
uv run python -m ai_scraper.competition run benchmarks/fixtures.json \
  --candidate local-fixtures --candidate-version YOUR_COMMIT --output comparison.json
```

The [comparison protocol](benchmarks/README.md) separates business accuracy, evidence validation, source-context fit and parser components. Its authored fixtures and component timings do not establish representative web accuracy or an overall lead. The [competitive review](docs/competitive-review.md) records measured scope and remaining gaps.

For optional paid adapters, configure credentials, explicit provider/model identifiers, actual account prices and a spending cap before live calls. See [model capability checks](docs/model-selection.md). Raw Unlocker rendering stays disabled unless its observed redirect destination can be verified.

## License

[MIT](LICENSE). Dependencies and separately obtained models retain their own licenses.
