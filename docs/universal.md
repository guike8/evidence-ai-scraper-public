# General, preset and custom extraction

`ai-scraper extract URL` uses `page_v1`. It returns literal text blocks, metadata, headings, links and tables, together with source evidence and application-assigned provenance. It is useful on an arbitrary supported public HTML/text page without forcing its contents into a product schema.

```sh
ai-scraper schemas
ai-scraper schema article --provider-compatible
ai-scraper extract https://example.com --output page.json
```

These addresses illustrate command syntax rather than an evaluated corpus.

## Choose a contract

| Schema | Short aliases | Data |
| --- | --- | --- |
| `page_v1` | `page`, `general`, `auto`, `document` | Title, exact text blocks, headings, links, tables, description, author, date and language. |
| `article_v1` | `article`, `news` | Main article title, author, published date, description and exact text blocks. |
| `organization_v1` | `organization`, `org`, `company` | Name, description, website, public emails, phone numbers and addresses. |
| `person_v1` | `person`, `contact` | Name, role, organization, biography, website and published emails. |
| `job_v1` | `job` | Title, organization, location, employment type, salary text and description. |
| `event_v1` | `event` | Name, source-exact start/end dates, location, organizer and description. |
| `listing_v1` | `listing` | Listing title, description and listed items with names, URLs and descriptions. |
| `product_v1` | `product` | The original product fields and completeness rules. |
| `custom_v1` | `custom` | A request-specific set of bounded typed fields. |

```sh
ai-scraper extract https://example.com/article --schema article
ai-scraper extract https://example.com/about --schema org
ai-scraper extract https://example.com/team/ada --schema person
ai-scraper extract https://example.com/job --schema job
ai-scraper extract https://example.com/event --schema event
ai-scraper extract https://example.com/directory --schema listing
ai-scraper extract https://example.com/product --schema product
```

Dates, salaries and other text facts are preserved as written. Missing information stays null or empty. An ambiguous source entity is not resolved by guessing.

## Request your own fields

```sh
ai-scraper extract https://example.com/about \
  --fields 'name,contact,employee_count:integer,tags:string_list' \
  --ask 'Extract only explicitly published organization information.' \
  --output custom.json

ai-scraper extract https://example.com/about \
  --fields-file examples/universal-fields.json --output custom.json
```

A bare field name means `string`. Other allowed types are `number`, `integer`, `boolean` and `string_list`. A descriptor file is a JSON list with `name`, `type`, `description`, `required` and `critical`. The shipped file is an illustrative organization contract with no private or real contact data.

Field names are bounded snake_case identifiers, and at most 32 fields are accepted. Descriptions are bounded to 500 characters and the request instruction to 2,000. Arbitrary JSON Schema, imports and executable definitions are not accepted. Custom field definitions are request-scoped and cannot mutate another request's model schema.

All custom scalar types allow null. Lists use an empty array for absence. Required fields determine whether a validated result is complete; critical fields determine whether missing information is terminal. A critical field is also required. A `number` must be finite, `integer` does not accept a boolean, and text cannot be substituted for a numeric value merely to satisfy a request.

Custom fields select `custom_v1` automatically when used with the default/general contract. They cannot be combined with an explicit unrelated preset such as `product_v1`.

## Ask an extractive question

```sh
ai-scraper extract https://example.com/visit \
  --ask 'Find the stated opening hours and wheelchair access information.' \
  --output answers.json
```

With no field list and the general/default schema, `--ask` creates a `matches` string array under `custom_v1`. The full instruction remains attached to the request. Each answer must be literal source text with a verifiable quote. Local validation rejects paraphrases or inferred values that lack exact source support. An exact quote can still be irrelevant to the requested field, and a model may repeat passages; evaluate task meaning on reviewed examples. If no supported answer is found, the result reports missing information.

An instruction supplied with an article, organization or other specific preset scopes that preset; it does not remove required fields. General/page/auto selections with an instruction and no fields use extractive `matches`.

## MCP

Start the local stdio server with `uv run ai-scraper-mcp`. For an MCP client, use an absolute project path and the environment needed by that process:

```json
{
  "mcpServers": {
    "evidence-scraper": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/evidence-ai-scraper", "run", "ai-scraper-mcp"],
      "env": {
        "SCRAPER_RETRIEVAL_BACKEND": "direct",
        "SCRAPER_ALLOWED_DOMAINS": "example.org",
        "SCRAPER_INFERENCE_BACKEND": "local",
        "LOCAL_BASE_URL": "http://127.0.0.1:8000/v1",
        "LOCAL_MODEL": "your-local-model-id"
      }
    }
  }
}
```

Configure and test the local model server separately. Client configuration does not start or download a model. The tools are `list_schemas`, `extract_url`, `resume_extraction`, `crawl_site`, `resume_crawl` and `crawl_status`, with `page_v1` as the extraction default. Discovery itself starts no retrieval, model or browser clients. See [crawl tools and browser actions](extensions.md#mcp) and the [portable deployment guide](deployment.md) for remote stdio.

For example, call `extract_url` with:

```json
{
  "url": "https://example.com/about",
  "fields": [
    {"name": "name", "type": "string", "critical": true},
    {"name": "contact", "type": "string", "required": false}
  ],
  "instruction": "Use only explicitly displayed organization information."
}
```

Or supply only `url` and `instruction` for extractive `matches`. The same request builder validates CLI and MCP inputs. `resume_extraction(job_id)` preserves the original request's contract and source; it does not substitute a new question for a pending job. Typed results retain timestamps and structured data until the framework performs JSON serialization.

## Source evidence, caches and waiting jobs

Page text is returned as source-aligned blocks so each populated value can be verified. Every nonempty leaf, including table cells and nested items, requires a literal section excerpt or a verified JSON-LD pointer. Evidence occurrence and schema validity do not establish every semantic relationship. Link normalization is limited to resolving a literal source URL against the saved page URL. Structural empty table cells are not fabricated facts.

Different contracts for the same URL can reuse one fresh saved retrieval. If that retrieval is still pending, a dependent extraction request is persisted and can be resumed when the original retrieval completes. Schema fingerprints, field descriptions and task instructions isolate extraction caches and repair limits, so one field definition or question cannot receive another's cached answer. Replay has a separate content-hashed identity and cannot satisfy live retrieval.

General inference can deduplicate identical textual representations and use a declared main/article scope for oversized page/article inputs. The complete snapshot, all JSON-LD and attributes remain retained; the saved selection descriptor preserves original section IDs. Custom tasks keep full-page source coverage and fail explicitly if the source cannot fit. Each operation permits at most one semantic repair against the same immutable source. Required and critical fields, evidence and domain rules remain local checks even when a provider accepts the JSON schema.

## Public-site access and limits

The credentialless local profile uses direct retrieval and optional local inference. `SCRAPER_ALLOWED_DOMAINS=*` permits any supported public host, with the same DNS/IP/port checks on every redirect. Private/local destinations, unsafe ports and credential-bearing URLs remain rejected. Native direct connections pin a verified public IP and preserve TLS SNI/hostname verification to prevent a DNS-rebinding gap. Exact hostnames can replace `*` to narrow scope.

This universal contract interface does not make every format or access path supported. Public HTML/text pages support bounded link discovery and optional anonymous JavaScript rendering with caller-authored click, selector-wait and scroll actions; see [crawling and browser limits](extensions.md). Authenticated sessions, form submission, challenges and binary/PDF/OCR inputs need additional adapters. The scraper fails explicitly on unsupported or oversized content. Local inference needs no API account; new public URLs still need a working internet connection.

The legacy library constructor default remains `product_v1`. New library calls can use `ExtractionRequest(url=..., schema_id="page_v1")` or the shared `ai_scraper.cli.build_request` helper for the CLI/MCP routing behavior.
