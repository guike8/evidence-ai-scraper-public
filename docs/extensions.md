# Crawling and anonymous browser rendering

The crawler discovers public links from saved pages and runs the existing
extraction pipeline for each permitted URL. Browser retrieval supplies rendered
HTML for JavaScript pages. Extraction schemas, validation, evidence and JSON
exports remain shared with single-URL extraction.

## Crawl a site

```sh
uv run ai-scraper crawl https://example.org/ \
  --max-pages 5 --max-depth 1 --wait-seconds 60 --output crawl.json
```

The crawl command defaults to direct HTTP and the exact seed hosts when no
retrieval backend or domain allowlist is configured. No API credentials are
needed. Semantic extraction uses a model only when an inference backend is
explicitly configured; otherwise deterministic extraction supplies available
facts. The existing single-URL extraction defaults are unchanged.

The default bounds are 20 pages, depth 2, a 1,000-entry frontier, 500 candidate
links per page and a one-second host delay. `robots.txt` rules and pacing apply.
HTTP 404 for robots means no rules were found; a denied or temporarily
unavailable robots policy is retained as a skipped/pending condition rather
than silently permitting requests.

Links resolve against the observed page URL and an eligible HTML base URL.
Fragments are removed for deduplication; query ordering and repeated parameters
are preserved. Off-scope hosts, nonpublic addresses, credential-bearing URLs
and unsafe schemes/ports remain subject to the URL policy. Add an exact host
with `--scope-domain docs.example.org` when the crawl should include it. An
explicit global allowlist still restricts which hosts may be selected.

Use the same schema interface for every discovered page:

```sh
uv run ai-scraper crawl https://example.org/news/ --schema article_v1 \
  --max-pages 10 --max-depth 1
uv run ai-scraper crawl https://example.org/directory/ \
  --fields name,year:integer --instruction 'Only the displayed name and year' \
  --max-pages 10
```

The crawl summary contains per-page job IDs and saved result paths. Each page
result has the usual business data, status, source evidence and provenance.
Failed extraction and scope/budget limits remain visible in the summary.

## Resume and inspect

```sh
uv run ai-scraper crawl-status CRAWL_ID
uv run ai-scraper resume-crawl CRAWL_ID --wait-seconds 60 --output crawl.json
```

The frontier, host pacing and page job IDs persist in SQLite. The crawler
commits each extraction job ID before fetching that page. Restart uses that
same job, including a pending vendor snapshot or a submission requiring
reconciliation. A crawl ID refers to its saved plan; a new plan gets a new
crawl. Status inspection submits no retrieval or model work.

Resume with the original extraction and browser settings. For example, a crawl
started with `--render-fallback` needs that flag on `resume-crawl` too, unless
`SCRAPER_RETRIEVAL_BACKEND=auto` is configured. A changed configuration is
reported rather than silently changing the saved work.

## Render public JavaScript pages

Install the optional browser dependency and its Chromium build on a machine
with outbound internet:

```sh
uv sync --extra browser
uv run playwright install chromium
```

Select Chromium for every URL with `--browser`, or use direct HTTP with
JavaScript fallback through `--render-fallback`:

```sh
export SCRAPER_ALLOWED_DOMAINS=example.org
uv run ai-scraper extract https://example.org/app/ --browser --output page.json
uv run ai-scraper crawl https://example.org/app/ --render-fallback --max-pages 5
```

The environment equivalents are `SCRAPER_RETRIEVAL_BACKEND=browser` and
`SCRAPER_RETRIEVAL_BACKEND=auto`. The browser is initialized only when retrieval
actually needs it. It uses an anonymous, isolated Chromium context rather than
the user's Chrome profile. Public browser requests use the pinned native HTTP
transport; authorization and cookie headers are stripped. Request, byte,
attempt and time limits bound each render. Browser routing is not an OS network
sandbox.

Top-level HTTP redirects are followed by the controller through checked
navigation requests, with at most five hops. Redirecting subresources and
iframes currently fail with `BROWSER_RESOURCE_REDIRECT_UNSUPPORTED`; their
redirects are never handed to Chromium outside interception. Declare required
asset hosts explicitly with `BROWSER_RESOURCE_DOMAINS`, including transitive
font or script dependencies. The navigation scope stays separate.

Supported interactions are caller-authored CSS clicks, selector waits and
scrolls. Save a plan such as:

```json
[
  {"kind": "click", "selector": "button.more"},
  {"kind": "wait_for", "selector": "#details", "timeout_ms": 5000},
  {"kind": "scroll", "pixels": 600}
]
```

Then supply `--browser-actions actions.json` with `--browser` or
`--render-fallback`. Plans are limited to ten steps and 64 KiB. They are trusted
request configuration; the extraction model and webpage cannot supply actions.
Click and wait selectors must each identify one element; ambiguous selectors
are rejected by Chromium rather than choosing an arbitrary match.
Page HTTP requests use GET/HEAD. Forms, account sessions and CAPTCHA solving
are outside this anonymous retrieval interface.

`SCRAPER_USER_AGENT` sets the crawler/browser user agent and robots identity.
The browser settings in `.env.example` control the render timeout, request and
byte budgets, attempts, optional resource hosts and installed executable path.
Domain extensions for resources still undergo public-address checks.

The Studio's VPN/egress policy remains an independent deployment setting.
Offline browser installation and fixture tests can run there with transferred
packages; live retrieval still needs permitted outbound access.

## MCP

The stdio server exposes `crawl_site`, `resume_crawl` and `crawl_status` alongside
`extract_url`, `resume_extraction` and `list_schemas`. Tool discovery initializes
no retrieval, model or browser clients. `crawl_site` accepts the same bounds,
schemas, typed fields and browser actions as the CLI, with `retrieval_mode`
selecting configured, direct, browser or auto retrieval. `extract_url` accepts
browser actions through its configured browser/auto profile. Use
`resume_extraction` with a page job ID to obtain that page's typed result.
Set `resume_crawl.retrieval_mode` to the original mode when it was supplied on
`crawl_site`. An explicit MCP `scope_domains` list includes the seed hosts and
any additional exact hosts; omitting it keeps the crawl within seed hosts.
