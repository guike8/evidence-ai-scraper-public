# Deploy the CLI and MCP service

This layout keeps application releases separate from private settings and durable state. Choose your own installation root, operating-system account, service manager and SSH host alias. No machine-specific launcher or SSH key is distributed.

```text
installation-root/
  releases/<release-id>/    # application source and its local .venv
  current -> releases/<release-id>
  config.json              # owned by the service user, mode 0600
  state/                   # jobs, sources, descriptors and budgets, mode 0700
  bin/                     # optional stable launchers
```

Create each virtual environment on its target machine with `uv sync --locked --python 3.12`. Keep private settings, state, model weights and SSH keys outside source control. Model installation and service startup are separate from this Python package.

## Private configuration

`config.json` is a flat JSON object mapping supported environment-variable names to string values. For a direct/local profile:

```json
{
  "SCRAPER_RETRIEVAL_BACKEND": "direct",
  "SCRAPER_INFERENCE_BACKEND": "local",
  "SCRAPER_ALLOWED_DOMAINS": "example.org",
  "LOCAL_BASE_URL": "http://127.0.0.1:8000/v1",
  "LOCAL_MODEL": "your-local-model-id",
  "HF_TIMEOUT_SECONDS": "300"
}
```

The configuration must be a regular file owned by the running user with no group/other permissions. Omit unconfigured credentials and paid prices. The launcher validates its supported keys without evaluating shell text and fixes `SCRAPER_STATE_DIR` to the shared `state/` directory. Its status mode reports setting presence rather than secret values.

Use absolute paths chosen for your deployment:

```sh
RELEASE_PYTHON=/absolute/install/current/.venv/bin/python
LAUNCHER=/absolute/install/current/deploy/launch.py
INSTALL_ROOT=/absolute/install
"$RELEASE_PYTHON" "$LAUNCHER" --root "$INSTALL_ROOT" status
"$RELEASE_PYTHON" "$LAUNCHER" --root "$INSTALL_ROOT" cli schemas
"$RELEASE_PYTHON" "$LAUNCHER" --root "$INSTALL_ROOT" mcp
```

The model endpoint remains on loopback. The local adapter targets oMLX structured output; test the exact installed server/model/schema with the CLI probes. Other OpenAI-compatible endpoints require their own compatibility checks.

## Upgrade and recover

Before activation, run the fixture suite, deterministic replay, selected local-model probes and MCP discovery. Activate a release by atomically replacing the `current` symlink. Retain the previous release and use the same state directory so pending jobs, source identities and spending reservations survive upgrades.

Schedule `ai-scraper prune` according to the retention policy. Do not delete the state directory to reset jobs or spending. Back up private configuration and state using your deployment's storage policy. Snapshots contain retrieved page content and should be retained only as needed.

## SSH stdio

A client can launch the MCP server through an existing SSH configuration, without exposing a public model or MCP listener:

```json
{
  "mcpServers": {
    "evidence-scraper": {
      "command": "ssh",
      "args": ["-T", "YOUR_SSH_HOST_ALIAS", "/absolute/install/bin/ai-scraper-mcp"]
    }
  }
}
```

Configure the host alias, trusted host identity and authentication locally. A dedicated forced-command key can restrict access to the MCP launcher. Keep private keys outside the repository and avoid credentials in SSH arguments. The stdio launcher must write protocol messages to stdout and diagnostics to stderr.

Network routing, firewall and VPN configuration belong to the deployment. New public URLs require outbound connectivity; saved-record replay can operate without public-network access.
