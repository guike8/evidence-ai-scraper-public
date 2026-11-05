# Deployment launcher

`launch.py` loads a private JSON configuration, validates supported environment keys, sets a shared state directory and starts CLI, stdio MCP or status mode. It does not install a model, create a service, configure SSH or change network settings.

```sh
/absolute/install/current/.venv/bin/python \
  /absolute/install/current/deploy/launch.py \
  --root /absolute/install status
```

The installation root and configuration path must be absolute. The default private configuration is `config.json` under that root; `--config` can select another absolute file. Values must be strings, the file must be owned by the running user, and its mode must be 0600. Unknown settings, invalid prices and unsupported startup arguments fail without printing configuration values.

Use `cli` followed by normal CLI arguments, or `mcp` without extra arguments. Status prints setting-presence flags and missing setting names; it does not fetch pages or call inference. A configured profile is not a live verification result.

Keep configuration and `state/` outside release trees. Reuse the same state when changing `current` so pending jobs and spending reservations remain durable. Build virtual environments on the target machine instead of copying them.

See the [portable deployment guide](../docs/deployment.md) for layout, upgrade checks and SSH stdio. Never commit deployment credentials, SSH keys, retrieved private content or model weights.
