"""Load private deployment settings without shell evaluation, then run the app."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
from pathlib import Path

ALLOWED_KEYS = frozenset(
    {
        "BRIGHTDATA_API_KEY",
        "HF_TOKEN",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_BASE_URL",
        "FIRECRAWL_RUNTIME_ID",
        "FIRECRAWL_TRUST_SERVICE_EGRESS",
        "HF_PROVIDER",
        "HF_MODEL",
        "SCRAPER_INFERENCE_BACKEND",
        "SCRAPER_RETRIEVAL_BACKEND",
        "SCRAPER_USER_AGENT",
        "BROWSER_TIMEOUT_SECONDS",
        "BROWSER_MAX_REQUESTS",
        "BROWSER_MAX_TOTAL_BYTES",
        "BROWSER_MAX_ATTEMPTS",
        "BROWSER_RESOURCE_DOMAINS",
        "BROWSER_EXECUTABLE_PATH",
        "LOCAL_MODEL",
        "LOCAL_BASE_URL",
        "LOCAL_API_KEY",
        "LOCAL_SOURCE_BOUND_EVIDENCE",
        "LOCAL_SOURCE_QUOTE_EVIDENCE",
        "LOCAL_TWO_STAGE_GROUNDING",
        "SCRAPER_ALLOWED_DOMAINS",
        "SCRAPER_WAIT_SECONDS",
        "SCRAPER_MAX_SPEND_USD",
        "BRIGHTDATA_RECORD_USD",
        "HF_INPUT_MILLION_USD",
        "HF_OUTPUT_MILLION_USD",
        "BRIGHTDATA_UNLOCKER_ZONE",
        "BRIGHTDATA_RENDER_RECORD_USD",
        "FIRECRAWL_RECORD_USD",
        "SCRAPER_RETENTION_DAYS",
        "HF_TIMEOUT_SECONDS",
        "SCRAPER_MAX_SOURCE_CHARS",
        "SCRAPER_MAX_OUTPUT_TOKENS",
        "SCRAPER_MAX_RESPONSE_BYTES",
        "SCRAPER_MAX_SNAPSHOT_BYTES",
    }
)
RETRIEVAL_KEYS = (
    "BRIGHTDATA_API_KEY",
    "SCRAPER_ALLOWED_DOMAINS",
    "SCRAPER_MAX_SPEND_USD",
    "BRIGHTDATA_RECORD_USD",
)
INFERENCE_KEYS = (
    "HF_TOKEN",
    "HF_PROVIDER",
    "HF_MODEL",
    "HF_INPUT_MILLION_USD",
    "HF_OUTPUT_MILLION_USD",
)
PRICE_KEYS = (
    "SCRAPER_MAX_SPEND_USD",
    "BRIGHTDATA_RECORD_USD",
    "HF_INPUT_MILLION_USD",
    "HF_OUTPUT_MILLION_USD",
    "BRIGHTDATA_RENDER_RECORD_USD",
    "FIRECRAWL_RECORD_USD",
)


class ConfigurationError(Exception):
    """Intentionally contains only a stable code, never untrusted config data."""


def load_environment(root: Path, config: Path, inherited: dict[str, str] | None = None) -> dict[str, str]:
    if not root.is_absolute() or not config.is_absolute():
        raise ConfigurationError("DEPLOY_PATH_INVALID")
    try:
        info = config.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ConfigurationError("DEPLOY_CONFIG_PERMISSIONS")
        if info.st_size > 65_536:
            raise ConfigurationError("DEPLOY_CONFIG_TOO_LARGE")
        values = json.loads(config.read_text())
    except ConfigurationError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise ConfigurationError("DEPLOY_CONFIG_UNREADABLE") from None
    if not isinstance(values, dict) or any(
        key not in ALLOWED_KEYS
        or not isinstance(value, str)
        or any(ord(character) < 32 for character in value)
        for key, value in values.items()
    ):
        raise ConfigurationError("DEPLOY_CONFIG_INVALID")
    environment = dict(os.environ if inherited is None else inherited)
    environment.update(values)
    for key in PRICE_KEYS:
        if environment.get(key):
            try:
                number = float(environment[key])
                valid = math.isfinite(number) and number >= 0
            except ValueError:
                valid = False
            if not valid:
                raise ConfigurationError("DEPLOY_PRICING_INVALID")
    if environment.get("HF_PROVIDER") == "auto":
        raise ConfigurationError("DEPLOY_PROVIDER_INVALID")
    environment["SCRAPER_STATE_DIR"] = str(root.resolve() / "state")
    return environment


def status(environment: dict[str, str]) -> dict:
    configured = {key: bool(environment.get(key, "").strip()) for key in sorted(ALLOWED_KEYS)}
    retrieval_backend = environment.get("SCRAPER_RETRIEVAL_BACKEND") or "brightdata"
    credentialless_retrieval = retrieval_backend in {"direct", "browser", "auto", "firecrawl"}
    browser_runtime_required = retrieval_backend in {"browser", "auto"}
    missing_retrieval = [
        key
        for key in (("SCRAPER_ALLOWED_DOMAINS",) if credentialless_retrieval else RETRIEVAL_KEYS)
        if not configured[key]
    ]
    if retrieval_backend == "firecrawl":
        missing_retrieval.extend(
            key
            for key in ("FIRECRAWL_BASE_URL", "FIRECRAWL_RUNTIME_ID", "FIRECRAWL_TRUST_SERVICE_EGRESS")
            if not configured[key]
        )
        if (
            environment.get("FIRECRAWL_TRUST_SERVICE_EGRESS", "").strip().lower() not in {"true", "1"}
            and "FIRECRAWL_TRUST_SERVICE_EGRESS" not in missing_retrieval
        ):
            missing_retrieval.append("FIRECRAWL_TRUST_SERVICE_EGRESS")
    local = environment.get("SCRAPER_INFERENCE_BACKEND") == "local"
    missing_inference = [
        key for key in (("LOCAL_MODEL", "LOCAL_BASE_URL") if local else INFERENCE_KEYS) if not configured[key]
    ]
    return {
        "status": "configuration_valid",
        "configured": configured,
        "state_directory": environment["SCRAPER_STATE_DIR"],
        "live_retrieval_configured": not missing_retrieval,
        "live_inference_configured": not missing_inference,
        "inference_backend": "local" if local else "huggingface",
        "retrieval_backend": retrieval_backend,
        "browser_runtime_required": browser_runtime_required,
        "browser_runtime_verified": False,
        "missing_for_retrieval": missing_retrieval,
        "missing_for_inference": missing_retrieval + missing_inference,
        "live_verified": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("mode", choices=("cli", "mcp", "status"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        environment = load_environment(args.root, args.config or args.root / "config.json")
        os.environ.update(environment)
        try:
            from ai_scraper.config import Settings

            Settings.from_env()
        except (ValueError, TypeError):
            raise ConfigurationError("DEPLOY_CONFIG_INVALID") from None
        if args.mode == "status":
            if args.arguments:
                raise ConfigurationError("DEPLOY_ARGUMENTS_INVALID")
            print(json.dumps(status(environment), indent=2))
            return 0
        if args.mode == "mcp" and args.arguments:
            raise ConfigurationError("DEPLOY_ARGUMENTS_INVALID")
        if args.mode == "cli":
            from ai_scraper.cli import main as cli_main

            return cli_main(args.arguments)
        from ai_scraper.mcp_server import main as mcp_main

        mcp_main()
        return 0
    except ConfigurationError as error:
        print(f"{error}: Invalid or unavailable private deployment configuration", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 -- do not expose private settings in startup tracebacks
        print("DEPLOY_START_FAILED: Application startup failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
