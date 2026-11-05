import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr

from .schemas import StrictModel


class Settings(StrictModel):
    state_dir: Path = Path(".scraper")
    brightdata_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None
    firecrawl_api_key: SecretStr | None = None
    hf_provider: str | None = None
    hf_model: str | None = None
    inference_backend: Literal["huggingface", "local"] = "huggingface"
    retrieval_backend: Literal["brightdata", "direct"] = "brightdata"
    local_model: str | None = None
    local_base_url: str = "http://127.0.0.1:8000/v1"
    local_api_key: SecretStr | None = None
    allowed_domains: list[str] = Field(default_factory=list)
    wait_seconds: float = Field(default=180.0, ge=0, le=1800)
    hf_timeout: float = Field(default=60.0, gt=0, le=300)
    max_source_chars: int = Field(default=48000, gt=0)
    max_response_bytes: int = Field(default=5_000_000, ge=1024, le=100_000_000)
    max_snapshot_bytes: int = Field(default=20_000_000, ge=1024, le=400_000_000)
    max_output_tokens: int = Field(default=4096, gt=0)
    max_spend_usd: float | None = Field(default=None, ge=0)
    retrieval_record_usd: float | None = Field(default=None, ge=0)
    hf_input_million_usd: float | None = Field(default=None, ge=0)
    hf_output_million_usd: float | None = Field(default=None, ge=0)
    unlocker_zone: str | None = None
    render_record_usd: float | None = Field(default=None, ge=0)
    firecrawl_record_usd: float | None = Field(default=None, ge=0)
    retention_days: int = Field(default=30, ge=1)

    @classmethod
    def from_env(cls):
        fields = {
            "SCRAPER_STATE_DIR": ("state_dir", Path),
            "BRIGHTDATA_API_KEY": ("brightdata_api_key", SecretStr),
            "HF_TOKEN": ("hf_token", SecretStr),
            "FIRECRAWL_API_KEY": ("firecrawl_api_key", SecretStr),
            "HF_PROVIDER": ("hf_provider", str),
            "HF_MODEL": ("hf_model", str),
            "SCRAPER_INFERENCE_BACKEND": ("inference_backend", str),
            "SCRAPER_RETRIEVAL_BACKEND": ("retrieval_backend", str),
            "LOCAL_MODEL": ("local_model", str),
            "LOCAL_BASE_URL": ("local_base_url", str),
            "LOCAL_API_KEY": ("local_api_key", SecretStr),
            "SCRAPER_ALLOWED_DOMAINS": (
                "allowed_domains",
                lambda s: [x.strip().lower() for x in s.split(",") if x.strip()],
            ),
            "SCRAPER_WAIT_SECONDS": ("wait_seconds", float),
            "SCRAPER_MAX_SPEND_USD": ("max_spend_usd", float),
            "BRIGHTDATA_RECORD_USD": ("retrieval_record_usd", float),
            "HF_INPUT_MILLION_USD": ("hf_input_million_usd", float),
            "HF_OUTPUT_MILLION_USD": ("hf_output_million_usd", float),
            "BRIGHTDATA_UNLOCKER_ZONE": ("unlocker_zone", str),
            "BRIGHTDATA_RENDER_RECORD_USD": ("render_record_usd", float),
            "FIRECRAWL_RECORD_USD": ("firecrawl_record_usd", float),
            "SCRAPER_RETENTION_DAYS": ("retention_days", int),
            "HF_TIMEOUT_SECONDS": ("hf_timeout", float),
            "SCRAPER_MAX_RESPONSE_BYTES": ("max_response_bytes", int),
            "SCRAPER_MAX_SNAPSHOT_BYTES": ("max_snapshot_bytes", int),
        }
        return cls(
            **{
                field: converter(os.environ[key])
                for key, (field, converter) in fields.items()
                if os.environ.get(key)
            }
        )
