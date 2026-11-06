import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator

from .schemas import StrictModel


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0"}:
        raise ValueError("Boolean settings must be true, false, 1 or 0")
    return normalized in {"true", "1"}


class Settings(StrictModel):
    state_dir: Path = Path(".scraper")
    brightdata_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None
    firecrawl_api_key: SecretStr | None = None
    hf_provider: str | None = None
    hf_model: str | None = None
    inference_backend: Literal["huggingface", "local"] = "huggingface"
    retrieval_backend: Literal["brightdata", "direct", "browser", "auto", "firecrawl"] = "brightdata"
    firecrawl_base_url: str | None = Field(default=None, max_length=200)
    firecrawl_runtime_id: str | None = Field(default=None, max_length=80)
    firecrawl_trust_service_egress: bool = False
    user_agent: str = Field(default="EvidenceAIScraper/0.5", min_length=1, max_length=200)
    browser_timeout_seconds: float = Field(default=45.0, ge=1, le=120)
    browser_max_requests: int = Field(default=80, ge=1, le=500)
    browser_max_total_bytes: int = Field(default=10_000_000, ge=1024, le=100_000_000)
    browser_max_attempts: int = Field(default=2, ge=1, le=3)
    browser_resource_domains: list[str] = Field(default_factory=list)
    browser_executable_path: str | None = Field(default=None, max_length=4000)
    local_model: str | None = None
    local_base_url: str = "http://127.0.0.1:8000/v1"
    local_api_key: SecretStr | None = None
    local_source_bound_evidence: bool = False
    local_source_quote_evidence: bool = False
    local_two_stage_grounding: bool = False
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

    @field_validator("user_agent")
    @classmethod
    def safe_user_agent(cls, value: str) -> str:
        if not value.isascii() or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("User agent must be printable ASCII without control characters")
        if not value.strip():
            raise ValueError("User agent cannot be blank")
        return value

    @model_validator(mode="after")
    def exclusive_local_citations(self):
        if (
            sum(
                (
                    self.local_source_bound_evidence,
                    self.local_source_quote_evidence,
                    self.local_two_stage_grounding,
                )
            )
            > 1
        ):
            raise ValueError("Local citation modes are mutually exclusive")
        return self

    @model_validator(mode="after")
    def firecrawl_service_configuration(self):
        if self.retrieval_backend == "firecrawl":
            from .retrieval.firecrawl import firecrawl_cache_options

            firecrawl_cache_options(
                self.firecrawl_base_url, self.firecrawl_runtime_id, self.firecrawl_trust_service_egress
            )
        return self

    @classmethod
    def from_env(cls):
        fields = {
            "SCRAPER_STATE_DIR": ("state_dir", Path),
            "BRIGHTDATA_API_KEY": ("brightdata_api_key", SecretStr),
            "HF_TOKEN": ("hf_token", SecretStr),
            "FIRECRAWL_API_KEY": ("firecrawl_api_key", SecretStr),
            "FIRECRAWL_BASE_URL": ("firecrawl_base_url", str),
            "FIRECRAWL_RUNTIME_ID": ("firecrawl_runtime_id", str),
            "FIRECRAWL_TRUST_SERVICE_EGRESS": ("firecrawl_trust_service_egress", _boolean),
            "HF_PROVIDER": ("hf_provider", str),
            "HF_MODEL": ("hf_model", str),
            "SCRAPER_INFERENCE_BACKEND": ("inference_backend", str),
            "SCRAPER_RETRIEVAL_BACKEND": ("retrieval_backend", str),
            "SCRAPER_USER_AGENT": ("user_agent", str),
            "BROWSER_TIMEOUT_SECONDS": ("browser_timeout_seconds", float),
            "BROWSER_MAX_REQUESTS": ("browser_max_requests", int),
            "BROWSER_MAX_TOTAL_BYTES": ("browser_max_total_bytes", int),
            "BROWSER_MAX_ATTEMPTS": ("browser_max_attempts", int),
            "BROWSER_RESOURCE_DOMAINS": (
                "browser_resource_domains",
                lambda s: [x.strip().lower() for x in s.split(",") if x.strip()],
            ),
            "BROWSER_EXECUTABLE_PATH": ("browser_executable_path", str),
            "LOCAL_MODEL": ("local_model", str),
            "LOCAL_BASE_URL": ("local_base_url", str),
            "LOCAL_API_KEY": ("local_api_key", SecretStr),
            "LOCAL_SOURCE_BOUND_EVIDENCE": ("local_source_bound_evidence", _boolean),
            "LOCAL_SOURCE_QUOTE_EVIDENCE": ("local_source_quote_evidence", _boolean),
            "LOCAL_TWO_STAGE_GROUNDING": ("local_two_stage_grounding", _boolean),
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
            "SCRAPER_MAX_SOURCE_CHARS": ("max_source_chars", int),
            "SCRAPER_MAX_OUTPUT_TOKENS": ("max_output_tokens", int),
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
