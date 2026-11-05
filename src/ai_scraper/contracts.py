from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .schemas import AppError, ExtractionRequest, SourceSnapshot


@dataclass
class RetrievalOutcome:
    status: str  # ready, pending, failed
    record: dict[str, Any] | None = None
    snapshot_id: str | None = None
    retrieval_method: str = "brightdata_fetch"
    error: AppError | None = None
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0


@dataclass
class InferenceResponse:
    payload: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)


class ScraperError(Exception):
    def __init__(self, code: str, message: str, *, vendor_details=None, retryable=False):
        self.error = AppError(code=code, message=message, vendor_details=vendor_details, retryable=retryable)
        super().__init__(message)


class Retriever(Protocol):
    async def retrieve(
        self,
        request: ExtractionRequest,
        job: dict[str, Any],
        checkpoint: Callable[[dict[str, Any]], None],
        wait_seconds: float,
    ) -> RetrievalOutcome: ...


class Inference(Protocol):
    provider: str
    model: str

    async def extract(self, snapshot: SourceSnapshot, feedback: str | None = None) -> InferenceResponse: ...
