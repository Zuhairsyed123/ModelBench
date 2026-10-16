"""Pydantic data schemas for ModelBench.

Defines validated request payloads, individual model benchmark results,
aggregate metrics, and health response models.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator


class ExecutionStatus(str, Enum):
    """Execution status of an individual model benchmark run."""

    SUCCESS = "success"
    ERROR = "error"


class ProviderType(str, Enum):
    """Identifies the model provider backend."""

    MOCK = "mock"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class TokenUsage(BaseModel):
    """Token consumption breakdown for an LLM response."""

    prompt_tokens: int = Field(default=0, ge=0, description="Tokens in input prompt")
    completion_tokens: int = Field(default=0, ge=0, description="Tokens in completion")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed")

    @classmethod
    def calculate(cls, prompt_tokens: int, completion_tokens: int) -> "TokenUsage":
        """Factory method to calculate total tokens automatically."""
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


class ModelPricingConfig(BaseModel):
    """Configurable pricing rates in USD per million tokens."""

    input_cost_per_million: float = Field(
        default=0.0, ge=0.0, description="Cost in USD per 1M input/prompt tokens"
    )
    output_cost_per_million: float = Field(
        default=0.0, ge=0.0, description="Cost in USD per 1M output/completion tokens"
    )


class BenchmarkRequest(BaseModel):
    """Payload for initiating a model benchmark run."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="Prompt text to benchmark across the selected models",
        examples=["Explain the difference between process and thread in 2 sentences."],
    )
    models: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of model identifiers to evaluate",
        examples=[["mock-gpt-4o", "mock-claude-3-5-sonnet", "mock-gemini-1-5-pro"]],
    )
    system_prompt: str | None = Field(
        default=None,
        max_length=10_000,
        description="Optional system prompt to prepend or guide execution",
    )
    temperature: float | None = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for variability (if applicable)",
    )
    max_tokens: int | None = Field(
        default=512,
        ge=1,
        le=16_384,
        description="Maximum tokens allowed for generation",
    )
    mock_delay_ms: float | None = Field(
        default=None,
        ge=0.0,
        le=5_000.0,
        description="Optional simulation delay in milliseconds for mock providers",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Prompt must not be empty or contain only whitespace.")
        return stripped

    @field_validator("models")
    @classmethod
    def validate_models(cls, v: list[str]) -> list[str]:
        cleaned = [m.strip() for m in v if m and m.strip()]
        if not cleaned:
            raise ValueError("Models list must contain at least one valid model identifier.")
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for m in cleaned:
            if m not in seen:
                seen.add(m)
                deduped.append(m)
        return deduped


class ModelBenchmarkResult(BaseModel):
    """Execution result and telemetry for a single model run."""

    model: str = Field(..., description="Model identifier")
    provider: str = Field(default="mock", description="Provider backend identifier")
    status: ExecutionStatus = Field(..., description="Run status ('success' or 'error')")
    response_text: str | None = Field(
        default=None, description="Generated response content (None on error)"
    )
    latency_seconds: float = Field(
        ..., ge=0.0, description="Elapsed response latency in seconds (monotonic clock)"
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="Elapsed response latency in milliseconds"
    )
    token_usage: TokenUsage | None = Field(
        default=None, description="Token consumption details if available"
    )
    estimated_cost_usd: float | None = Field(
        default=None, ge=0.0, description="Estimated monetary cost in USD"
    )
    error_message: str | None = Field(
        default=None, description="Descriptive error details if execution failed"
    )
    is_mock: bool = Field(
        default=True,
        description="Explicit flag confirming whether result is mock/simulated (never claimed as real)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of benchmark completion",
    )


class BenchmarkMetrics(BaseModel):
    """Statistical summary across all model results in a benchmark run."""

    total_models: int = Field(default=0, ge=0, description="Total models evaluated")
    successful_runs: int = Field(default=0, ge=0, description="Number of successful runs")
    failed_runs: int = Field(default=0, ge=0, description="Number of failed runs")
    average_latency_ms: float | None = Field(
        default=None, description="Mean latency across successful runs in milliseconds"
    )
    median_latency_ms: float | None = Field(
        default=None, description="Median latency across successful runs in milliseconds"
    )
    min_latency_ms: float | None = Field(
        default=None, description="Minimum latency across successful runs in milliseconds"
    )
    max_latency_ms: float | None = Field(
        default=None, description="Maximum latency across successful runs in milliseconds"
    )
    total_cost_usd: float | None = Field(
        default=None, description="Aggregate estimated cost in USD"
    )
    total_tokens: int | None = Field(
        default=None, description="Aggregate tokens consumed across all runs"
    )
    fastest_model: str | None = Field(
        default=None, description="Model identifier with lowest latency"
    )
    cheapest_model: str | None = Field(
        default=None, description="Model identifier with lowest estimated cost"
    )


class BenchmarkSummary(BaseModel):
    """Top-level response payload for a completed benchmark execution."""

    benchmark_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique benchmark execution identifier",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Execution timestamp",
    )
    prompt: str = Field(..., description="Original input prompt evaluated")
    is_mock: bool = Field(
        default=True,
        description="Indicates that results are simulated mocks, not live external provider calls",
    )
    results: list[ModelBenchmarkResult] = Field(
        default_factory=list, description="List of per-model benchmark results"
    )
    metrics: BenchmarkMetrics = Field(
        ..., description="Aggregated latency, cost, and throughput metrics"
    )


class HealthResponse(BaseModel):
    """Application health check response."""

    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application semantic version")
    status: str = Field(default="healthy", description="Application operational status")
    environment: str = Field(..., description="Current deployment environment")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Current server time (UTC)",
    )
