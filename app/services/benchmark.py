"""Benchmark service and model provider abstractions for ModelBench.

Provides a clean separation between provider-specific API integrations and
the core benchmarking orchestrator. Includes a deterministic mock provider
for local, zero-cost, reproducible benchmarking without external credentials.
"""

from abc import ABC, abstractmethod
import asyncio
import hashlib
import time
from typing import Any
from app.config import get_settings
from app.schemas import (
    BenchmarkMetrics,
    BenchmarkRequest,
    BenchmarkSummary,
    ExecutionStatus,
    ModelBenchmarkResult,
    TokenUsage,
)

# Reference pricing table (USD per 1M tokens) for realistic mock cost estimation.
# These match publicly published reference rates as of 2025/2026.
MODEL_PRICING_TABLE: dict[str, dict[str, float]] = {
    "mock-gpt-4o": {"input": 2.50, "output": 10.00},
    "mock-gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "mock-claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "mock-claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "mock-gemini-1-5-pro": {"input": 1.25, "output": 5.00},
    "mock-gemini-1-5-flash": {"input": 0.075, "output": 0.30},
    "mock-llama-3-3-70b": {"input": 0.60, "output": 1.80},
    "mock-deepseek-r1": {"input": 0.55, "output": 2.19},
}
DEFAULT_FALLBACK_PRICING = {"input": 1.00, "output": 3.00}


class BaseProvider(ABC):
    """Abstract base class defining the provider execution interface."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique provider identifier string."""
        pass

    @abstractmethod
    async def generate(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = 512,
        mock_delay_ms: float | None = None,
    ) -> ModelBenchmarkResult:
        """Execute a completion request and return benchmark measurements."""
        pass


class MockProvider(BaseProvider):
    """Deterministic mock provider for offline evaluation and testing.

    IMPORTANT:
    This provider generates synthetic completions for testing and architectural
    verification. All results are explicitly marked with `is_mock=True` and
    `provider='mock'`. Elapsed latency is genuinely measured using a high-resolution
    monotonic clock during asynchronous execution.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    def _get_pricing(self, model: str) -> dict[str, float]:
        """Lookup pricing per million tokens for a given model identifier."""
        return MODEL_PRICING_TABLE.get(model, DEFAULT_FALLBACK_PRICING)

    def _estimate_tokens(self, prompt: str, completion: str) -> TokenUsage:
        """Estimate token usage deterministically based on character count heuristics (~4 chars/token)."""
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(completion) // 4)
        return TokenUsage.calculate(prompt_tokens, completion_tokens)

    def _calculate_cost(self, usage: TokenUsage, pricing: dict[str, float]) -> float:
        """Calculate estimated cost in USD based on per-million rates."""
        input_cost = (usage.prompt_tokens / 1_000_000.0) * pricing["input"]
        output_cost = (usage.completion_tokens / 1_000_000.0) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def _generate_mock_text(self, model: str, prompt: str) -> str:
        """Produce a deterministic, clearly labeled mock response."""
        # Use SHA-256 hash to ensure repeatable, deterministic outputs
        digest = hashlib.sha256(f"{model}:{prompt}".encode("utf-8")).hexdigest()
        short_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt

        return (
            f"[MOCK RESPONSE - Not a real model output]\n"
            f"Model: {model}\n"
            f"Prompt Evaluated: \"{short_prompt}\"\n"
            f"Deterministic Seed: {digest[:8]}\n"
            f"Summary: Evaluated prompt successfully with simulated reasoning and response synthesis."
        )

    async def generate(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = 512,
        mock_delay_ms: float | None = None,
    ) -> ModelBenchmarkResult:
        """Execute a mock benchmark run with genuine monotonic clock timing."""
        settings = get_settings()

        # Determine artificial simulation delay (in milliseconds)
        delay_ms = (
            mock_delay_ms
            if mock_delay_ms is not None
            else settings.default_mock_delay_ms
        )

        # Monotonic high-resolution clock start
        start_time = time.perf_counter()

        try:
            # Simulate configurable network/inference latency
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)

            # Support deliberate error simulation for testing and resilience
            lower_model = model.lower()
            if (
                lower_model.startswith("mock-error")
                or ":fail" in lower_model
                or "error" in lower_model
            ):
                raise RuntimeError(
                    f"Simulated execution failure for mock model '{model}'"
                )

            # Synthesize mock response and tokens
            response_text = self._generate_mock_text(model, prompt)
            token_usage = self._estimate_tokens(prompt, response_text)
            pricing = self._get_pricing(model)
            cost_usd = self._calculate_cost(token_usage, pricing)

            elapsed_seconds = time.perf_counter() - start_time
            elapsed_ms = round(elapsed_seconds * 1000.0, 2)

            return ModelBenchmarkResult(
                model=model,
                provider=self.provider_name,
                status=ExecutionStatus.SUCCESS,
                response_text=response_text,
                latency_seconds=round(elapsed_seconds, 4),
                latency_ms=elapsed_ms,
                token_usage=token_usage,
                estimated_cost_usd=cost_usd,
                error_message=None,
                is_mock=True,
            )

        except Exception as exc:
            elapsed_seconds = time.perf_counter() - start_time
            elapsed_ms = round(elapsed_seconds * 1000.0, 2)

            return ModelBenchmarkResult(
                model=model,
                provider=self.provider_name,
                status=ExecutionStatus.ERROR,
                response_text=None,
                latency_seconds=round(elapsed_seconds, 4),
                latency_ms=elapsed_ms,
                token_usage=None,
                estimated_cost_usd=None,
                error_message=str(exc),
                is_mock=True,
            )


class BenchmarkService:
    """Provider-independent benchmarking orchestrator.

    Executes prompt benchmarks concurrently across specified models, measures
    performance, aggregates metrics, and cleanly separates provider implementations
    from benchmarking workflows.
    """

    def __init__(self, default_provider: BaseProvider | None = None) -> None:
        self.mock_provider = default_provider or MockProvider()
        # Registry mapping provider keys to provider instances
        self._providers: dict[str, BaseProvider] = {
            "mock": self.mock_provider,
        }

    def get_provider(self, model: str) -> BaseProvider:
        """Resolve the provider implementation for a given model identifier.

        In the current stage, all models use the deterministic MockProvider.
        Future stages can register real providers (e.g. OpenAI, Anthropic, Ollama).
        """
        return self.mock_provider

    async def benchmark_model(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = 512,
        mock_delay_ms: float | None = None,
    ) -> ModelBenchmarkResult:
        """Benchmark a single model execution with isolated exception handling."""
        provider = self.get_provider(model)
        try:
            return await provider.generate(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                mock_delay_ms=mock_delay_ms,
            )
        except Exception as exc:
            return ModelBenchmarkResult(
                model=model,
                provider=getattr(provider, "provider_name", "unknown"),
                status=ExecutionStatus.ERROR,
                response_text=None,
                latency_seconds=0.0,
                latency_ms=0.0,
                token_usage=None,
                estimated_cost_usd=None,
                error_message=f"Unhandled provider error: {str(exc)}",
                is_mock=True,
            )

    async def run_benchmark(self, request: BenchmarkRequest) -> BenchmarkSummary:
        """Execute concurrent benchmarks across all requested models and compute summary."""
        tasks = [
            self.benchmark_model(
                model=model,
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                mock_delay_ms=request.mock_delay_ms,
            )
            for model in request.models
        ]

        # Concurrently evaluate all models
        results: list[ModelBenchmarkResult] = await asyncio.gather(*tasks)

        # Lazy import to allow modular service layering
        try:
            from app.services.metrics import compute_benchmark_metrics
            metrics = compute_benchmark_metrics(results)
        except ImportError:
            # Fallback basic metric calculation if metrics module is not yet loaded
            successful = [r for r in results if r.status == ExecutionStatus.SUCCESS]
            failed = [r for r in results if r.status == ExecutionStatus.ERROR]
            latencies = [r.latency_ms for r in successful]
            avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else None
            metrics = BenchmarkMetrics(
                total_models=len(results),
                successful_runs=len(successful),
                failed_runs=len(failed),
                average_latency_ms=avg_lat,
            )

        return BenchmarkSummary(
            prompt=request.prompt,
            is_mock=True,
            results=results,
            metrics=metrics,
        )
