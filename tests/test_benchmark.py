"""Integration and unit tests for ModelBench benchmark service and API endpoints.

Validates deterministic mock generation, monotonic timing, error resilience,
and API request/response contracts.
"""

import asyncio
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import BenchmarkRequest, ExecutionStatus
from app.services.benchmark import BenchmarkService, MockProvider

client = TestClient(app)


# ==============================================================================
# Mock Provider & Service Unit Tests
# ==============================================================================


def test_mock_provider_deterministic_generation():
    """Verify that the mock provider produces consistent output for the same input."""
    async def _test():
        provider = MockProvider()
        res1 = await provider.generate(model="mock-gpt-4o", prompt="What is an API?", mock_delay_ms=0)
        res2 = await provider.generate(model="mock-gpt-4o", prompt="What is an API?", mock_delay_ms=0)

        assert res1.status == ExecutionStatus.SUCCESS
        assert res1.is_mock is True
        assert res1.response_text is not None
        assert "[MOCK RESPONSE" in res1.response_text
        # Deterministic generation ensures repeatable text and tokens
        assert res1.response_text == res2.response_text
        assert res1.token_usage.total_tokens == res2.token_usage.total_tokens

    asyncio.run(_test())


def test_mock_provider_monotonic_timing():
    """Verify that elapsed latency reflects genuine monotonic measurement."""
    async def _test():
        provider = MockProvider()
        delay_ms = 40.0
        res = await provider.generate(model="mock-gpt-4o-mini", prompt="Test prompt", mock_delay_ms=delay_ms)

        assert res.status == ExecutionStatus.SUCCESS
        assert res.latency_ms >= 30.0  # Monotonic clock should measure at least the simulated delay
        assert res.latency_seconds > 0.0

    asyncio.run(_test())


def test_mock_provider_error_handling():
    """Verify that simulated error models return structured ERROR status without crashing."""
    async def _test():
        provider = MockProvider()
        res = await provider.generate(model="mock-error-timeout", prompt="Test prompt", mock_delay_ms=0)

        assert res.status == ExecutionStatus.ERROR
        assert res.response_text is None
        assert res.error_message is not None
        assert "Simulated execution failure" in res.error_message
        assert res.is_mock is True

    asyncio.run(_test())


def test_benchmark_service_multi_model():
    """Verify benchmark service evaluates multiple models and returns aggregated summary."""
    async def _test():
        service = BenchmarkService()
        request = BenchmarkRequest(
            prompt="Explain recursion in programming.",
            models=["mock-gpt-4o", "mock-claude-3-5-sonnet", "mock-gemini-1-5-pro"],
            mock_delay_ms=10.0,
        )

        summary = await service.run_benchmark(request)

        assert summary.prompt == request.prompt
        assert summary.is_mock is True
        assert len(summary.results) == 3
        assert summary.metrics.total_models == 3
        assert summary.metrics.successful_runs == 3
        assert summary.metrics.failed_runs == 0
        assert summary.metrics.average_latency_ms is not None
        assert summary.metrics.total_cost_usd is not None
        assert summary.metrics.fastest_model in ["mock-gpt-4o", "mock-claude-3-5-sonnet", "mock-gemini-1-5-pro"]

    asyncio.run(_test())


# ==============================================================================
# API Endpoint Integration Tests
# ==============================================================================


def test_api_benchmark_success():
    """Verify POST /benchmark succeeds with valid payload and returns structured summary."""
    payload = {
        "prompt": "Write a Python function to compute Fibonacci numbers.",
        "models": ["mock-gpt-4o", "mock-llama-3-3-70b"],
        "mock_delay_ms": 10.0,
    }
    response = client.post("/benchmark", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "benchmark_id" in data
    assert data["prompt"] == payload["prompt"]
    assert data["is_mock"] is True
    assert len(data["results"]) == 2

    # Check individual model result
    first_result = data["results"][0]
    assert first_result["is_mock"] is True
    assert first_result["status"] == "success"
    assert first_result["latency_ms"] > 0
    assert first_result["token_usage"]["total_tokens"] > 0
    assert first_result["estimated_cost_usd"] is not None

    # Check aggregated metrics
    metrics = data["metrics"]
    assert metrics["total_models"] == 2
    assert metrics["successful_runs"] == 2
    assert metrics["failed_runs"] == 0
    assert metrics["average_latency_ms"] is not None


def test_api_benchmark_validation_empty_prompt():
    """Verify POST /benchmark returns 422 when prompt is empty or blank."""
    payload = {
        "prompt": "   ",
        "models": ["mock-gpt-4o"],
    }
    response = client.post("/benchmark", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
    assert len(data["details"]) > 0


def test_api_benchmark_validation_empty_models():
    """Verify POST /benchmark returns 422 when models list is empty."""
    payload = {
        "prompt": "Valid prompt text",
        "models": [],
    }
    response = client.post("/benchmark", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
    assert len(data["details"]) > 0


def test_api_benchmark_mixed_success_and_error():
    """Verify POST /benchmark handles partial failure gracefully with 200 status and metric counts."""
    payload = {
        "prompt": "Test query for resilience.",
        "models": ["mock-gpt-4o", "mock-error-service-down"],
        "mock_delay_ms": 5.0,
    }
    response = client.post("/benchmark", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["metrics"]["total_models"] == 2
    assert data["metrics"]["successful_runs"] == 1
    assert data["metrics"]["failed_runs"] == 1

    # Verify the failed run structure
    failed_result = next(r for r in data["results"] if r["status"] == "error")
    assert failed_result["model"] == "mock-error-service-down"
    assert failed_result["response_text"] is None
    assert failed_result["error_message"] is not None
    assert failed_result["is_mock"] is True
