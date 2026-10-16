"""Unit tests for ModelBench metrics service.

Tests statistical latency calculations, token cost estimation, and aggregate
metrics computation across normal, empty, and edge cases.
"""

import pytest
from app.schemas import ExecutionStatus, ModelBenchmarkResult, TokenUsage
from app.services.metrics import (
    calculate_average_latency,
    calculate_median_latency,
    calculate_token_cost,
    compute_benchmark_metrics,
)


# ==============================================================================
# Average Latency Tests
# ==============================================================================


def test_calculate_average_latency_normal():
    """Verify standard arithmetic mean calculation with valid values."""
    latencies = [100.0, 200.0, 300.0]
    assert calculate_average_latency(latencies) == 200.0


def test_calculate_average_latency_empty():
    """Verify empty list returns None safely without raising ZeroDivisionError."""
    assert calculate_average_latency([]) is None


def test_calculate_average_latency_handles_none_and_negatives():
    """Verify None and negative values are safely ignored."""
    latencies = [100.0, None, -50.0, 200.0]
    assert calculate_average_latency(latencies) == 150.0


def test_calculate_average_latency_all_invalid():
    """Verify list with only invalid values returns None."""
    latencies = [None, -10.0]
    assert calculate_average_latency(latencies) == None


# ==============================================================================
# Median Latency Tests
# ==============================================================================


def test_calculate_median_latency_odd():
    """Verify median calculation for an odd number of items."""
    latencies = [300.0, 100.0, 200.0]
    assert calculate_median_latency(latencies) == 200.0


def test_calculate_median_latency_even():
    """Verify median calculation for an even number of items."""
    latencies = [100.0, 200.0, 300.0, 400.0]
    assert calculate_median_latency(latencies) == 250.0


def test_calculate_median_latency_single():
    """Verify median calculation for a single element."""
    assert calculate_median_latency([125.45]) == 125.45


def test_calculate_median_latency_empty():
    """Verify median calculation returns None on empty data."""
    assert calculate_median_latency([]) is None


def test_calculate_median_latency_handles_none_and_negatives():
    """Verify median ignores None and negative values."""
    latencies = [100.0, None, 500.0, -10.0, 300.0]
    # Valid sorted: [100.0, 300.0, 500.0] -> median is 300.0
    assert calculate_median_latency(latencies) == 300.0


# ==============================================================================
# Token Cost Tests
# ==============================================================================


def test_calculate_token_cost_normal():
    """Verify cost calculation with standard token counts and rates."""
    # 1,000 prompt tokens @ $2.50/M = $0.0025
    # 500 completion tokens @ $10.00/M = $0.0050
    # Total = $0.0075
    cost = calculate_token_cost(
        prompt_tokens=1000,
        completion_tokens=500,
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    )
    assert cost == 0.0075


def test_calculate_token_cost_zero_tokens():
    """Verify zero token usage results in 0.0 cost."""
    cost = calculate_token_cost(
        prompt_tokens=0,
        completion_tokens=0,
        input_cost_per_million=5.00,
        output_cost_per_million=15.00,
    )
    assert cost == 0.0


def test_calculate_token_cost_none_tokens():
    """Verify None token values are treated safely as 0."""
    cost = calculate_token_cost(
        prompt_tokens=None,
        completion_tokens=None,
        input_cost_per_million=5.00,
        output_cost_per_million=15.00,
    )
    assert cost == 0.0


def test_calculate_token_cost_custom_rates():
    """Verify calculation respects custom input and output rates."""
    # 2,000,000 prompt tokens @ $1.50/M = $3.00
    # 1,000,000 completion tokens @ $3.00/M = $3.00
    # Total = $6.00
    cost = calculate_token_cost(
        prompt_tokens=2_000_000,
        completion_tokens=1_000_000,
        input_cost_per_million=1.50,
        output_cost_per_million=3.00,
    )
    assert cost == 6.0


# ==============================================================================
# Aggregate Benchmark Metrics Tests
# ==============================================================================


def test_compute_benchmark_metrics_empty():
    """Verify metric computation on empty results returns safe default object."""
    metrics = compute_benchmark_metrics([])
    assert metrics.total_models == 0
    assert metrics.successful_runs == 0
    assert metrics.failed_runs == 0
    assert metrics.average_latency_ms is None
    assert metrics.median_latency_ms is None
    assert metrics.total_cost_usd is None
    assert metrics.total_tokens is None
    assert metrics.fastest_model is None
    assert metrics.cheapest_model is None


def test_compute_benchmark_metrics_all_success():
    """Verify metrics summary when all runs succeed."""
    r1 = ModelBenchmarkResult(
        model="mock-gpt-4o",
        provider="mock",
        status=ExecutionStatus.SUCCESS,
        response_text="Fast response",
        latency_seconds=0.1,
        latency_ms=100.0,
        token_usage=TokenUsage.calculate(100, 50),
        estimated_cost_usd=0.001,
        is_mock=True,
    )
    r2 = ModelBenchmarkResult(
        model="mock-claude-3-5-sonnet",
        provider="mock",
        status=ExecutionStatus.SUCCESS,
        response_text="Detailed response",
        latency_seconds=0.3,
        latency_ms=300.0,
        token_usage=TokenUsage.calculate(200, 100),
        estimated_cost_usd=0.003,
        is_mock=True,
    )

    metrics = compute_benchmark_metrics([r1, r2])
    assert metrics.total_models == 2
    assert metrics.successful_runs == 2
    assert metrics.failed_runs == 0
    assert metrics.average_latency_ms == 200.0
    assert metrics.median_latency_ms == 200.0
    assert metrics.min_latency_ms == 100.0
    assert metrics.max_latency_ms == 300.0
    assert metrics.total_cost_usd == 0.004
    assert metrics.total_tokens == 450
    assert metrics.fastest_model == "mock-gpt-4o"
    assert metrics.cheapest_model == "mock-gpt-4o"


def test_compute_benchmark_metrics_all_failed():
    """Verify metrics summary when all runs fail."""
    r1 = ModelBenchmarkResult(
        model="mock-fail-1",
        provider="mock",
        status=ExecutionStatus.ERROR,
        response_text=None,
        latency_seconds=0.05,
        latency_ms=50.0,
        error_message="Network timeout",
        is_mock=True,
    )
    r2 = ModelBenchmarkResult(
        model="mock-fail-2",
        provider="mock",
        status=ExecutionStatus.ERROR,
        response_text=None,
        latency_seconds=0.06,
        latency_ms=60.0,
        error_message="Rate limit exceeded",
        is_mock=True,
    )

    metrics = compute_benchmark_metrics([r1, r2])
    assert metrics.total_models == 2
    assert metrics.successful_runs == 0
    assert metrics.failed_runs == 2
    assert metrics.average_latency_ms is None
    assert metrics.median_latency_ms is None
    assert metrics.fastest_model is None
    assert metrics.cheapest_model is None


def test_compute_benchmark_metrics_mixed():
    """Verify metrics summary when some runs succeed and others fail."""
    success_run = ModelBenchmarkResult(
        model="mock-gpt-4o",
        provider="mock",
        status=ExecutionStatus.SUCCESS,
        response_text="Good output",
        latency_seconds=0.15,
        latency_ms=150.0,
        token_usage=TokenUsage.calculate(80, 40),
        estimated_cost_usd=0.0005,
        is_mock=True,
    )
    failed_run = ModelBenchmarkResult(
        model="mock-error-model",
        provider="mock",
        status=ExecutionStatus.ERROR,
        response_text=None,
        latency_seconds=0.02,
        latency_ms=20.0,
        error_message="Simulated failure",
        is_mock=True,
    )

    metrics = compute_benchmark_metrics([success_run, failed_run])
    assert metrics.total_models == 2
    assert metrics.successful_runs == 1
    assert metrics.failed_runs == 1
    assert metrics.average_latency_ms == 150.0
    assert metrics.median_latency_ms == 150.0
    assert metrics.total_cost_usd == 0.0005
    assert metrics.fastest_model == "mock-gpt-4o"
    assert metrics.cheapest_model == "mock-gpt-4o"


def test_compute_benchmark_metrics_missing_tokens_and_cost():
    """Verify metrics calculation when telemetry fields are missing/None."""
    run = ModelBenchmarkResult(
        model="mock-custom",
        provider="mock",
        status=ExecutionStatus.SUCCESS,
        response_text="No telemetry provided",
        latency_seconds=0.2,
        latency_ms=200.0,
        token_usage=None,
        estimated_cost_usd=None,
        is_mock=True,
    )

    metrics = compute_benchmark_metrics([run])
    assert metrics.successful_runs == 1
    assert metrics.total_tokens is None
    assert metrics.total_cost_usd is None
    assert metrics.cheapest_model is None
    assert metrics.fastest_model == "mock-custom"
