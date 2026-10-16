"""Metrics calculation service for ModelBench.

Provides reusable, robust statistical functions for evaluating latency,
estimating token costs, and aggregating benchmark results. Safely handles
empty datasets, missing telemetry, and partial run failures.
"""

import statistics
from typing import Sequence
from app.schemas import BenchmarkMetrics, ExecutionStatus, ModelBenchmarkResult


def calculate_average_latency(latencies: Sequence[float | None]) -> float | None:
    """Calculate the arithmetic mean of a sequence of latencies.

    Args:
        latencies: Sequence of latency values in milliseconds or seconds.
            Can include None or negative values (which will be safely ignored).

    Returns:
        Mean latency rounded to 2 decimal places, or None if no valid values exist.
    """
    valid = [v for v in latencies if v is not None and v >= 0.0]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 2)


def calculate_median_latency(latencies: Sequence[float | None]) -> float | None:
    """Calculate the median of a sequence of latencies.

    Args:
        latencies: Sequence of latency values in milliseconds or seconds.
            Can include None or negative values (which will be safely ignored).

    Returns:
        Median latency rounded to 2 decimal places, or None if no valid values exist.
    """
    valid = [v for v in latencies if v is not None and v >= 0.0]
    if not valid:
        return None
    return round(float(statistics.median(valid)), 2)


def calculate_token_cost(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    """Calculate estimated cost in USD using per-million token rates.

    Args:
        prompt_tokens: Number of prompt/input tokens (treated as 0 if None or negative).
        completion_tokens: Number of completion/output tokens (treated as 0 if None or negative).
        input_cost_per_million: Cost in USD per 1M prompt tokens.
        output_cost_per_million: Cost in USD per 1M completion tokens.

    Returns:
        Estimated monetary cost rounded to 6 decimal places.
    """
    safe_prompt = max(0, prompt_tokens or 0)
    safe_completion = max(0, completion_tokens or 0)
    safe_input_rate = max(0.0, input_cost_per_million)
    safe_output_rate = max(0.0, output_cost_per_million)

    cost_input = (safe_prompt / 1_000_000.0) * safe_input_rate
    cost_output = (safe_completion / 1_000_000.0) * safe_output_rate

    return round(cost_input + cost_output, 6)


def compute_benchmark_metrics(
    results: Sequence[ModelBenchmarkResult],
) -> BenchmarkMetrics:
    """Compute aggregate statistical metrics across a set of model benchmark results.

    Safely handles:
    - Empty result lists
    - 100% failed runs
    - Mixed success and failure runs
    - Missing token counts or missing costs

    Args:
        results: Sequence of ModelBenchmarkResult objects.

    Returns:
        BenchmarkMetrics containing counts, latency statistics, total costs,
        and fastest/cheapest model designations.
    """
    if not results:
        return BenchmarkMetrics(
            total_models=0,
            successful_runs=0,
            failed_runs=0,
            average_latency_ms=None,
            median_latency_ms=None,
            min_latency_ms=None,
            max_latency_ms=None,
            total_cost_usd=None,
            total_tokens=None,
            fastest_model=None,
            cheapest_model=None,
        )

    total_models = len(results)
    successful_runs = [r for r in results if r.status == ExecutionStatus.SUCCESS]
    failed_runs = [r for r in results if r.status == ExecutionStatus.ERROR]

    if not successful_runs:
        return BenchmarkMetrics(
            total_models=total_models,
            successful_runs=0,
            failed_runs=len(failed_runs),
            average_latency_ms=None,
            median_latency_ms=None,
            min_latency_ms=None,
            max_latency_ms=None,
            total_cost_usd=None,
            total_tokens=None,
            fastest_model=None,
            cheapest_model=None,
        )

    # Collect latencies from successful runs
    latencies = [r.latency_ms for r in successful_runs if r.latency_ms is not None]
    avg_latency = calculate_average_latency(latencies)
    med_latency = calculate_median_latency(latencies)
    min_latency = round(min(latencies), 2) if latencies else None
    max_latency = round(max(latencies), 2) if latencies else None

    # Total cost aggregation
    costs = [
        r.estimated_cost_usd
        for r in successful_runs
        if r.estimated_cost_usd is not None
    ]
    total_cost = round(sum(costs), 6) if costs else None

    # Total token aggregation
    tokens = [
        r.token_usage.total_tokens
        for r in successful_runs
        if r.token_usage is not None and r.token_usage.total_tokens is not None
    ]
    total_tokens = sum(tokens) if tokens else None

    # Fastest model (minimum latency_ms among successful runs)
    fastest_model = None
    if latencies:
        fastest_run = min(successful_runs, key=lambda r: r.latency_ms)
        fastest_model = fastest_run.model

    # Cheapest model (minimum estimated_cost_usd among successful runs with valid cost)
    cheapest_model = None
    runs_with_cost = [
        r for r in successful_runs if r.estimated_cost_usd is not None
    ]
    if runs_with_cost:
        cheapest_run = min(runs_with_cost, key=lambda r: r.estimated_cost_usd)  # type: ignore[arg-type]
        cheapest_model = cheapest_run.model

    return BenchmarkMetrics(
        total_models=total_models,
        successful_runs=len(successful_runs),
        failed_runs=len(failed_runs),
        average_latency_ms=avg_latency,
        median_latency_ms=med_latency,
        min_latency_ms=min_latency,
        max_latency_ms=max_latency,
        total_cost_usd=total_cost,
        total_tokens=total_tokens,
        fastest_model=fastest_model,
        cheapest_model=cheapest_model,
    )
