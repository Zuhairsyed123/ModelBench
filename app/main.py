"""FastAPI Application for ModelBench — LLM Evaluation and Smart Routing Platform.

Exposes REST API endpoints for health monitoring, platform metadata, and
multi-model evaluation benchmarks.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.config import Settings, get_settings
from app.schemas import (
    BenchmarkRequest,
    BenchmarkSummary,
    HealthResponse,
)
from app.services.benchmark import BenchmarkService

# Configure safe logging that avoids leaking sensitive prompts or tokens
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("modelbench.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown routines."""
    settings = get_settings()
    logger.info(
        "Starting %s v%s in %s mode",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title="ModelBench API",
    version="0.1.0",
    description=(
        "Production-minded LLM evaluation and smart routing platform. "
        "Benchmark language models across latency, token efficiency, and estimated costs."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ==============================================================================
# Dependency Providers
# ==============================================================================


def get_service() -> BenchmarkService:
    """Dependency provider for BenchmarkService instance."""
    return BenchmarkService()


# ==============================================================================
# Custom Exception Handlers
# ==============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request payload validation errors with a clean, serialized JSON structure."""
    clean_errors = [
        {
            "loc": [str(item) for item in err.get("loc", [])],
            "msg": str(err.get("msg", "")),
            "type": str(err.get("type", "")),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "The request payload failed schema validation.",
            "details": clean_errors,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected server errors safely without leaking internal stack traces."""
    logger.error(
        "Unhandled internal exception during request: %s", str(exc), exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing the request.",
        },
    )


# ==============================================================================
# API Endpoints
# ==============================================================================


@app.get(
    "/",
    tags=["Root"],
    summary="Platform Overview",
    description="Returns platform overview, API version, and links to documentation.",
)
async def root(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Root metadata endpoint."""
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "description": "ModelBench — LLM Evaluation and Smart Routing Platform",
        "mode": "deterministic_mock",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "health": "GET /health",
            "benchmark": "POST /benchmark",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Service Health Check",
    description="Returns operational status, semantic version, and environment.",
)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Service health check endpoint."""
    return HealthResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        status="healthy",
        environment=settings.app_env,
    )


@app.post(
    "/benchmark",
    response_model=BenchmarkSummary,
    status_code=200,
    tags=["Benchmarking"],
    summary="Run Model Evaluation Benchmark",
    description=(
        "Executes a benchmark evaluation across specified language models. "
        "Operates deterministically in mock mode without requiring paid external API keys."
    ),
)
async def run_benchmark(
    request: BenchmarkRequest,
    service: BenchmarkService = Depends(get_service),
) -> BenchmarkSummary:
    """Execute model benchmark evaluation."""
    # Log benchmark initiation safely without printing full sensitive prompt text
    logger.info(
        "Benchmarking %d model(s) [prompt length: %d chars, delay: %s ms]",
        len(request.models),
        len(request.prompt),
        str(request.mock_delay_ms) if request.mock_delay_ms is not None else "default",
    )

    try:
        summary = await service.run_benchmark(request)
        logger.info(
            "Benchmark completed: %d successful, %d failed",
            summary.metrics.successful_runs,
            summary.metrics.failed_runs,
        )
        return summary
    except Exception as exc:
        logger.error("Failed to execute benchmark: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark execution failed: {str(exc)}",
        )
