# ModelBench — LLM Evaluation and Smart Routing Platform

A production-minded, modular framework for evaluating, benchmarking, and routing large language model (LLM) requests. ModelBench measures and tracks response latency, token consumption, estimated monetary costs, and response-quality metrics across models.

---

## Important: Mock vs. Real Benchmarks

> **Zero-Cost Local Development:** ModelBench currently runs in **deterministic mock mode** (`is_mock: true`).
>
> - **Synthetic Outputs:** Responses and token counts are deterministically synthesized based on model identifiers and prompt text.
> - **Monotonic Timing:** Latency measurements represent genuine elapsed time measured using Python's high-resolution monotonic clock (`time.perf_counter()`).
> - **No Paid Credentials Needed:** You do **not** need an OpenAI, Anthropic, or any third-party API key to run, test, or evaluate this platform.
> - **Clear Labeling:** All mock outputs are explicitly prefixed with `[MOCK RESPONSE - Not a real model output]` and flagged with `is_mock=True`. No mock data is presented as real model outputs.
> - **Extensible Architecture:** The core `BaseProvider` interface separates provider integrations from benchmarking workflows, ensuring that live provider connectors (e.g., OpenAI, Anthropic, Ollama) can be plugged in during subsequent stages without modifying orchestration or metrics logic.

---

## Project Structure

```
ModelBench/
├── app/
│   ├── __init__.py           # Application package metadata and version
│   ├── main.py               # FastAPI application with REST endpoints & lifespan
│   ├── config.py             # Pydantic Settings reading environment variables
│   ├── schemas.py            # Validated Pydantic models (requests, results, metrics)
│   └── services/
│       ├── __init__.py       # Services package marker
│       ├── benchmark.py      # BaseProvider, MockProvider, and BenchmarkService
│       └── metrics.py        # Latency statistics, cost formulas, and aggregations
├── tests/
│   ├── test_health.py        # Integration tests for /health and / endpoints
│   ├── test_metrics.py       # Unit tests for latency, cost, and edge cases
│   └── test_benchmark.py     # Unit and integration tests for benchmarking service & API
├── .env.example              # Template environment configuration (safe defaults)
├── .gitignore                # Excludes secrets, caches, virtual environments, logs
├── pytest.ini                # Pytest configuration with test paths and pythonpath
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation and usage guide
```

---

## Features

- **FastAPI Backend:** Fully asynchronous REST API with automatic OpenAPI documentation at `/docs` and `/redoc`.
- **Strict Data Validation:** Type-safe Pydantic v2 schemas validating prompt constraints, model identifier lists, token usage, and pricing.
- **Concurrent Benchmarking:** Orchestrates parallel model evaluations via `asyncio.gather` for low-overhead throughput.
- **High-Resolution Timing:** Monotonic clock measurement (`time.perf_counter()`) for accurate latency metrics in seconds and milliseconds.
- **Reference Cost Modeling:** Configurable per-million input/output token pricing calculations with safe fallbacks and zero-division protection.
- **Statistical Aggregation:** Computes mean, median, min, and max latencies, aggregate token usage, total estimated costs, and designates fastest/cheapest models.
- **Resilience and Error Isolation:** Per-model error handling ensures that single-model failures do not abort the entire benchmark run.
- **Comprehensive Test Suite:** 28 unit and integration tests covering standard operations, empty payloads, and failure scenarios.

---

## Installation & Setup

### Prerequisites

- Python 3.10+ (tested with Python 3.14)
- Git

### 1. Clone & Enter Directory

```bash
git clone <repository-url>
cd ModelBench
```

### 2. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example configuration:
```bash
cp .env.example .env
```

The application runs with safe defaults out of the box. Key settings in `.env`:
| Variable | Description | Default |
| :--- | :--- | :--- |
| `MODELBENCH_APP_NAME` | Application name | `"ModelBench"` |
| `MODELBENCH_APP_ENV` | Environment (`development`, `production`) | `"development"` |
| `MODELBENCH_API_HOST` | Host address | `"0.0.0.0"` |
| `MODELBENCH_API_PORT` | Port number | `8000` |
| `MODELBENCH_DEFAULT_MOCK_DELAY_MS` | Simulated mock latency in ms | `120.0` |

---

## Running the API

Start the local development server with live reload:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Once running:
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## API Reference & Examples

### 1. Health Check

**Request:**
```bash
curl -X GET http://localhost:8000/health
```

**Response (`200 OK`):**
```json
{
  "app_name": "ModelBench",
  "version": "0.1.0",
  "status": "healthy",
  "environment": "development",
  "timestamp": "2026-10-16T15:00:00.000000Z"
}
```

---

### 2. Platform Overview

**Request:**
```bash
curl -X GET http://localhost:8000/
```

**Response (`200 OK`):**
```json
{
  "app_name": "ModelBench",
  "version": "0.1.0",
  "environment": "development",
  "description": "ModelBench — LLM Evaluation and Smart Routing Platform",
  "mode": "deterministic_mock",
  "documentation": "/docs",
  "health_check": "/health",
  "endpoints": {
    "health": "GET /health",
    "benchmark": "POST /benchmark"
  }
}
```

---

### 3. Run Benchmark

**Request (`curl`):**
```bash
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the difference between a process and a thread.",
    "models": [
      "mock-gpt-4o",
      "mock-claude-3-5-sonnet",
      "mock-gemini-1-5-flash"
    ],
    "mock_delay_ms": 50.0
  }'
```

**Python Example (`httpx`):**
```python
import httpx

payload = {
    "prompt": "Explain the difference between a process and a thread.",
    "models": ["mock-gpt-4o", "mock-claude-3-5-sonnet", "mock-gemini-1-5-flash"],
    "mock_delay_ms": 50.0,
}

response = httpx.post("http://localhost:8000/benchmark", json=payload)
data = response.json()

print(f"Benchmark ID: {data['benchmark_id']}")
print(f"Fastest Model: {data['metrics']['fastest_model']}")
print(f"Cheapest Model: {data['metrics']['cheapest_model']}")
print(f"Average Latency: {data['metrics']['average_latency_ms']} ms")
```

**Sample Response (`200 OK`):**
```json
{
  "benchmark_id": "7b8efb49-b5f7-4aa7-b50a-b15a6b09dc99",
  "created_at": "2026-10-16T15:00:00.000000Z",
  "prompt": "Explain the difference between a process and a thread.",
  "is_mock": true,
  "results": [
    {
      "model": "mock-gpt-4o",
      "provider": "mock",
      "status": "success",
      "response_text": "[MOCK RESPONSE - Not a real model output]\nModel: mock-gpt-4o\n...",
      "latency_seconds": 0.0521,
      "latency_ms": 52.1,
      "token_usage": {
        "prompt_tokens": 14,
        "completion_tokens": 58,
        "total_tokens": 72
      },
      "estimated_cost_usd": 0.000615,
      "error_message": null,
      "is_mock": true,
      "timestamp": "2026-10-16T15:00:00.052000Z"
    }
  ],
  "metrics": {
    "total_models": 3,
    "successful_runs": 3,
    "failed_runs": 0,
    "average_latency_ms": 53.4,
    "median_latency_ms": 52.8,
    "min_latency_ms": 51.2,
    "max_latency_ms": 56.2,
    "total_cost_usd": 0.00184,
    "total_tokens": 216,
    "fastest_model": "mock-gemini-1-5-flash",
    "cheapest_model": "mock-gemini-1-5-flash"
  }
}
```

---

## Running Tests

Execute the full automated test suite using `pytest`:

```bash
pytest -v
```

Run specific test modules:
```bash
pytest tests/test_metrics.py -v
pytest tests/test_benchmark.py -v
pytest tests/test_health.py -v
```

---

## Next Stages & Roadmap

1. **Live Provider Integrations:** Add implementations of `BaseProvider` for OpenAI, Anthropic, Google Gemini, and local Ollama instances.
2. **Intelligent Smart Router:** Route prompts dynamically according to cost constraints, latency SLA, or prompt complexity.
3. **Interactive Streamlit Dashboard:** Visual UI for executing benchmarks, comparing response diffs side-by-side, and rendering latency/cost charts.
4. **LLM-as-a-Judge Evaluation:** Integrate automated response scoring for coherence, factual accuracy, and instruction following.
