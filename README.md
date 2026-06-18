# InferRoute

Production-grade low latency LLM inference router and observability gateway. It acts as an intelligent intermediary between your client apps/agents and inference models (such as local engines like vLLM and cloud providers like OpenAI).

## Core Features
1. **Multi-Objective Routing**: Scoring based on historical TTFT, expected costs, failure rates, and caching indicators.
2. **Unified API Endpoint**: OpenAI compatible `/v1/chat/completions` supporting both streaming and blocking responses.
3. **Resilient Failovers**: Transparent fallback handling: automatically routes to a secondary cloud backend (e.g. OpenAI) if the local inference fails.
4. **Exact caching**: Redis cache lookups to reduce costs and TTFT to sub-millisecond ranges.
5. **Observability Stack**: Integrated OTel tracing, Prometheus metrics export, and PostgreSQL audit logger.

---

## Getting Started

### 1. Requirements
- Python 3.12+
- Docker & Docker Compose

### 2. Configure Environment
Copy `.env.example` to `.env` and set your API keys:
```bash
cp .env.example .env
```

### 3. Spin up Container Infrastructure
Start PostgreSQL, Redis, OTel Collector, Jaeger, Prometheus, and Grafana:
```bash
docker compose up -d
```

### 4. Setup Local Python Virtualenv & Run Gateway
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API Gateway
uvicorn inferroute.main:app --host 0.0.0.0 --port 8080 --reload
```

---

## Testing the Gateway

You can make requests to `/v1/chat/completions` using the standard OpenAI payload:

### Blocking request (JSON schema validation)
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-inferroute-demo" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "edge/auto",
    "messages": [
      {"role": "user", "content": "Generate invoice data as JSON for transaction 500."}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "invoice",
        "schema": {
          "type": "object",
          "properties": {
            "invoice_id": {"type": "string"},
            "amount": {"type": "number"}
          },
          "required": ["invoice_id", "amount"]
        }
      }
    }
  }'
```

### Streaming request
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-inferroute-demo" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "edge/auto",
    "stream": true,
    "messages": [
      {"role": "user", "content": "How does LLM caching optimize TTFT?"}
    ]
  }'
```

---

## Observability Dashboards
- **Jaeger UI (Traces)**: [http://localhost:16686](http://localhost:16686)
- **Prometheus (Metrics)**: [http://localhost:9090](http://localhost:9090)
- **Grafana (Dashboards)**: [http://localhost:3000](http://localhost:3000) (Default user/pass: `admin`/`admin`)
