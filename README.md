# 🎯 InferRoute: High-Availability LLM Inference Gateway & Observability Router

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-emerald.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-v0.111.0-darkviolet.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/PRs-Welcome-goldenrod.svg" alt="PRs Welcome">
</p>

> 🌐 **Interactive Academic Hub & Cascade Simulator**: **[https://ypeng12.github.io/InferRoute/](https://ypeng12.github.io/InferRoute/)**  
> Go here to explore detailed mathematical derivations, request lifecycle sequence charts, paper downloads, and run the sequential cascade simulation.

---

## 📖 Overview

**InferRoute** is a lightweight, production-grade LLM inference gateway and reliability proxy. Sitting between your client applications/AI agents and model backends (local Ollama/vLLM instances and cloud APIs like OpenAI/Gemini), it dynamically routes queries to optimize for **cost, latency, and quality** in real-time.

By incorporating optimization techniques from Stanford's **FrugalGPT** and Martian's **RouterBench**, InferRoute helps developers reduce API operational costs by over **60%** while respecting latency and formatting SLAs.

---

## ⚡ Key Features

* **🔄 Streaming Request Coalescing**: Uses a Redis-backed lock to merge concurrent duplicate prompts. Only the first request calls the upstream LLM, broadcasting stream token chunks to all callers via Redis Pub/Sub, preventing cache stampedes.
* **🌳 Radix Trie prefix KV-Cache Affinity**: Hashes and stores prompt system instruction prefixes in a Prefix Tree. Directs requests to local GPU backends holding warm KV caches, reducing TTFT (Time-to-First-Token) by up to 80%.
* **🛡️ Vegas Adaptive Concurrency Control**: Tracks active RTT queues dynamically using a TCP Vegas congestion control loop to protect local GPUs from OOM failures under sudden traffic spikes.
* **🔀 Speculative Fallback Cascades**: Buffers stream output from cheap local endpoints, running real-time loop and validation checks. Automatically and transparently escalates requests to premium cloud endpoints mid-stream if quality checks fail.
* **🧠 Learned Predictive Routers**: Supports content-aware KNN and MLP scoring models, optimizing the cost-quality trade-off dynamically.
* **💳 Multi-Tenant Billing Gateway**: Validates tenant keys and credit balances against an asynchronous PostgreSQL audit ledger, enforcing a resilient fail-open policy.

---

## 🚀 Quick Start (Simulation Mode)

You can run InferRoute completely offline with zero API keys using our built-in mock simulation mode:

### 1. Installation
```bash
git clone https://github.com/ypeng12/InferRoute.git
cd InferRoute

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root:
```env
DATABASE_URL=sqlite+aiosqlite:///inferroute.db
MOCK_OPENAI=true
MOCK_GEMINI=true
MOCK_VLLM=true
MOCK_OLLAMA=true
```

### 3. Start Gateway
```bash
python -m uvicorn inferroute.main:app --host 127.0.0.1 --port 8080 --reload
```
Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** to interact with the Live Playground and Chaos panel.

---

## 🛠️ Integration Example (OpenAI Client)

InferRoute exposes a fully OpenAI-compatible chat completions interface. Switch your existing application client in one line:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-inferroute-demo"  # Tenant API key
)

response = client.chat.completions.create(
    model="edge/auto",  # Dynamic auto-routing policy
    messages=[{"role": "user", "content": "Write a quicksort in Python."}],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

---

## 📚 Technical Documentation

For deeper engineering writeups, check our documentation guides:
* **[Academic Foundations & Theory](docs/academic_foundations.md)**: Stanford FrugalGPT and RouterBench cost-quality derivations.
* **[Performance & Cost Benchmarks](docs/benchmark.md)**: Pareto Sweeps, latency comparisons, and metrics.
* **[Gateway Request Lifecycle](docs/architecture.md)**: Interceptors, concurrency queues, and fallback sequence diagrams.
* **[Resilience & Failure Injection](docs/failure-injection.md)**: Self-healing circuit breakers and Vegas metrics.

---

## 📊 Containerized Production Stack

In production, run the Docker Compose stack to enable the full observability pipeline:
```bash
docker compose up -d
```
* **Grafana Dashboard (Performance & Latency)**: [http://localhost:3000](http://localhost:3000) (Admin/Admin)
* **Jaeger UI (Microservice Trace Tracking)**: [http://localhost:16686](http://localhost:16686)

---

## 🛡️ License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
