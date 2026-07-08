# 📊 InferRoute Router Benchmark Evaluation Summary

| Router Scenario | Avg Cost ($ USD) | Avg Quality Score (0-1) | Avg Latency (ms) | Avg TTFT (ms) | SLO Compliance (%) | Fallback Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **always-openai** | $0.000044 | 0.12 | 250.0ms | 250.0ms | 100.0% | 0.0% |
| **always-gemini** | $0.000026 | 0.05 | 250.0ms | 250.0ms | 100.0% | 0.0% |
| **always-vllm** | $0.000001 | 0.06 | 300.0ms | 150.0ms | 100.0% | 0.0% |
| **cheapest-first** | $0.000000 | 0.12 | 180.0ms | 180.0ms | 100.0% | 0.0% |
| **fastest-first** | $0.000000 | 0.12 | 180.0ms | 180.0ms | 100.0% | 0.0% |
| **heuristic-reliability** | $0.000000 | 0.12 | 180.0ms | 180.0ms | 100.0% | 0.0% |
| **learned-router** | $0.000031 | 0.12 | 266.7ms | 216.7ms | 100.0% | 0.0% |
| **cascade-router** | $0.000045 | 0.12 | 550.0ms | 150.0ms | 100.0% | 100.0% |