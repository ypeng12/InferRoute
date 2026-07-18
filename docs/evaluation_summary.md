# 📊 InferRoute RouterBench & FrugalGPT Evaluation Summary

Inspired by the RouterBench framework (`withmartian/routerbench`) and FrugalGPT cascading LLMs, this report evaluates routing policies on cost, quality, and SLA compliance. We plot the Pareto curves by sweeping the willingness-to-pay ($\lambda$), mixture ratio ($p$), and cascade threshold ($\tau$).

## 📈 Curve Efficiency: AIQ (Area Under the Trade-off Curve)
AIQ measures the average quality efficiency score of a router across its swept cost range (normalized AUC, bounded between 0% and 100%). Higher is better.

| Routing Curve | AIQ Score (Normalized AUC) | Description |
| :--- | :--- | :--- |
| **Oracle Router Upper Bound** | *Theoretical Optimal* | Represents the perfect offline selection. |
| **Cascade Router (FrugalGPT)** | 61.1% | Server-side cascading model escalation. |
| **KNN Router** | 66.0% | Jaccard similarity nearest-neighbor routing. |
| **MLP Router** | 64.8% | Content-aware classifier routing. |
| **Zero Router Baseline** | 64.5% | Non-content-aware random model mixture. |

## 📋 Comprehensive Performance Table
| Scenario | Avg Cost ($ USD) | Avg Quality (0-1) | Avg Latency (ms) | Avg TTFT (ms) | SLO Compliance (%) | Fallback Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **always-gemini** | $0.000026 | 0.78 | 250.0ms | 250.0ms | 100.0% | 0.0% |
| **always-ollama** | $0.000000 | 0.31 | 190.0ms | 177.5ms | 100.0% | 8.3% |
| **always-openai** | $0.000044 | 0.75 | 250.0ms | 250.0ms | 100.0% | 0.0% |
| **always-vllm** | $0.000001 | 0.52 | 300.0ms | 150.0ms | 100.0% | 0.0% |
| **cascade-router_t0.00** | $0.000000 | 0.27 | 180.0ms | 180.0ms | 100.0% | 0.0% |
| **cascade-router_t0.20** | $0.000007 | 0.53 | 237.5ms | 187.5ms | 100.0% | 58.3% |
| **cascade-router_t0.40** | $0.000010 | 0.58 | 243.3ms | 193.3ms | 100.0% | 66.7% |
| **cascade-router_t0.60** | $0.000017 | 0.62 | 239.2ms | 201.7ms | 100.0% | 66.7% |
| **cascade-router_t0.80** | $0.000026 | 0.78 | 252.5ms | 227.5ms | 100.0% | 91.7% |
| **cascade-router_t1.00** | $0.000031 | 0.78 | 248.3ms | 235.8ms | 100.0% | 91.7% |
| **knn-router_l0.00** | $0.000000 | 0.31 | 190.0ms | 177.5ms | 100.0% | 8.3% |
| **knn-router_l0.25** | $0.000007 | 0.60 | 277.5ms | 177.5ms | 100.0% | 0.0% |
| **knn-router_l0.50** | $0.000012 | 0.67 | 279.2ms | 191.7ms | 100.0% | 0.0% |
| **knn-router_l1.00** | $0.000019 | 0.75 | 266.7ms | 216.7ms | 100.0% | 0.0% |
| **knn-router_l2.00** | $0.000026 | 0.75 | 258.3ms | 233.3ms | 100.0% | 0.0% |
| **knn-router_l5.00** | $0.000033 | 0.75 | 250.0ms | 250.0ms | 100.0% | 0.0% |
| **mlp-router_l0.00** | $0.000000 | 0.31 | 190.0ms | 177.5ms | 100.0% | 8.3% |
| **mlp-router_l0.25** | $0.000000 | 0.41 | 220.0ms | 170.0ms | 100.0% | 0.0% |
| **mlp-router_l0.50** | $0.000023 | 0.75 | 258.3ms | 233.3ms | 100.0% | 0.0% |
| **mlp-router_l1.00** | $0.000026 | 0.78 | 250.0ms | 250.0ms | 100.0% | 0.0% |
| **mlp-router_l2.00** | $0.000026 | 0.78 | 250.0ms | 250.0ms | 100.0% | 0.0% |
| **mlp-router_l5.00** | $0.000037 | 0.75 | 250.0ms | 250.0ms | 100.0% | 0.0% |
| **oracle-router** | $0.000022 | 0.78 | 258.3ms | 233.3ms | 100.0% | 0.0% |
| **rule-router** | $0.000013 | 0.54 | 243.3ms | 193.3ms | 100.0% | 0.0% |
| **zero-router_p0.0** | $0.000001 | 0.52 | 300.0ms | 150.0ms | 100.0% | 0.0% |
| **zero-router_p0.2** | $0.000001 | 0.52 | 300.0ms | 150.0ms | 100.0% | 0.0% |
| **zero-router_p0.4** | $0.000019 | 0.65 | 279.2ms | 191.7ms | 100.0% | 0.0% |
| **zero-router_p0.6** | $0.000025 | 0.67 | 270.8ms | 208.3ms | 100.0% | 0.0% |
| **zero-router_p0.8** | $0.000032 | 0.68 | 262.5ms | 225.0ms | 100.0% | 0.0% |
| **zero-router_p1.0** | $0.000044 | 0.75 | 250.0ms | 250.0ms | 100.0% | 0.0% |
