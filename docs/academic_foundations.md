# 📖 Academic Foundations: FrugalGPT & RouterBench Integration

InferRoute's core routing architecture is built on the theoretical and mathematical foundations of two landmark papers in LLM cost-performance optimization:

1. **FrugalGPT**: *"FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance"* (Chen et al., Stanford University, 2023).
2. **RouterBench**: *"RouterBench: A Benchmark for Multi-LLM Routing System"* (Li et al., Martian, 2024).

This document outlines the theoretical concepts introduced in these papers and details how they are implemented within the InferRoute gateway codebase.

---

## 🔄 1. FrugalGPT: Cascading Inference & Prompt Adaptation

The primary goal of FrugalGPT is to minimize the cost of querying Large Language Models while maintaining (or even improving) the quality of responses by leveraging heterogeneous APIs. The paper identifies three main pillars of cost-efficiency:

```
                  ┌─────────────────────────────────────────┐
                  │          Incoming Query / Request       │
                  └────────────────────┬────────────────────┘
                                       │
                         [ 1. Prompt Adaptation ]
                  (Trim few-shot examples for cheap backends)
                                       │
                                       ▼
                       [ 2. LLM Approximation (Cache) ]
                  ─────────────► Exact Cache Hit? ────────────► Return Answer
                                       │ (Miss)
                                       ▼
                               [ 3. LLM Cascade ]
                             ┌───────────────────┐
                             │    Cheap Model    │
                             └─────────┬─────────┘
                                       │
                                Reliability Judge
                                       ├──────────────────────┐
                                       │ (Score < τ)          │ (Score ≥ τ)
                                       ▼                      ▼
                             ┌───────────────────┐  ┌───────────────────┐
                             │  Escalated Model  │  │  Accept Response  │
                             └───────────────────┘  └───────────────────┘
```

### 1.1 Prompt Adaptation
* **Theory**: Premium models (e.g., GPT-4o) handle complex prompts with many few-shot examples well, but charging per-token on long prompts is expensive. Cheap models (e.g., Llama-3-8B) cannot leverage long few-shot examples effectively anyway, so sending them full prompts wastes money. Prompt Adaptation dynamically reduces the prompt size (e.g., by pruning examples or vocabulary) for low-cost models.
* **Code Implementation**:
  * Located in [prompt_adapter.py](file:///c:/Users/pengy/OneDrive/Desktop/InferRoute/inferroute/prompt_adapter.py).
  * The method `compress_few_shot_examples` matches few-shot patterns (like `Example 1: ...`, `Q: ... A: ...`) and trims the prompt down to at most 1 example if the target model is cheap (`ollama` or `vllm`).
  * The method `adapt_prompt` automatically decides whether to compress prompt messages based on target backend tiering.

### 1.2 LLM Approximation (Completion Cache)
* **Theory**: Caching and retrieving historical model completions for exact or semantically identical queries avoids model execution fees entirely.
* **Code Implementation**:
  * Located in `inferroute/cache.py` (Redis exact completions cache layer).

### 1.3 LLM Cascade
* **Theory**: Queries are routed sequentially from the cheapest model to the most expensive model. At each step, a **Reliability Judge** checks if the response is correct/acceptable with a confidence threshold $\tau \in [0, 1]$. If accepted, the cascade terminates and returns the response. Otherwise, it escalates to the next model tier.
* **Code Implementation**:
  * **Routing Policy**: Registered as `"cascade"` in [router.py](file:///c:/Users/pengy/OneDrive/Desktop/InferRoute/inferroute/router.py). Generates a cost-sorted list of healthy models (e.g. `ollama` ➔ `vllm` ➔ `gemini` ➔ `openai`) and parses the threshold parameter `acceptance_threshold` ($\tau$).
  * **Blocking Cascade Flow**: Handled by `handle_cascade_blocking_flow` in [main.py](file:///c:/Users/pengy/OneDrive/Desktop/InferRoute/inferroute/main.py). It queries backends sequentially, calls `reliability_scorer.evaluate_reliability`, logs intermediate step scores, and stops upon meeting the threshold, accumulating cumulative costs.
  * **Streaming Cascade Flow**: Handled by `handle_cascade_streaming_flow` in [main.py](file:///c:/Users/pengy/OneDrive/Desktop/InferRoute/inferroute/main.py). To maintain SSE streaming compatibility without leaking garbage output, the gateway buffers all streamed tokens inside an internal generator, scores the full content, and only pumps the SSE stream to the client if quality checks pass.
  * **Reliability Judge**: Implemented as `ReliabilityScorer` in [validator.py](file:///c:/Users/pengy/OneDrive/Desktop/InferRoute/inferroute/validator.py). It performs keyword matching, math digit evaluations, python syntax AST parses, and JSON schema structural validation, alongside repetition loop penalties.

---

## 🧠 2. RouterBench: Predictive Utility & Frontier Evaluation

RouterBench formalizes the multi-LLM routing system as a mathematical optimization framework. It introduces the utility trade-off curve and defines standard routing baselines.

### 2.1 Mathematical Utility Optimization
Choosing which model $m$ should process a prompt $x$ is framed as maximizing a utility score:

$$\text{Score}(m, x) = \lambda \cdot \text{Quality}_{\text{pred}}(m, x) - \text{Cost}(m)$$

Where:
* **$\lambda$ (lambda)**: The user's *willingness to pay* (cost-quality trade-off parameter). A high $\lambda$ (e.g., 5.0) heavily weights quality, guiding routing to cloud models. A low $\lambda$ (e.g., 0.1) weights cost savings, guiding routing to local nodes.
* **$\text{Quality}_{\text{pred}}(m, x)$**: The predicted performance rating of model $m$ on query $x$, scaled between $0.0$ and $1.0$.
* **$\text{Cost}(m)$**: The estimated economic fee of model $m$ (cost per million tokens).

In InferRoute, this is implemented inside the routing policy loops in [router.py](file:///c:/Users/pengy/OneDrive/Desktop/InferRoute/inferroute/router.py).

### 2.2 Standardized Routing Policies
InferRoute implements the exact policy types benchmarked in RouterBench:
1. **Zero Router Baseline (`zero`)**: Randomly assigns a fraction $p$ of queries to premium cloud backends and $1-p$ to local backends. Sweeping $p \in [0, 1]$ forms the random baseline performance curve.
2. **Rule-Based Router (`rule`)**: Inspects query intent (e.g., code snippets, math symbols, prompt length) to direct traffic using heuristic rules.
3. **KNN-Based Router (`knn`)**: Finds historical benchmark queries with high Jaccard similarity. Computes average quality scores per model for those nearest neighbors, and applies the utility formula.
4. **MLP-Based Router (`mlp`)**: Employs a content-aware classifier model. Extracts query features (`is_code`, `is_math`, `is_json`, `is_long`) to predict quality score probabilities, maximizing the utility formula.
5. **Oracle Router Upper Bound (`oracle`)**: A theoretical router with perfect offline knowledge of whether each model will succeed. It selects the cheapest backend that achieves a quality score $\ge 0.8$.

### 2.3 Evaluation Metric: AIQ (Area under the Curve)
* **Theory**: To compare different routers globally rather than at a single budget, RouterBench computes the **AIQ (Area under the cost-quality trade-off curve)**. A higher AIQ indicates a more cost-effective Pareto frontier:

$$\text{AIQ} = \int_{c_{\min}}^{c_{\max}} Q(c) \, dc \approx \sum_{i=0}^{n-1} \frac{q_i + q_{i+1}}{2} \cdot (c_{i+1} - c_i)$$

* **Code Implementation**:
  * Handled in [plot_results.py](file:///c:/Users/pengy/OneDrive/Desktop/InferRoute/benchmarks/plot_results.py) via `calculate_auc()`, using the trapezoidal rule over swept scenarios.
