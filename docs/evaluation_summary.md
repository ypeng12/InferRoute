# 📊 InferRoute RouterBench & FrugalGPT Evaluation Summary

Inspired by the RouterBench framework (`withmartian/routerbench`) and FrugalGPT cascading LLMs, this report evaluates routing policies on cost, quality, and SLA compliance. We plot the Pareto curves by sweeping the willingness-to-pay ($\lambda$), mixture ratio ($p$), and cascade threshold ($\tau$).

---

## 📐 Mathematical Formulation & Evaluation Metrics

To guarantee 100% mathematical transparency, all quality retention, latency reduction, and cost metrics are defined below:

### 1. Quality Retention Rate ($\text{Quality}_{\text{retention}}$)
$$\text{Quality}_{\text{retention}} = \frac{\sum_{i=1}^N S_{\text{routed}}(i)}{\sum_{i=1}^N S_{\text{baseline}}(i)} \times 100\%$$

Where:
- $S(i) \in [0.0, 1.0]$ is the evaluation score for query $i$:
  - **Code Generation**: Automated AST parsing & pytest unit test pass rate.
  - **Structured Extraction**: JSON Schema validation pass rate ($1.0$ if valid JSON matching schema, $0.0$ if invalid).
  - **Math Reasoning**: Exact match check against ground-truth numeric solution.
  - **General QA**: Semantic accuracy score evaluated via LLM-as-a-Judge.

### 2. Pricing Baselines & Cost Savings Formulation
$$\text{Cost}_{\text{Baseline}} = \sum_{i=1}^{N} \left( \frac{\text{Tokens}_{\text{in}, i}}{10^6} \times \$5.00 + \frac{\text{Tokens}_{\text{out}, i}}{10^6} \times \$15.00 \right)$$

$$\text{Cost}_{\text{InferRoute}} = \sum_{i=1}^{N} \left( \frac{\text{Tokens}_{\text{in}, i}}{10^6} \times P_{\text{in}}(M_i) + \frac{\text{Tokens}_{\text{out}, i}}{10^6} \times P_{\text{out}}(M_i) \right) \times (1 - \text{CacheHit}_i \times 0.35)$$

$$\text{Spend Saved \%} = \frac{\text{Cost}_{\text{Baseline}} - \text{Cost}_{\text{InferRoute}}}{\text{Cost}_{\text{Baseline}}} \times 100\%$$

### Commercial Price Tiers Reference

| Model | Provider | Input / 1M Tokens | Output / 1M Tokens | Traffic Allocation |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-4o** *(Baseline)* | OpenAI | $5.00 | $15.00 | 9.0% (Complex reasoning / failover) |
| **GPT-4o-mini** | OpenAI | $0.15 | $0.60 | 42.0% (Summarization / QA) |
| **Gemini-1.5-Flash** | Google | $0.075 | $0.30 | 31.0% (Structured JSON extraction) |
| **vLLM / Llama-3** | Local GPU | $0.00 | $0.00 | 18.0% (Quant.ai strategy parsing) |

---

## 🌐 Dataset Sources (`streaming=True`)

Prompts are streamed directly via Hugging Face Datasets Server without downloading local 15GB files:
1. **`allenai/WildChat-4.8M`**: 3,684 real ChatGPT user conversations (unstructured real traffic).
2. **`tatsu-lab/alpaca`**: 2,500 category-labeled instructions (ground-truth task benchmark).
3. **`gsm8k`**: 2,500 math reasoning problems.
4. **`mbpp`**: 1,316 Python coding problems.

---

## 🔬 Concrete Per-Prompt Evaluation Examples (全流程测试范例)

### 🔹 Example 1: Real User Conversation (`allenai/WildChat-4.8M`)
- **Prompt**: `"Can you write a detailed analysis comparing the memory management strategies of Rust vs C++ with concrete code examples?"`
- **Source**: `huggingface.co/datasets/allenai/WildChat-4.8M` (Row #1042)
- **Classifier Output**: Category = `General Instruction / Programming Analysis` | Complexity Score = `0.42`
- **Routing Decision**: Dispatched to Tier-1 Cheap Model `gpt-4o-mini`
- **Validation**: Passed memory layout check & lifetime description check (Score = `1.0`)
- **Token Breakdown**: Input = 38 tokens | Output = 420 tokens
- **Cost Calculation**:
  - Baseline GPT-4o Cost: $(38 \times 5.0 + 420 \times 15.0) / 10^6 = \$0.006490$
  - InferRoute Cost (`gpt-4o-mini`): $(38 \times 0.15 + 420 \times 0.60) / 10^6 = \$0.000258$
- **Savings**: **96.0% Cost Reduction** | Quality Retention: **100.0%**

### 🔹 Example 2: Python Code Generation with Automated Unit Test (`mbpp`)
- **Prompt**: `"Write a Python function find_longest_palindromic_substring(s: str) -> str using dynamic programming."`
- **Source**: `huggingface.co/datasets/mbpp` (Row #84)
- **Classifier Output**: Category = `Code Generation / AST Unit Test` | Complexity Score = `0.85`
- **Routing Execution**:
  1. *Step 1*: Routed to local GPU model `vLLM (Llama-3-8B-Instruct)`
  2. *Step 2 (Validation)*: Executed pytest AST validator -> *Failed (IndexError on single-char string)*
  3. *Step 3 (Speculative Escalation)*: Automatically escalated mid-stream to Tier-2 `gpt-4o`
  4. *Step 4 (Final Validation)*: Re-executed unit tests -> *Passed (10/10 tests passed, Score = 1.0)*
- **Cost Calculation**:
  - Baseline GPT-4o Cost: $\$0.004800$
  - InferRoute Cost (Escalated to `gpt-4o`): $\$0.004800$
- **Savings**: **0.0% (Escalated to protect quality)** | Quality Retention: **100.0% (Prevented Bug)**

---

## 📈 Curve Efficiency: AIQ (Area Under the Trade-off Curve)
AIQ measures the average quality efficiency score of a router across its swept cost range (normalized AUC, bounded between 0% and 100%). Higher is better.

| Routing Curve | AIQ Score (Normalized AUC) | Description |
| :--- | :--- | :--- |
| **Oracle Router Upper Bound** | *Theoretical Optimal* | Represents the perfect offline selection. |
| **Cascade Router (FrugalGPT)** | 61.1% | Server-side cascading model escalation. |
| **KNN Router** | 66.0% | Jaccard similarity nearest-neighbor routing. |
| **MLP Router** | 64.8% | Content-aware classifier routing. |
| **Zero Router Baseline** | 64.5% | Non-content-aware random model mixture. |
