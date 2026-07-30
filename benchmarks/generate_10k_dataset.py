"""
Dataset generator & Hugging Face benchmark loader for InferRoute.
Builds a 10,000-item evaluation dataset (workload_10k.json) covering 4 key industry task profiles:
1. Customer support summarization (42% -> 4,200 prompts)
2. Structured information extraction (31% -> 3,100 prompts)
3. Quant.ai strategy parsing & generation (18% -> 1,800 prompts)
4. Complex reasoning & multi-step coding (9% -> 900 prompts)
"""

import os
import json
import random

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(DATASETS_DIR, "datasets", "workload_10k.json")

SUMMARIZATION_TEMPLATES = [
    "Summarize the following customer support ticket regarding order #{id}: The customer reported a delayed package shipped via express delivery.",
    "Extract key summary points from invoice #{id}: Total amount $1,240, paid via Credit Card, items include cloud hosting and domain renewal.",
    "Summarize the refund request for user #{id}: User claims product arrived with minor packaging damage.",
    "Provide a 2-sentence executive summary of the monthly IT infrastructure incident report #{id}."
]

EXTRACTION_TEMPLATES = [
    "Extract JSON schema from vendor agreement #{id}: Vendor Name, Contract Value, Expiration Date, Renewal Notice Days.",
    "Parse customer details into JSON format for account #{id}: Full Name, Email Address, Subscription Tier, Monthly Spend.",
    "Extract structured error metadata from application log entry #{id}: Timestamp, Severity Level, Error Code, Stack Summary.",
    "Parse purchase order #{id} into standard JSON schema with line items and subtotal."
]

QUANT_TEMPLATES = [
    "Generate Quant.ai momentum strategy for ticker AAPL_{id}: Stop loss 5%, Take profit 15%, Position weight 0.25, Entry condition RSI > 60.",
    "Parse quantitative backtest parameters for strategy ID {id}: Ticker NVDA, Timeframe 15m, Max Drawdown 12%, Sharpe Ratio 2.1.",
    "Validate portfolio rebalancing allocation for risk model #{id}: Asset weights AAPL: 40%, NVDA: 35%, TSLA: 25%.",
    "Generate Quant.ai mean-reversion trading rules for ticker TSLA_{id} with 20-day Bollinger Band breakout logic."
]

REASONING_TEMPLATES = [
    "Write a production-grade Python class implementing a thread-safe LRU cache with O(1) time complexity for get and put methods, request ID #{id}.",
    "Explain CAP theorem tradeoffs in distributed database design when choosing between Cassandra and PostgreSQL for high-write workloads #{id}.",
    "Derive the time and space complexity analysis for a multi-threaded graph traversal algorithm handling 1M nodes #{id}.",
    "Write a Rust function implementing a lock-free ring buffer for low-latency market data processing #{id}."
]

def generate_10k_workload():
    os.makedirs(os.path.join(DATASETS_DIR, "datasets"), exist_ok=True)
    dataset = []

    # 1. Summarization (4,200)
    for i in range(1, 4201):
        tmpl = random.choice(SUMMARIZATION_TEMPLATES)
        dataset.append({
            "id": f"sum_{i:04d}",
            "category": "summarization",
            "prompt": tmpl.format(id=1000 + i),
            "requires_json": False,
            "target_model_tier": "cheap"
        })

    # 2. Structured Extraction (3,100)
    for i in range(1, 3101):
        tmpl = random.choice(EXTRACTION_TEMPLATES)
        dataset.append({
            "id": f"ext_{i:04d}",
            "category": "structured_extraction",
            "prompt": tmpl.format(id=2000 + i),
            "requires_json": True,
            "target_model_tier": "cheap_or_medium"
        })

    # 3. Quant.ai Strategy (1,800)
    for i in range(1, 1801):
        tmpl = random.choice(QUANT_TEMPLATES)
        dataset.append({
            "id": f"quant_{i:04d}",
            "category": "quant_strategy",
            "prompt": tmpl.format(id=3000 + i),
            "requires_json": True,
            "target_model_tier": "local_vllm"
        })

    # 4. Complex Reasoning (900)
    for i in range(1, 901):
        tmpl = random.choice(REASONING_TEMPLATES)
        dataset.append({
            "id": f"reason_{i:04d}",
            "category": "complex_reasoning",
            "prompt": tmpl.format(id=4000 + i),
            "requires_json": False,
            "target_model_tier": "strong"
        })

    random.shuffle(dataset)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"[SUCCESS] Generated 10,000-item benchmark dataset at: {OUTPUT_FILE}")
    print(f"   - Summarization: 4,200 (42%)")
    print(f"   - Extraction:    3,100 (31%)")
    print(f"   - Quant.ai:      1,800 (18%)")
    print(f"   - Reasoning:       900 (9%)")

if __name__ == "__main__":
    generate_10k_workload()
