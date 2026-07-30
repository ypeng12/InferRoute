"""
Unified Cost Calculation Engine for InferRoute Gateway.

Pricing Snapshot Date: 2026-07-30
All prices per 1,000,000 (1M) tokens based on official provider tiers.
"""

PRICING_SNAPSHOT_DATE = "2026-07-30"

PROVIDER_PRICING = {
    "gpt-4o": {
        "provider": "OpenAI Cloud",
        "input_per_1m": 5.00,
        "output_per_1m": 15.00,
        "cached_input_per_1m": 2.50
    },
    "gpt-4o-mini": {
        "provider": "OpenAI Cloud",
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
        "cached_input_per_1m": 0.075
    },
    "gemini-1.5-flash": {
        "provider": "Google Cloud API",
        "input_per_1m": 0.075,
        "output_per_1m": 0.30,
        "cached_input_per_1m": 0.01875
    },
    "claude-3-5-sonnet": {
        "provider": "Anthropic API",
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "cached_input_per_1m": 0.30
    },
    "vllm-llama-3": {
        "provider": "On-Prem vLLM Cluster",
        "input_per_1m": 0.00,
        "output_per_1m": 0.00,
        "cached_input_per_1m": 0.00,
        "marginal_api_cost": 0.00,
        "est_infra_cost_per_1k_reqs": 0.42  # Infrastructure & compute depreciation estimate
    }
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0
) -> dict:
    """
    Unified cost calculator for all InferRoute endpoints & dashboard cards.
    Guarantees 100% mathematical consistency across Live Trace, Calculator, and Benchmark views.
    """
    p_info = PROVIDER_PRICING.get(model, PROVIDER_PRICING["gpt-4o"])
    
    cached_in = min(cached_input_tokens, input_tokens)
    uncached_in = max(0, input_tokens - cached_in)

    input_cost = (uncached_in * p_info["input_per_1m"] + cached_in * p_info.get("cached_input_per_1m", p_info["input_per_1m"])) / 1e6
    output_cost = (output_tokens * p_info["output_per_1m"]) / 1e6
    total_cost = input_cost + output_cost

    # Baseline GPT-4o cost for identical tokens
    baseline_info = PROVIDER_PRICING["gpt-4o"]
    baseline_cost = (input_tokens * baseline_info["input_per_1m"] + output_tokens * baseline_info["output_per_1m"]) / 1e6

    savings_usd = max(0.0, baseline_cost - total_cost)
    savings_pct = round((savings_usd / baseline_cost) * 100, 1) if baseline_cost > 0 else 0.0

    return {
        "model": model,
        "provider": p_info["provider"],
        "pricing_date": PRICING_SNAPSHOT_DATE,
        "input_tokens": input_tokens,
        "uncached_input_tokens": uncached_in,
        "cached_input_tokens": cached_in,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "baseline_gpt4o_cost_usd": round(baseline_cost, 6),
        "savings_usd": round(savings_usd, 6),
        "savings_percent": savings_pct,
        "marginal_api_cost_usd": round(total_cost, 6),
        "est_infra_cost_per_1k_reqs": p_info.get("est_infra_cost_per_1k_reqs", 0.0)
    }
