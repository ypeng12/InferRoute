import json
import logging
from typing import Any, Optional, NamedTuple
from edgeserve.auth import get_redis_client
from edgeserve.observability import ROUTING_DECISION_TOTAL

logger = logging.getLogger("edgeserve.router")

# Base baseline performance configurations (used if no dynamic history exists)
BASELINES = {
    "vllm": {
        "ttft_ms": 150.0,
        "p95_ms": 350.0,
        "cost_per_token": 0.00001 / 1000, # cheap local amortization
        "failure_risk": 0.0,
        "cache_bonus": 50.0,
    },
    "openai": {
        "ttft_ms": 250.0,
        "p95_ms": 800.0,
        "cost_per_token": 0.15 / 1e6, # gpt-4o-mini pricing
        "failure_risk": 0.0,
        "cache_bonus": 0.0,
    }
}

class BackendScore(NamedTuple):
    name: str
    score: float
    predicted_ttft_ms: float
    expected_cost_usd: float
    failure_risk: float
    reason: str

class Router:
    async def get_backend_stats(self, backend: str) -> dict[str, float]:
        """
        Retrieves running metrics for a backend from Redis, falling back to static baselines.
        """
        client = get_redis_client()
        if client is None:
            return BASELINES[backend]

        try:
            # We fetch rolling averages stored in Redis
            # In a real system, these would be computed asynchronously or written after each request
            ttft_raw = await client.get(f"edgeserve:stats:{backend}:ttft")
            p95_raw = await client.get(f"edgeserve:stats:{backend}:p95")
            fail_raw = await client.get(f"edgeserve:stats:{backend}:failure_rate")
            
            stats = BASELINES[backend].copy()
            if ttft_raw:
                stats["ttft_ms"] = float(ttft_raw)
            if p95_raw:
                stats["p95_ms"] = float(p95_raw)
            if fail_raw:
                stats["failure_risk"] = float(fail_raw)
                
            return stats
        except Exception as e:
            logger.warning(f"Failed to fetch dynamic stats for {backend}: {e}. Using baselines.")
            return BASELINES[backend]

    async def record_metrics(self, backend: str, ttft_ms: float, latency_ms: float, success: bool) -> None:
        """
        Updates the running metrics for a backend in Redis.
        Uses exponential moving average for simplicity.
        """
        client = get_redis_client()
        if client is None:
            return
            
        try:
            alpha = 0.2 # EMA smoothing factor
            
            # Update TTFT
            old_ttft = await client.get(f"edgeserve:stats:{backend}:ttft")
            new_ttft = ttft_ms if not old_ttft else (alpha * ttft_ms + (1 - alpha) * float(old_ttft))
            await client.set(f"edgeserve:stats:{backend}:ttft", str(new_ttft))
            
            # Update P95
            old_p95 = await client.get(f"edgeserve:stats:{backend}:p95")
            new_p95 = latency_ms if not old_p95 else (alpha * latency_ms + (1 - alpha) * float(old_p95))
            await client.set(f"edgeserve:stats:{backend}:p95", str(new_p95))
            
            # Update Failure rate (1 for fail, 0 for success)
            fail_val = 1.0 if not success else 0.0
            old_fail = await client.get(f"edgeserve:stats:{backend}:failure_rate")
            new_fail = fail_val if not old_fail else (alpha * fail_val + (1 - alpha) * float(old_fail))
            await client.set(f"edgeserve:stats:{backend}:failure_rate", str(new_fail))
            
        except Exception as e:
            logger.error(f"Failed to update Redis metrics for {backend}: {e}")

    async def choose_backend(self, req: dict[str, Any]) -> tuple[str, Optional[str], str]:
        """
        Evaluates backend options and returns (primary_backend, fallback_backend, reasoning).
        Respects hard routing constraints (e.g. routing.allow_local, routing.allow_cloud).
        """
        model_req = req.get("model", "edge/auto")
        
        # Hard pin constraints based on requested model name
        if model_req in ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]:
            ROUTING_DECISION_TOTAL.labels(policy="hard_pin", backend="openai").inc()
            return "openai", None, f"Model pinned to OpenAI: {model_req}"
            
        if model_req in ["meta-llama/Meta-Llama-3-8B-Instruct", "llama-3-8b"]:
            ROUTING_DECISION_TOTAL.labels(policy="hard_pin", backend="vllm").inc()
            return "vllm", "openai", f"Model pinned to local vLLM: {model_req}"
            
        # Parse routing policy options
        routing_opts = req.get("routing", {})
        allow_local = routing_opts.get("allow_local", True)
        allow_cloud = routing_opts.get("allow_cloud", True)
        policy = routing_opts.get("policy", "latency") # latency or cost

        candidates = []
        if allow_local:
            candidates.append("vllm")
        if allow_cloud:
            candidates.append("openai")

        if not candidates:
            # Fallback default if inputs are contradictory
            candidates = ["vllm", "openai"]

        if len(candidates) == 1:
            selected = candidates[0]
            fallback = "openai" if selected == "vllm" else "vllm"
            ROUTING_DECISION_TOTAL.labels(policy="single_candidate", backend=selected).inc()
            return selected, fallback, f"Only one backend allowed: {selected}"

        # Evaluate weights based on policy
        if policy == "cost":
            # Cost-focused optimization: heavy weight on cost, moderate on latency
            w_ttft = 0.1
            w_p95 = 0.1
            w_cost = 0.7
            w_fail = 0.1
        else: # latency
            # Latency-focused optimization: heavy weight on TTFT/latency, low on cost
            w_ttft = 0.4
            w_p95 = 0.3
            w_cost = 0.1
            w_fail = 0.2

        scores = []
        # Estimate request prompt size to forecast cost
        prompt_len = sum(len(m.get("content", "")) for m in req.get("messages", []))
        input_tokens = prompt_len // 4 + 10
        expected_output = req.get("max_output_tokens", 256)

        for backend in candidates:
            stats = await self.get_backend_stats(backend)
            
            # Predict costs
            expected_cost = 0.0
            if backend == "openai":
                prices = BASELINES["openai"]
                expected_cost = (input_tokens * prices["cost_per_token"]) + (expected_output * prices["cost_per_token"] * 4)
            else: # vllm
                prices = BASELINES["vllm"]
                expected_cost = (input_tokens * prices["cost_per_token"]) + (expected_output * prices["cost_per_token"] * 1.5)

            # Score function
            score_val = (
                w_ttft * stats["ttft_ms"]
                + w_p95 * stats["p95_ms"]
                + w_cost * expected_cost * 100000.0 # Scale cost for comparison
                + w_fail * stats["failure_risk"] * 1000.0
                - stats["cache_bonus"] # cache bonus reduces score (lower is better)
            )
            
            scores.append(BackendScore(
                name=backend,
                score=score_val,
                predicted_ttft_ms=stats["ttft_ms"],
                expected_cost_usd=expected_cost,
                failure_risk=stats["failure_risk"],
                reason=f"score={score_val:.2f} (ttft={stats['ttft_ms']}ms, cost=${expected_cost:.6f}, fail_risk={stats['failure_risk']})"
            ))

        # Sort scores (ascending order: lowest score is the best option)
        scores.sort(key=lambda x: x.score)
        primary = scores[0].name
        fallback = scores[1].name if len(scores) > 1 else None
        
        ROUTING_DECISION_TOTAL.labels(policy=policy, backend=primary).inc()
        reasoning = f"Selected {primary} using policy '{policy}'. Candidates: " + ", ".join(f"{s.name}={s.score:.2f}" for s in scores)
        logger.info(f"Router decision: {reasoning}")
        
        return primary, fallback, reasoning
