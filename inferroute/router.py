"""
SLO-aware routing engine for InferRoute.

Selects the best backend using a multi-objective scoring function that incorporates:
- Sliding-window latency percentiles (p50/p95/p99) from Redis sorted sets
- Cost estimates per provider
- Circuit breaker state (OPEN backends are skipped)
- SLO target compliance (penalizes backends violating SLO targets)
- Prefix-cache bonus (backends with warm context get lower score)
- Routing policy: 'latency', 'cost', or 'reliability'

Backends: openai, gemini, vllm, ollama
"""
import json
import logging
import time
from typing import Any, Optional, NamedTuple

from inferroute.auth import get_redis_client
from inferroute import circuit_breaker
from inferroute.config import settings
from inferroute.observability import (
    ROUTING_DECISION_TOTAL,
    SLO_VIOLATION_TOTAL,
    P50_LATENCY_MS,
    P95_LATENCY_MS,
    P99_LATENCY_MS,
)

logger = logging.getLogger("inferroute.router")

# ── Static baseline performance priors ───────────────────────────────────────
# Used when no dynamic Redis history exists yet.
BASELINES: dict[str, dict[str, Any]] = {
    "vllm": {
        "ttft_ms": 150.0, "p95_ms": 350.0, "p99_ms": 600.0,
        "cost_per_token": 0.00001 / 1000,
        "failure_risk": 0.0, "cache_bonus": 50.0,
    },
    "ollama": {
        "ttft_ms": 120.0, "p95_ms": 280.0, "p99_ms": 500.0,
        "cost_per_token": 0.0,   # local — free
        "failure_risk": 0.0, "cache_bonus": 60.0,
    },
    "openai": {
        "ttft_ms": 250.0, "p95_ms": 800.0, "p99_ms": 1500.0,
        "cost_per_token": 0.15 / 1e6,  # gpt-4o-mini
        "failure_risk": 0.0, "cache_bonus": 0.0,
    },
    "gemini": {
        "ttft_ms": 300.0, "p95_ms": 900.0, "p99_ms": 1800.0,
        "cost_per_token": 0.075 / 1e6,  # gemini-1.5-flash
        "failure_risk": 0.0, "cache_bonus": 0.0,
    },
}

ALL_BACKENDS = list(BASELINES.keys())

# Latency sliding-window size (number of samples tracked per backend)
WINDOW_SIZE = 200


class RoutingDecision(NamedTuple):
    primary: str
    fallback: Optional[str]
    reason: str
    slo_compliant: bool          # whether primary is expected to meet SLO
    policy: str


class BackendScore(NamedTuple):
    name: str
    score: float
    predicted_ttft_ms: float
    expected_cost_usd: float
    failure_risk: float
    p95_ms: float
    p99_ms: float
    reason: str


class Router:

    # ── Redis helpers ─────────────────────────────────────────────────────────

    async def _push_latency_sample(self, backend: str, latency_ms: float) -> None:
        """Push a latency sample into a Redis sorted set (score = timestamp)."""
        client = get_redis_client()
        if client is None:
            return
        key = f"inferroute:latency:{backend}:samples"
        now = time.time()
        try:
            pipe = client.pipeline()
            pipe.zadd(key, {str(latency_ms): now})
            # Keep only the last WINDOW_SIZE samples
            pipe.zremrangebyrank(key, 0, -(WINDOW_SIZE + 1))
            pipe.expire(key, 3600)
            await pipe.execute()
        except Exception as e:
            logger.warning(f"[Router] Failed to push latency sample for {backend}: {e}")

    async def _get_percentiles(self, backend: str) -> dict[str, float]:
        """
        Compute p50/p95/p99 from the Redis sliding window.
        Falls back to BASELINES when insufficient samples exist.
        """
        client = get_redis_client()
        if client is None:
            return {
                "p50_ms": BASELINES[backend]["ttft_ms"],
                "p95_ms": BASELINES[backend]["p95_ms"],
                "p99_ms": BASELINES[backend]["p99_ms"],
            }

        key = f"inferroute:latency:{backend}:samples"
        try:
            # Members sorted by score (timestamp); retrieve all as (member, score) pairs
            members = await client.zrange(key, 0, -1)  # oldest → newest
            if len(members) < 5:
                # Too few samples — use baselines
                return {
                    "p50_ms": BASELINES[backend]["ttft_ms"],
                    "p95_ms": BASELINES[backend]["p95_ms"],
                    "p99_ms": BASELINES[backend]["p99_ms"],
                }

            latencies = sorted(float(m) for m in members)
            n = len(latencies)

            def percentile(p: float) -> float:
                idx = int(p / 100.0 * n)
                return latencies[min(idx, n - 1)]

            return {
                "p50_ms": percentile(50),
                "p95_ms": percentile(95),
                "p99_ms": percentile(99),
            }
        except Exception as e:
            logger.warning(f"[Router] Failed to get percentiles for {backend}: {e}")
            return {
                "p50_ms": BASELINES[backend]["ttft_ms"],
                "p95_ms": BASELINES[backend]["p95_ms"],
                "p99_ms": BASELINES[backend]["p99_ms"],
            }

    async def get_backend_stats(self, backend: str) -> dict[str, Any]:
        """
        Returns merged stats: EMA failure risk + percentile latencies from Redis,
        with BASELINES as fallback.
        """
        client = get_redis_client()
        stats = BASELINES[backend].copy()

        if client is not None:
            try:
                fail_raw = await client.get(f"inferroute:stats:{backend}:failure_rate")
                if fail_raw:
                    stats["failure_risk"] = float(fail_raw)
            except Exception:
                pass

        percentiles = await self._get_percentiles(backend)
        stats["ttft_ms"] = percentiles["p50_ms"]
        stats["p95_ms"] = percentiles["p95_ms"]
        stats["p99_ms"] = percentiles["p99_ms"]
        return stats

    async def record_metrics(
        self, backend: str, ttft_ms: float, latency_ms: float, success: bool
    ) -> None:
        """
        Updates EMA failure rate and pushes latency sample to sliding window.
        Also updates Prometheus percentile gauges.
        """
        client = get_redis_client()

        # Push sample to sliding window
        await self._push_latency_sample(backend, latency_ms)

        # Update EMA failure rate in Redis
        if client is not None:
            try:
                alpha = 0.2
                fail_val = 1.0 if not success else 0.0
                old_fail = await client.get(f"inferroute:stats:{backend}:failure_rate")
                new_fail = (
                    fail_val if not old_fail
                    else (alpha * fail_val + (1 - alpha) * float(old_fail))
                )
                await client.set(f"inferroute:stats:{backend}:failure_rate", str(new_fail))
            except Exception as e:
                logger.error(f"[Router] Failed to update Redis metrics for {backend}: {e}")

        # Refresh Prometheus percentile gauges
        try:
            percentiles = await self._get_percentiles(backend)
            P50_LATENCY_MS.labels(backend=backend).set(percentiles["p50_ms"])
            P95_LATENCY_MS.labels(backend=backend).set(percentiles["p95_ms"])
            P99_LATENCY_MS.labels(backend=backend).set(percentiles["p99_ms"])
        except Exception:
            pass

    # ── SLO checking ──────────────────────────────────────────────────────────

    def _check_slo(self, backend: str, stats: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Returns (compliant, violations) where violations is a list of strings
        describing which SLO targets are being exceeded.
        """
        violations = []
        if stats["ttft_ms"] > settings.SLO_P50_MS:
            violations.append(f"p50={stats['ttft_ms']:.0f}ms > target={settings.SLO_P50_MS:.0f}ms")
            SLO_VIOLATION_TOTAL.labels(backend=backend, percentile="p50").inc()
        if stats["p95_ms"] > settings.SLO_P95_MS:
            violations.append(f"p95={stats['p95_ms']:.0f}ms > target={settings.SLO_P95_MS:.0f}ms")
            SLO_VIOLATION_TOTAL.labels(backend=backend, percentile="p95").inc()
        if stats["p99_ms"] > settings.SLO_P99_MS:
            violations.append(f"p99={stats['p99_ms']:.0f}ms > target={settings.SLO_P99_MS:.0f}ms")
            SLO_VIOLATION_TOTAL.labels(backend=backend, percentile="p99").inc()
        return len(violations) == 0, violations

    # ── Main routing logic ────────────────────────────────────────────────────

    async def choose_backend(self, req: dict[str, Any]) -> RoutingDecision:
        """
        Evaluates all available backends and returns the best routing decision.

        Selection order:
        1. Hard-pin by model name.
        2. Filter by routing opts (allow_local, allow_cloud).
        3. Filter out OPEN circuit breakers.
        4. Score remaining candidates by policy.
        5. Select primary (lowest score) and fallback (second-lowest).
        """
        model_req = req.get("model", "edge/auto")
        routing_opts = req.get("routing", {})
        policy = routing_opts.get("policy", "latency")

        # ── Hard pins ────────────────────────────────────────────────────────
        if model_req in ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"):
            ROUTING_DECISION_TOTAL.labels(policy="hard_pin", backend="openai").inc()
            return RoutingDecision("openai", "gemini", f"Pinned to OpenAI: {model_req}", True, "hard_pin")

        if model_req in ("gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"):
            ROUTING_DECISION_TOTAL.labels(policy="hard_pin", backend="gemini").inc()
            return RoutingDecision("gemini", "openai", f"Pinned to Gemini: {model_req}", True, "hard_pin")

        if model_req in ("meta-llama/Meta-Llama-3-8B-Instruct", "llama-3-8b"):
            ROUTING_DECISION_TOTAL.labels(policy="hard_pin", backend="vllm").inc()
            return RoutingDecision("vllm", "openai", f"Pinned to vLLM: {model_req}", True, "hard_pin")

        if model_req.startswith("ollama/") or model_req in ("llama3", "mistral", "phi3"):
            ROUTING_DECISION_TOTAL.labels(policy="hard_pin", backend="ollama").inc()
            return RoutingDecision("ollama", "vllm", f"Pinned to Ollama: {model_req}", True, "hard_pin")

        # ── Candidate filtering ───────────────────────────────────────────────
        allow_local = routing_opts.get("allow_local", True)
        allow_cloud = routing_opts.get("allow_cloud", True)

        local_backends = {"vllm", "ollama"}
        cloud_backends = {"openai", "gemini"}

        candidates: list[str] = []
        for b in ALL_BACKENDS:
            if b in local_backends and not allow_local:
                continue
            if b in cloud_backends and not allow_cloud:
                continue
            candidates.append(b)

        if not candidates:
            candidates = ALL_BACKENDS[:]

        # ── Circuit breaker filtering ─────────────────────────────────────────
        available: list[str] = []
        for b in candidates:
            cb = circuit_breaker.get_circuit_breaker(b)
            if await cb.allow_request():
                available.append(b)
            else:
                logger.info(f"[Router] Skipping {b} — circuit breaker OPEN")

        if not available:
            # All circuit breakers open — try candidates anyway (fail-open)
            logger.warning("[Router] All circuit breakers OPEN — routing to first candidate anyway")
            available = candidates[:1]

        # ── Single candidate shortcut ─────────────────────────────────────────
        if len(available) == 1:
            selected = available[0]
            fallback_candidates = [b for b in candidates if b != selected]
            fallback = fallback_candidates[0] if fallback_candidates else None
            ROUTING_DECISION_TOTAL.labels(policy="single_candidate", backend=selected).inc()
            return RoutingDecision(selected, fallback, f"Only available backend: {selected}", True, policy)

        # ── Multi-objective scoring ───────────────────────────────────────────
        if policy == "cost":
            w_ttft, w_p95, w_cost, w_fail = 0.05, 0.05, 0.80, 0.10
        elif policy == "reliability":
            w_ttft, w_p95, w_cost, w_fail = 0.20, 0.20, 0.05, 0.55
        else:  # latency (default)
            w_ttft, w_p95, w_cost, w_fail = 0.40, 0.35, 0.10, 0.15

        # Estimate prompt token count for cost forecasting
        prompt_len = sum(len(m.get("content", "")) for m in req.get("messages", []))
        input_tokens = prompt_len // 4 + 10
        expected_output = req.get("max_output_tokens", 256)

        scores: list[BackendScore] = []
        for backend in available:
            stats = await self.get_backend_stats(backend)

            # Cost estimate
            cpt = stats["cost_per_token"]
            expected_cost = (input_tokens * cpt) + (expected_output * cpt * 2.0)

            # SLO compliance penalty (adds 500ms equivalent to score if violating)
            slo_ok, slo_violations = self._check_slo(backend, stats)
            slo_penalty = 0.0 if slo_ok else 500.0

            score_val = (
                w_ttft * stats["ttft_ms"]
                + w_p95 * stats["p95_ms"]
                + w_cost * expected_cost * 100_000.0   # scale to ms-comparable
                + w_fail * stats["failure_risk"] * 1000.0
                - stats["cache_bonus"]                  # warm cache lowers score (good)
                + slo_penalty
            )

            scores.append(BackendScore(
                name=backend,
                score=score_val,
                predicted_ttft_ms=stats["ttft_ms"],
                expected_cost_usd=expected_cost,
                failure_risk=stats["failure_risk"],
                p95_ms=stats["p95_ms"],
                p99_ms=stats["p99_ms"],
                reason=(
                    f"score={score_val:.1f} "
                    f"(ttft={stats['ttft_ms']:.0f}ms, p95={stats['p95_ms']:.0f}ms, "
                    f"cost=${expected_cost:.6f}, fail={stats['failure_risk']:.3f}, "
                    f"slo_ok={slo_ok})"
                ),
            ))

        scores.sort(key=lambda x: x.score)
        primary = scores[0]
        fallback_name = scores[1].name if len(scores) > 1 else None

        slo_compliant, _ = self._check_slo(primary.name, await self.get_backend_stats(primary.name))

        ROUTING_DECISION_TOTAL.labels(policy=policy, backend=primary.name).inc()
        reasoning = (
            f"Selected '{primary.name}' via policy='{policy}'. "
            + "Scores: "
            + ", ".join(f"{s.name}={s.score:.1f}" for s in scores)
        )
        logger.info(f"[Router] {reasoning}")

        return RoutingDecision(
            primary=primary.name,
            fallback=fallback_name,
            reason=reasoning,
            slo_compliant=slo_compliant,
            policy=policy,
        )
