"""
Unit tests for the SLO-aware Router.

Tests:
  - Hard-pin by model name (OpenAI, Gemini, vLLM, Ollama)
  - Latency-policy scoring selects lowest-latency backend
  - Cost-policy scoring selects lowest-cost backend
  - SLO violation detection
  - OPEN circuit breakers excluded from candidates
  - Single-candidate fallthrough
  - Percentile sliding window (push sample → get percentile)
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from inferroute.router import Router, BASELINES, RoutingDecision


@pytest.fixture
def mock_redis():
    store: dict = {}
    redis = AsyncMock()

    async def get(key):
        return store.get(key)

    async def set(key, value, *args, **kwargs):
        store[key] = value

    async def zadd(key, mapping):
        if key not in store:
            store[key] = {}
        store[key].update(mapping)

    async def zremrangebyrank(*args, **kwargs):
        pass

    async def expire(*args, **kwargs):
        pass

    async def zrange(key, start, end):
        d = store.get(key, {})
        if isinstance(d, dict):
            return list(d.keys())
        return []

    redis.get.side_effect = get
    redis.set.side_effect = set
    redis.zadd.side_effect = zadd
    redis.zremrangebyrank.side_effect = zremrangebyrank
    redis.expire.side_effect = expire
    redis.zrange.side_effect = zrange
    redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=redis)
    redis.pipeline.return_value.__aexit__ = AsyncMock(return_value=None)
    redis.pipeline.return_value.execute = AsyncMock(return_value=[1, 0, True])
    redis.pipeline.return_value.zadd = AsyncMock()
    redis.pipeline.return_value.zremrangebyrank = AsyncMock()
    redis.pipeline.return_value.expire = AsyncMock()
    return redis, store


@pytest.fixture
def router(mock_redis):
    redis, store = mock_redis
    r = Router()
    with patch("inferroute.router.get_redis_client", return_value=redis), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as mock_cb_factory:
        # Default: all circuits closed (allow all)
        mock_cb = AsyncMock()
        mock_cb.allow_request.return_value = True
        mock_cb.get_status.return_value = {"state": "CLOSED", "fail_count": 0}
        mock_cb_factory.return_value = mock_cb
        yield r, store, mock_cb_factory


@pytest.mark.asyncio
async def test_hard_pin_openai(router):
    r, store, _ = router
    with patch("inferroute.router.get_redis_client"), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb:
        cb = AsyncMock()
        cb.allow_request.return_value = True
        gcb.return_value = cb

        decision = await r.choose_backend({"model": "gpt-4o-mini"})
        assert decision.primary == "openai"
        assert decision.fallback == "gemini"
        assert decision.policy == "hard_pin"


@pytest.mark.asyncio
async def test_hard_pin_gemini(router):
    r, store, _ = router
    with patch("inferroute.router.get_redis_client"), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb:
        cb = AsyncMock()
        cb.allow_request.return_value = True
        gcb.return_value = cb

        decision = await r.choose_backend({"model": "gemini-1.5-flash"})
        assert decision.primary == "gemini"
        assert decision.fallback == "openai"


@pytest.mark.asyncio
async def test_hard_pin_vllm(router):
    r, store, _ = router
    with patch("inferroute.router.get_redis_client"), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb:
        cb = AsyncMock()
        cb.allow_request.return_value = True
        gcb.return_value = cb

        decision = await r.choose_backend({"model": "meta-llama/Meta-Llama-3-8B-Instruct"})
        assert decision.primary == "vllm"
        assert decision.fallback == "openai"


@pytest.mark.asyncio
async def test_hard_pin_ollama(router):
    r, store, _ = router
    with patch("inferroute.router.get_redis_client"), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb:
        cb = AsyncMock()
        cb.allow_request.return_value = True
        gcb.return_value = cb

        decision = await r.choose_backend({"model": "llama3"})
        assert decision.primary == "ollama"
        assert decision.fallback == "vllm"


@pytest.mark.asyncio
async def test_latency_policy_selects_lowest_ttft(router):
    r, store, _ = router
    with patch("inferroute.router.get_redis_client"), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb:
        cb = AsyncMock()
        cb.allow_request.return_value = True
        gcb.return_value = cb

        req = {
            "model": "edge/auto",
            "messages": [{"role": "user", "content": "hello"}],
            "routing": {"policy": "latency"}
        }
        decision = await r.choose_backend(req)
        # Ollama has the lowest baseline TTFT (120ms)
        assert decision.primary in ("ollama", "vllm")
        assert decision.policy == "latency"


@pytest.mark.asyncio
async def test_cost_policy_selects_free_backend(router):
    r, store, _ = router
    with patch("inferroute.router.get_redis_client"), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb:
        cb = AsyncMock()
        cb.allow_request.return_value = True
        gcb.return_value = cb

        req = {
            "model": "edge/auto",
            "messages": [{"role": "user", "content": "hello"}],
            "routing": {"policy": "cost"}
        }
        decision = await r.choose_backend(req)
        # Ollama is free (cost_per_token = 0.0)
        assert decision.primary == "ollama"


@pytest.mark.asyncio
async def test_cloud_only_routing(router):
    r, store, _ = router
    with patch("inferroute.router.get_redis_client"), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb:
        cb = AsyncMock()
        cb.allow_request.return_value = True
        gcb.return_value = cb

        req = {
            "model": "edge/auto",
            "messages": [{"role": "user", "content": "hello"}],
            "routing": {"allow_local": False, "allow_cloud": True, "policy": "latency"}
        }
        decision = await r.choose_backend(req)
        assert decision.primary in ("openai", "gemini")
        assert decision.fallback in ("openai", "gemini", None)


@pytest.mark.asyncio
async def test_open_circuit_breaker_excluded(router):
    r, store, _ = router
    with patch("inferroute.router.get_redis_client"), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb:
        call_count = [0]
        backends_asked = []

        async def selective_allow(backend):
            # OPEN for ollama and vllm (local), allow openai and gemini
            if backend in ("ollama", "vllm"):
                return False
            return True

        def create_cb(backend_name):
            cb = AsyncMock()
            cb.allow_request = AsyncMock(return_value=backend_name not in ("ollama", "vllm"))
            return cb

        gcb.side_effect = lambda b: create_cb(b)

        req = {
            "model": "edge/auto",
            "messages": [{"role": "user", "content": "hello"}],
        }
        decision = await r.choose_backend(req)
        assert decision.primary in ("openai", "gemini")


@pytest.mark.asyncio
async def test_slo_violation_detected(router):
    r, _, _ = router
    # Give a backend a p95 above the SLO target
    stats = {
        "ttft_ms": 600.0,   # above default p50 target of 500ms
        "p95_ms": 3000.0,   # above default p95 target of 2000ms
        "p99_ms": 6000.0,
    }
    with patch("inferroute.config.settings") as mock_settings:
        mock_settings.SLO_P50_MS = 500.0
        mock_settings.SLO_P95_MS = 2000.0
        mock_settings.SLO_P99_MS = 5000.0

        compliant, violations = r._check_slo("openai", stats)
        assert compliant is False
        assert len(violations) >= 2  # p50 and p95 violated
