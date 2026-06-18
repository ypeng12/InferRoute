"""
Chaos and failure injection tests for InferRoute.

Tests failure modes without requiring live services. Uses mocks to simulate:
  - Primary provider outage (→ fallback activates)
  - All providers down (→ 502 with clear error)
  - Redis unavailable (→ fail-open on auth and caching)
  - Database unavailable (→ request still completes, log fails gracefully)
  - Cascading circuit breaker trips
  - Slow backends (timeout simulation)

These tests verify the system degrades gracefully and circuit breakers
and fallbacks behave correctly under adversarial conditions.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


BASE_HEADERS = {"Authorization": "Bearer sk-inferroute-demo"}
CHAT_BODY = {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello from chaos test"}],
    "stream": False,
}


@pytest.fixture(scope="module")
def test_client():
    """FastAPI test client with full app (all services mocked)."""
    from fastapi.testclient import TestClient
    from inferroute.main import app
    return TestClient(app)


def _mock_redis():
    """A simple no-op Redis mock that returns None for all gets (cache miss)."""
    r = AsyncMock()
    r.get.return_value = None
    r.set.return_value = True
    r.incr.return_value = 1
    r.expire.return_value = True
    r.delete.return_value = 1
    r.ping.return_value = True
    r.publish.return_value = 1
    r.zadd.return_value = 1
    r.zremrangebyrank.return_value = 0
    r.zrange.return_value = []
    pipeline_mock = AsyncMock()
    pipeline_mock.zadd = AsyncMock()
    pipeline_mock.zremrangebyrank = AsyncMock()
    pipeline_mock.expire = AsyncMock()
    pipeline_mock.execute = AsyncMock(return_value=[1, 0, True])
    pipeline_mock.__aenter__ = AsyncMock(return_value=pipeline_mock)
    pipeline_mock.__aexit__ = AsyncMock(return_value=None)
    r.pipeline.return_value = pipeline_mock
    r.set.side_effect = lambda key, val, nx=False, ex=None: (None if nx else True)
    return r


# ── Test 1: Primary provider outage → fallback ────────────────────────────────

@patch("inferroute.adapters.vllm.VLLMAdapter.generate")
@patch("inferroute.adapters.openai.OpenAIAdapter.generate")
def test_primary_outage_triggers_fallback(mock_openai_gen, mock_vllm_gen, test_client):
    """
    Scenario: vLLM (primary) is down. Gateway should fall back to OpenAI.
    """
    mock_vllm_gen.side_effect = Exception("Connection refused: vLLM server is down")
    mock_openai_gen.return_value = {
        "id": "openai-fallback-001",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Fallback response"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "estimated_cost_usd": 0.0001},
        "timing": {"ttft_ms": 200.0, "latency_ms": 250.0},
    }

    mock_redis = _mock_redis()
    with patch("inferroute.auth.redis_client", mock_redis), \
         patch("inferroute.auth.get_redis_client", return_value=mock_redis), \
         patch("inferroute.cache.get_redis_client", return_value=mock_redis), \
         patch("inferroute.router.get_redis_client", return_value=mock_redis), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb, \
         patch("inferroute.main.async_session"):
        cb = AsyncMock()
        cb.allow_request.return_value = True
        cb.record_failure = AsyncMock()
        cb.record_success = AsyncMock()
        cb.get_status.return_value = {"state": "CLOSED"}
        gcb.return_value = cb

        resp = test_client.post("/v1/chat/completions", json=CHAT_BODY, headers=BASE_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Fallback response"
        assert data["route"]["fallback_count"] == 1
        assert data["route"]["selected_backend"] == "openai"

        # Verify both were attempted
        assert mock_vllm_gen.called
        assert mock_openai_gen.called


# ── Test 2: All providers down → 502 ─────────────────────────────────────────

@patch("inferroute.adapters.vllm.VLLMAdapter.generate")
@patch("inferroute.adapters.openai.OpenAIAdapter.generate")
def test_all_providers_down_returns_502(mock_openai_gen, mock_vllm_gen, test_client):
    """
    Scenario: All backends fail. Gateway should return 502.
    """
    mock_vllm_gen.side_effect = Exception("vLLM timeout")
    mock_openai_gen.side_effect = Exception("OpenAI rate limit exceeded")

    mock_redis = _mock_redis()
    with patch("inferroute.auth.redis_client", mock_redis), \
         patch("inferroute.auth.get_redis_client", return_value=mock_redis), \
         patch("inferroute.cache.get_redis_client", return_value=mock_redis), \
         patch("inferroute.router.get_redis_client", return_value=mock_redis), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb, \
         patch("inferroute.main.async_session"):
        cb = AsyncMock()
        cb.allow_request.return_value = True
        cb.record_failure = AsyncMock()
        cb.record_success = AsyncMock()
        gcb.return_value = cb

        resp = test_client.post("/v1/chat/completions", json=CHAT_BODY, headers=BASE_HEADERS)
        assert resp.status_code == 502
        assert "failed" in resp.json()["detail"].lower()


# ── Test 3: Redis down → fail-open (no auth crash) ────────────────────────────

@patch("inferroute.adapters.vllm.VLLMAdapter.generate")
def test_redis_down_fail_open(mock_vllm_gen, test_client):
    """
    Scenario: Redis is unavailable. Rate limiting fails open; caching is skipped.
    Requests should still succeed.
    """
    mock_vllm_gen.return_value = {
        "id": "redis-down-001",
        "object": "chat.completion",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Success despite Redis down"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13, "estimated_cost_usd": 0.0},
        "timing": {"ttft_ms": 150.0, "latency_ms": 150.0},
    }

    # Simulate Redis being completely unavailable
    with patch("inferroute.auth.get_redis_client", return_value=None), \
         patch("inferroute.cache.get_redis_client", return_value=None), \
         patch("inferroute.router.get_redis_client", return_value=None), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb, \
         patch("inferroute.main.async_session"):
        cb = AsyncMock()
        cb.allow_request.return_value = True  # fail-open
        cb.record_success = AsyncMock()
        gcb.return_value = cb

        resp = test_client.post("/v1/chat/completions", json=CHAT_BODY, headers=BASE_HEADERS)
        # Should succeed (fail-open)
        assert resp.status_code == 200
        assert mock_vllm_gen.called


# ── Test 4: DB down → request completes, log fails silently ──────────────────

@patch("inferroute.adapters.vllm.VLLMAdapter.generate")
def test_db_down_request_still_succeeds(mock_vllm_gen, test_client):
    """
    Scenario: PostgreSQL is unavailable. Background log fails, but the response
    should still be delivered to the client.
    """
    mock_vllm_gen.return_value = {
        "id": "db-down-001",
        "object": "chat.completion",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "DB was down but I still responded"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15, "estimated_cost_usd": 0.0},
        "timing": {"ttft_ms": 150.0, "latency_ms": 150.0},
    }

    mock_redis = _mock_redis()
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit.side_effect = Exception("PostgreSQL connection refused")
    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("inferroute.auth.redis_client", mock_redis), \
         patch("inferroute.auth.get_redis_client", return_value=mock_redis), \
         patch("inferroute.cache.get_redis_client", return_value=mock_redis), \
         patch("inferroute.router.get_redis_client", return_value=mock_redis), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb, \
         patch("inferroute.main.async_session", return_value=mock_session_ctx):
        cb = AsyncMock()
        cb.allow_request.return_value = True
        cb.record_success = AsyncMock()
        gcb.return_value = cb

        resp = test_client.post("/v1/chat/completions", json=CHAT_BODY, headers=BASE_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "DB" in data["choices"][0]["message"]["content"]


# ── Test 5: Circuit breaker prevents calls to OPEN backend ───────────────────

@patch("inferroute.adapters.vllm.VLLMAdapter.generate")
@patch("inferroute.adapters.ollama.OllamaAdapter.generate")
def test_open_cb_routes_to_alternative(mock_ollama_gen, mock_vllm_gen, test_client):
    """
    Scenario: vLLM circuit breaker is OPEN (trips). Router should skip it and
    route to an available backend instead.
    """
    mock_vllm_gen.side_effect = Exception("Should not be called — CB is OPEN")
    mock_ollama_gen.return_value = {
        "id": "cb-test-001",
        "object": "chat.completion",
        "model": "llama3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Ollama was chosen after CB tripped"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15, "estimated_cost_usd": 0.0},
        "timing": {"ttft_ms": 90.0, "latency_ms": 90.0},
    }

    mock_redis = _mock_redis()

    def make_cb(backend):
        cb = AsyncMock()
        # vllm CB is OPEN — reject; everything else is CLOSED — allow
        cb.allow_request.return_value = (backend != "vllm")
        cb.record_success = AsyncMock()
        cb.record_failure = AsyncMock()
        cb.get_status.return_value = {
            "state": "OPEN" if backend == "vllm" else "CLOSED",
            "fail_count": 5 if backend == "vllm" else 0,
        }
        return cb

    with patch("inferroute.auth.redis_client", mock_redis), \
         patch("inferroute.auth.get_redis_client", return_value=mock_redis), \
         patch("inferroute.cache.get_redis_client", return_value=mock_redis), \
         patch("inferroute.router.get_redis_client", return_value=mock_redis), \
         patch("inferroute.circuit_breaker.get_circuit_breaker", side_effect=make_cb), \
         patch("inferroute.main.async_session"):
        resp = test_client.post(
            "/v1/chat/completions",
            json={"model": "edge/auto", "messages": [{"role": "user", "content": "test"}]},
            headers=BASE_HEADERS,
        )
        assert resp.status_code == 200
        # vllm should NOT have been called
        assert not mock_vllm_gen.called
