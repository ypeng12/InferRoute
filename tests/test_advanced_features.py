"""
Unit and integration tests for advanced gateway features:
1. Streaming Request Deduplication
2. KV-Cache Affinity Routing
3. Speculative / Cascade Routing
4. Adaptive Rate Limiting
"""
import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from inferroute.main import app
from inferroute.rate_limiter import AdaptiveConcurrencyLimiter
from inferroute.router_trie import PrefixTrieRouter

BASE_HEADERS = {"Authorization": "Bearer sk-inferroute-demo"}
CHAT_BODY = {
    "model": "edge/auto",
    "messages": [{"role": "user", "content": "Explain KV caching in vLLM in detail."}],
    "stream": False,
}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _mock_redis():
    """A clean mock Redis client for routing and cache states."""
    r = AsyncMock()
    # Lock mock
    r.set.return_value = True
    r.get.return_value = None
    r.delete.return_value = 1
    r.exists.return_value = 0
    
    # Pub/sub mock
    pubsub_mock = AsyncMock()
    pubsub_mock.subscribe = AsyncMock()
    pubsub_mock.unsubscribe = AsyncMock()
    pubsub_mock.aclose = AsyncMock()
    
    # Simulate a stream sequence for Wait Stream Dedup:
    # We yield two chunks, then a done message
    chunks = [
        {"type": "message", "data": json.dumps({"index": 0, "chunk": {"choices": [{"delta": {"content": "Hello"}}]}})},
        {"type": "message", "data": json.dumps({"index": 1, "chunk": {"choices": [{"delta": {"content": " world"}}]}})},
        {"type": "message", "data": json.dumps({"index": "done", "final_index": 2})}
    ]
    
    async def get_message_mock(*args, **kwargs):
        if chunks:
            return chunks.pop(0)
        await asyncio.sleep(0.1)
        return None
        
    pubsub_mock.get_message.side_effect = get_message_mock
    # Make pubsub a synchronous MagicMock so it does not return a coroutine
    r.pubsub = MagicMock(return_value=pubsub_mock)
    
    r.lrange.return_value = []
    # Make pipeline a synchronous MagicMock so it does not return a coroutine
    r.pipeline = MagicMock(return_value=r)
    r.execute.return_value = [True, True]
    
    # Set mock key behaviors
    keys = {}
    async def set_side_effect(key, val, ex=None, nx=False):
        if nx and key in keys:
            return None
        keys[key] = val
        return True
        
    async def get_side_effect(key):
        return keys.get(key)
        
    async def delete_side_effect(key):
        return keys.pop(key, None)
        
    r.set.side_effect = set_side_effect
    r.get.side_effect = get_side_effect
    r.delete.side_effect = delete_side_effect
    return r


# ── Test 1: KV-Cache Affinity Routing ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_kv_cache_affinity_scoring():
    """
    Verify that registering a prefix tree affinity cache in Redis correctly
    gives a score bonus to the target host during routing.
    """
    redis = _mock_redis()
    trie = PrefixTrieRouter(redis)
    
    # 1. Register prompt prefix to 'vllm' host (must be long enough for PREFIX_LENGTHS)
    prompt_text = "This is a very long prompt prefix. " * 150 # 5000+ characters
    await trie.register_host_prefix("vllm", prompt_text)
    
    # Verify that Redis SMEMBERS was called for prefix hashes
    assert redis.sadd.call_count > 0
    
    # Set mock SMEMBERS to return our registered host
    redis.smembers.return_value = [b"vllm"]
    
    # Query affinity hosts
    affinity_hosts = await trie.get_affinity_hosts(prompt_text)
    assert "vllm" in affinity_hosts

    # 2. Test routing with warm affinity
    from inferroute.router import Router
    router = Router(redis)
    
    req = {
        "model": "edge/auto",
        "messages": [{"role": "user", "content": prompt_text}],
        "routing": {"policy": "latency"}
    }
    
    with patch("inferroute.router.get_redis_client", return_value=redis), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as mock_gcb:
        
        cb = AsyncMock()
        cb.allow_request.return_value = True
        mock_gcb.return_value = cb
        
        decision = await router.choose_backend(req)
        # Should prefer 'vllm' due to large warm cache bonus (250ms equivalent)
        assert decision.primary == "vllm"


# ── Test 2: Speculative Cascaded Validation ──────────────────────────────────

@patch("inferroute.adapters.vllm.VLLMAdapter.generate")
@patch("inferroute.adapters.openai.OpenAIAdapter.generate")
def test_speculative_cascade_on_loop_failure(mock_openai_gen, mock_vllm_gen, client):
    """
    Scenario: Speculative routing is active. Primary backend (vllm) returns
    a repetitive generation loop (e.g. 'hello hello hello hello hello').
    The speculative quality validator detects the loop, discards the output,
    and automatically upgrades/cascades to the fallback backend (openai).
    """
    mock_vllm_gen.return_value = {
        "id": "vllm-loop-001",
        "object": "chat.completion",
        "model": "llama-3-8b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello " * 20}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 20, "total_tokens": 25, "estimated_cost_usd": 0.0},
        "timing": {"ttft_ms": 50.0, "latency_ms": 100.0},
    }
    
    # 2. Premium backend returns high quality fallback response
    mock_openai_gen.return_value = {
        "id": "openai-cascade-001",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello! How can I help you today?"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13, "estimated_cost_usd": 0.0002},
        "timing": {"ttft_ms": 150.0, "latency_ms": 200.0},
    }

    redis = _mock_redis()
    with patch("inferroute.auth.redis_client", redis), \
         patch("inferroute.auth.get_redis_client", return_value=redis), \
         patch("inferroute.cache.get_redis_client", return_value=redis), \
         patch("inferroute.router.get_redis_client", return_value=redis), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb, \
         patch("inferroute.main.async_session"):
        
        # Configure gcb to return OPEN for 'ollama' so it doesn't choose ollama over vllm
        def gcb_side_effect(backend):
            mock_cb = AsyncMock()
            if backend == "ollama":
                mock_cb.allow_request.return_value = False
            else:
                mock_cb.allow_request.return_value = True
            mock_cb.record_success = AsyncMock()
            mock_cb.record_failure = AsyncMock()
            mock_cb.get_status.return_value = {"state": "CLOSED"}
            return mock_cb
        gcb.side_effect = gcb_side_effect

        body = CHAT_BODY.copy()
        body["routing"] = {"policy": "speculative"}
        
        resp = client.post("/v1/chat/completions", json=body, headers=BASE_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        
        # Verify it automatically upgraded to OpenAI because vLLM output failed loop quality check
        assert data["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
        assert data["route"]["selected_backend"] == "openai"
        assert data["route"]["fallback_count"] == 1


# ── Test 3: Adaptive Concurrency Limiting ────────────────────────────────────

@pytest.mark.asyncio
async def test_adaptive_concurrency_limiting_backpressure():
    """
    Verify that AdaptiveConcurrencyLimiter properly implements limit boundaries,
    rejects requests when limit is reached, and adapts dynamically under load.
    """
    limiter = AdaptiveConcurrencyLimiter(initial_limit=10, min_limit=2, max_limit=20, alpha=2, beta=5)
    
    # Establish a fast baseline RTT
    await limiter.release(latency_ms=100.0)
    assert limiter.min_latency_ms == 100.0
    
    # 1. Acquire up to limit
    for _ in range(10):
        assert await limiter.acquire() is True
    
    # 2. Next acquisition must be rejected (limit reached)
    assert await limiter.acquire() is False
    
    # 3. Release slots with high RTT (indicates queue forming)
    # The limiter should dynamically adjust the concurrency limit down
    for _ in range(5):
        await limiter.release(latency_ms=1000.0)
    
    # concurr limit should decrease due to high latency queue estimates
    assert limiter.limit < 10


# ── Test 4: Streaming Request Deduplication ───────────────────────────────────

def test_streaming_deduplication_sharing(client):
    """
    Verify that concurrent streaming requests share the active in-flight stream
    via wait_for_stream_dedup instead of invoking the backend multiple times.
    """
    redis = _mock_redis()
    
    # Set try_acquire_dedup_lock to return False (so client 2 is a waiter)
    redis.set.side_effect = lambda key, val, ex=None, nx=False: (None if nx else True)
    
    with patch("inferroute.auth.redis_client", redis), \
         patch("inferroute.auth.get_redis_client", return_value=redis), \
         patch("inferroute.cache.get_redis_client", return_value=redis), \
         patch("inferroute.router.get_redis_client", return_value=redis), \
         patch("inferroute.circuit_breaker.get_circuit_breaker") as gcb:
        
        cb = MagicMock()
        cb.allow_request = AsyncMock(return_value=True)
        gcb.return_value = cb
    
        body = CHAT_BODY.copy()
        body["stream"] = True

        # Simulate joining an active stream
        # This will trigger client 2 paths
        with patch("inferroute.cache.CacheLayer.try_acquire_dedup_lock", return_value=False):
            resp = client.post("/v1/chat/completions", json=body, headers=BASE_HEADERS)
            assert resp.status_code == 200
            
            lines = list(resp.iter_lines())
            assert len(lines) > 0
            
            # Verify stream yields chunks and DONE
            assert any("data: " in line for line in lines)
            assert any("DONE" in line for line in lines)


# ── Test 5: Multi-Tenant Billing & Wallet ─────────────────────────────────────

def test_billing_and_wallet_flow(client):
    """
    Verify get_balance, recharge_wallet, and 402 Payment Required block.
    """
    redis = _mock_redis()
    
    # 1. Mock DB Session for balance lookup
    mock_session = AsyncMock()
    mock_wallet = MagicMock()
    mock_wallet.tenant_id = "acme_corp"
    mock_wallet.balance_usd = 15.0
    
    # We mock execute() scalar result
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = mock_wallet
    mock_session.execute.return_value = execute_result

    with patch("inferroute.auth.redis_client", redis), \
         patch("inferroute.auth.get_redis_client", return_value=redis), \
         patch("inferroute.auth.async_session") as mock_sess_auth, \
         patch("inferroute.main.async_session") as mock_sess_main:
        
        mock_sess_auth.return_value.__aenter__.return_value = mock_session
        mock_sess_main.return_value.__aenter__.return_value = mock_session

        # Test balance retrieval
        resp = client.get("/v1/billing/balance", headers=BASE_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["balance_usd"] == 15.0

        # Test recharge
        mock_wallet.balance_usd = 15.0 # Reset
        resp = client.post("/v1/billing/recharge?amount=10.0", headers=BASE_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["new_balance_usd"] == 25.0

        # Test 402 Payment Required block
        mock_wallet.balance_usd = 0.0 # Dry wallet
        body = CHAT_BODY.copy()
        
        resp = client.post("/v1/chat/completions", json=body, headers=BASE_HEADERS)
        assert resp.status_code == 402
        assert "Payment Required" in resp.json()["detail"]
