import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from inferroute.main import app
from inferroute.router import Router
from inferroute.validator import OutputValidator
from inferroute.cache import CacheLayer

client = TestClient(app)

@pytest.fixture
def mock_redis():
    """Mocks redis client interactions."""
    with patch("inferroute.auth.get_redis_client") as mock_get_auth, \
         patch("inferroute.cache.get_redis_client") as mock_get_cache, \
         patch("inferroute.router.get_redis_client") as mock_get_router:
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None
        redis_mock.incr.return_value = 1
        redis_mock.expire.return_value = True
        mock_get_auth.return_value = redis_mock
        mock_get_cache.return_value = redis_mock
        mock_get_router.return_value = redis_mock
        yield redis_mock

@pytest.fixture
def mock_db():
    """Mocks database session logging interactions."""
    with patch("inferroute.main.async_session") as mock_sess:
        session_mock = AsyncMock()
        session_mock.add = MagicMock()
        mock_sess.return_value.__aenter__.return_value = session_mock
        yield session_mock


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_router_hard_pins():
    router = Router()
    
    # Test OpenAI pinning
    primary, fallback, reason = await router.choose_backend({"model": "gpt-4o-mini"})
    assert primary == "openai"
    assert fallback is None
    assert "gpt-4o-mini" in reason

    # Test vLLM pinning
    primary, fallback, reason = await router.choose_backend({"model": "meta-llama/Meta-Llama-3-8B-Instruct"})
    assert primary == "vllm"
    assert fallback == "openai"


@pytest.mark.asyncio
async def test_router_optimization():
    router = Router()
    
    # Latency preference (default)
    req = {
        "model": "edge/auto",
        "messages": [{"role": "user", "content": "hello"}],
        "routing": {"allow_local": True, "allow_cloud": True, "policy": "latency"}
    }
    
    # Because baselines set local (vllm) score higher due to caches, it will choose vllm
    primary, fallback, reason = await router.choose_backend(req)
    assert primary in ["vllm", "openai"]
    assert fallback is not None


def test_validator_schema():
    validator = OutputValidator()
    
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    
    # Correct structure
    res = validator.validate_schema('{"name": "Alice", "age": 30}', schema)
    assert res.ok
    assert res.reason is None
    
    # Bad JSON structure
    res2 = validator.validate_schema('{"name": "Alice", "age": 30', schema)
    assert not res2.ok
    assert "JSON decode error" in res2.reason
    
    # Schema violation
    res3 = validator.validate_schema('{"name": "Alice", "age": "thirty"}', schema)
    assert not res3.ok
    assert "validation error" in res3.reason.lower()


@patch("inferroute.adapters.vllm.VLLMAdapter.generate")
@patch("inferroute.adapters.openai.OpenAIAdapter.generate")
def test_gateway_chat_completion_blocking(mock_openai_gen, mock_vllm_gen, mock_redis, mock_db):
    """
    Tests standard chat completions endpoint. 
    Verifies authentication, routing decisions, execution, and DB logging.
    """
    # Setup mocks
    mock_vllm_gen.return_value = {
        "id": "mock-vllm-123",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi there!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "estimated_cost_usd": 0.0001},
        "timing": {"ttft_ms": 100.0, "latency_ms": 150.0}
    }
    
    # Call endpoint without key - unauthorized
    response = client.post(
        "/v1/chat/completions",
        json={"model": "edge/auto", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 401
    
    # Call with key
    headers = {"Authorization": "Bearer sk-inferroute-demo"}
    response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "user", "content": "hi"}]}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hi there!"
    assert data["route"]["selected_backend"] == "vllm"
    assert data["route"]["cache_hit"] is False
    
    # Verify cache lookup was executed
    assert mock_redis.get.called
    
    # Verify mock_vllm_gen was called
    assert mock_vllm_gen.called


@patch("inferroute.adapters.vllm.VLLMAdapter.generate")
@patch("inferroute.adapters.openai.OpenAIAdapter.generate")
def test_gateway_chat_completion_fallback(mock_openai_gen, mock_vllm_gen, mock_redis, mock_db):
    """
    Tests fallback logic.
    Primary backend (vllm) fails, Gateway should fallback to OpenAI.
    """
    # Setup mocks: primary vllm raises exception, fallback openai succeeds
    mock_vllm_gen.side_effect = Exception("GPU OOM / Timeout")
    mock_openai_gen.return_value = {
        "id": "mock-openai-999",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from Cloud!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18, "estimated_cost_usd": 0.0002},
        "timing": {"ttft_ms": 200.0, "latency_ms": 250.0}
    }
    
    headers = {"Authorization": "Bearer sk-inferroute-demo"}
    response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "user", "content": "hi"}]}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello from Cloud!"
    assert data["route"]["selected_backend"] == "openai"
    assert data["route"]["fallback_count"] == 1
    
    # Verify both were called
    assert mock_vllm_gen.called
    assert mock_openai_gen.called
