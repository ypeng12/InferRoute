"""
Adapter unit tests for Gemini and Ollama (mock mode).

Tests:
  - GeminiAdapter.generate() mock returns correct shape
  - GeminiAdapter.generate_stream() mock yields correct chunks
  - OllamaAdapter.generate() mock returns correct shape
  - OllamaAdapter.generate_stream() mock yields correct chunks
  - Both adapters handle json_schema response_format
  - Cost calculation is correct per model
"""
import asyncio
import pytest
from inferroute.adapters.gemini import GeminiAdapter, MODEL_PRICING
from inferroute.adapters.ollama import OllamaAdapter


SIMPLE_REQ = {
    "model": "gemini-1.5-flash",
    "messages": [{"role": "user", "content": "Hello, world!"}],
    "tenant_id": "test_tenant",
}

JSON_SCHEMA_REQ = {
    "model": "gemini-1.5-flash",
    "messages": [{"role": "user", "content": "Extract invoice fields"}],
    "tenant_id": "test_tenant",
    "response_format": {"type": "json_schema", "schema": {"type": "object"}},
}


# ── Gemini Adapter Tests ──────────────────────────────────────────────────────

class TestGeminiAdapterMock:

    @pytest.fixture
    def adapter(self):
        a = GeminiAdapter()
        a.mock_mode = True
        return a

    @pytest.mark.asyncio
    async def test_generate_returns_valid_shape(self, adapter):
        resp = await adapter.generate(SIMPLE_REQ)
        assert resp["object"] == "chat.completion"
        assert len(resp["choices"]) == 1
        assert resp["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(resp["choices"][0]["message"]["content"], str)
        assert "usage" in resp
        assert resp["usage"]["total_tokens"] > 0
        assert "timing" in resp
        assert resp["timing"]["ttft_ms"] > 0

    @pytest.mark.asyncio
    async def test_generate_stream_yields_chunks(self, adapter):
        chunks = []
        async for chunk in adapter.generate_stream(SIMPLE_REQ):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Last chunk should be the stats chunk (no choices, has usage)
        stats_chunk = chunks[-1]
        assert "usage" in stats_chunk
        assert "timing" in stats_chunk
        assert stats_chunk["usage"]["total_tokens"] > 0

        # At least one chunk should have content
        content_chunks = [
            c for c in chunks
            if c.get("choices") and c["choices"][0].get("delta", {}).get("content")
        ]
        assert len(content_chunks) > 0

    @pytest.mark.asyncio
    async def test_generate_json_schema_format(self, adapter):
        resp = await adapter.generate(JSON_SCHEMA_REQ)
        content = resp["choices"][0]["message"]["content"]
        assert "invoice_id" in content or "GEMINI" in content

    @pytest.mark.asyncio
    async def test_generate_stream_json_schema_format(self, adapter):
        all_content = []
        async for chunk in adapter.generate_stream(JSON_SCHEMA_REQ):
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    all_content.append(delta["content"])
        full = "".join(all_content)
        assert "GEMINI" in full or "invoice_id" in full

    def test_cost_calculation_flash(self, adapter):
        cost = adapter._get_cost("gemini-1.5-flash", 1000, 500)
        expected = 1000 * MODEL_PRICING["gemini-1.5-flash"]["input"] + 500 * MODEL_PRICING["gemini-1.5-flash"]["output"]
        assert abs(cost - expected) < 1e-10

    def test_cost_calculation_pro(self, adapter):
        cost = adapter._get_cost("gemini-1.5-pro", 1000, 500)
        expected = 1000 * MODEL_PRICING["gemini-1.5-pro"]["input"] + 500 * MODEL_PRICING["gemini-1.5-pro"]["output"]
        assert abs(cost - expected) < 1e-10
        # Pro should be more expensive than Flash
        flash_cost = adapter._get_cost("gemini-1.5-flash", 1000, 500)
        assert cost > flash_cost


# ── Ollama Adapter Tests ──────────────────────────────────────────────────────

OLLAMA_REQ = {
    "model": "llama3",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "tenant_id": "test_tenant",
}

OLLAMA_JSON_REQ = {
    "model": "llama3",
    "messages": [{"role": "user", "content": "Parse this invoice"}],
    "tenant_id": "test_tenant",
    "response_format": {"type": "json_schema", "schema": {"type": "object"}},
}


class TestOllamaAdapterMock:

    @pytest.fixture
    def adapter(self):
        a = OllamaAdapter()
        a.mock_mode = True
        return a

    @pytest.mark.asyncio
    async def test_generate_returns_valid_shape(self, adapter):
        resp = await adapter.generate(OLLAMA_REQ)
        assert resp["object"] == "chat.completion"
        assert len(resp["choices"]) == 1
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert resp["usage"]["estimated_cost_usd"] == 0.0  # local = free

    @pytest.mark.asyncio
    async def test_generate_stream_yields_chunks(self, adapter):
        chunks = []
        async for chunk in adapter.generate_stream(OLLAMA_REQ):
            chunks.append(chunk)
        assert len(chunks) > 0
        stats = chunks[-1]
        assert stats["usage"]["estimated_cost_usd"] == 0.0
        assert "timing" in stats

    @pytest.mark.asyncio
    async def test_generate_json_schema_format(self, adapter):
        resp = await adapter.generate(OLLAMA_JSON_REQ)
        content = resp["choices"][0]["message"]["content"]
        assert "OLLAMA" in content or "invoice_id" in content

    @pytest.mark.asyncio
    async def test_zero_cost(self, adapter):
        cost = adapter._get_cost(100, 200)
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_stream_ttft_faster_than_openai(self, adapter):
        """Ollama mock should have faster TTFT than OpenAI's 250ms mock."""
        import time
        start = time.time()
        async for _ in adapter.generate_stream(OLLAMA_REQ):
            break
        elapsed = (time.time() - start) * 1000.0
        # Ollama TTFT is ~80ms mock delay — should be well under OpenAI's 250ms
        assert elapsed < 250.0
