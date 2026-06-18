"""
Ollama adapter for InferRoute.

Uses httpx against the local Ollama server (http://localhost:11434).
Supports the Ollama /api/chat endpoint and the OpenAI-compatible /v1/chat/completions endpoint.
Zero-cost (local GPU/CPU) with mock mode for testing.
"""
import asyncio
import json
import time
import logging
from typing import AsyncGenerator, Any

import httpx

from inferroute.adapters.base import BaseAdapter
from inferroute.config import settings
from inferroute.observability import tracer, PROVIDER_COST_USD_TOTAL

logger = logging.getLogger("inferroute.adapters.ollama")

# Local inference: cost is essentially amortized hardware — model as near-zero
LOCAL_COST_PER_TOKEN = 0.0  # truly free for local inference


class OllamaAdapter(BaseAdapter):
    def __init__(self):
        self.api_url = settings.OLLAMA_API_URL.rstrip("/")
        self.model = settings.OLLAMA_MODEL
        self.mock_mode = settings.MOCK_OLLAMA
        self.client = httpx.AsyncClient(timeout=60.0)

    def _get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # Local inference is free (amortized hardware cost treated as 0 for routing)
        return 0.0

    async def generate(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.mock_mode:
            return await self._mock_generate(req)

        model = req.get("model", self.model)
        # Strip provider prefix if present (e.g. "ollama/llama3" → "llama3")
        if "/" in model and not model.startswith("meta-llama"):
            model = model.split("/", 1)[1]

        with tracer.start_as_current_span("ollama_generate") as span:
            span.set_attribute("llm.model", model)
            start_time = time.time()
            try:
                payload = {
                    "model": model,
                    "messages": req.get("messages", []),
                    "stream": False,
                    "options": {
                        "temperature": req.get("temperature", 0.7),
                        "num_predict": req.get("max_output_tokens", 512),
                    }
                }
                response = await self.client.post(
                    f"{self.api_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                latency = time.time() - start_time
                span.set_attribute("llm.latency_seconds", latency)

                message = data.get("message", {})
                content = message.get("content", "")
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                cost = self._get_cost(prompt_tokens, completion_tokens)

                PROVIDER_COST_USD_TOTAL.labels(
                    backend="ollama", tenant=req.get("tenant_id", "anonymous")
                ).inc(cost)

                return {
                    "id": f"ollama-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": data.get("model", model),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": data.get("done_reason", "stop"),
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "estimated_cost_usd": cost,
                    },
                    "timing": {
                        "ttft_ms": latency * 1000.0,
                        "latency_ms": latency * 1000.0,
                    }
                }
            except Exception as e:
                span.record_exception(e)
                logger.error(f"Ollama generate error: {e}")
                raise

    async def generate_stream(self, req: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        if self.mock_mode:
            async for chunk in self._mock_generate_stream(req):
                yield chunk
            return

        model = req.get("model", self.model)
        if "/" in model and not model.startswith("meta-llama"):
            model = model.split("/", 1)[1]

        span = tracer.start_span("ollama_generate_stream")
        span.set_attribute("llm.model", model)
        start_time = time.time()
        ttft_recorded = False
        ttft_ms = 0.0
        prompt_tokens = 0
        completion_tokens = 0

        try:
            payload = {
                "model": model,
                "messages": req.get("messages", []),
                "stream": True,
                "options": {
                    "temperature": req.get("temperature", 0.7),
                    "num_predict": req.get("max_output_tokens", 512),
                }
            }
            chunk_id = f"ollama-stream-{int(time.time())}"

            async with self.client.stream("POST", f"{self.api_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    message = data.get("message", {})
                    content = message.get("content", "")

                    if content and not ttft_recorded:
                        ttft_ms = (time.time() - start_time) * 1000.0
                        ttft_recorded = True
                        span.set_attribute("llm.ttft_seconds", ttft_ms / 1000.0)

                    # Ollama provides final stats on the last chunk (done=True)
                    if data.get("done"):
                        prompt_tokens = data.get("prompt_eval_count", 0)
                        completion_tokens = data.get("eval_count", 0)
                        break

                    yield {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": None, "content": content},
                            "finish_reason": None,
                        }]
                    }

            latency_ms = (time.time() - start_time) * 1000.0
            cost = self._get_cost(prompt_tokens, completion_tokens)
            PROVIDER_COST_USD_TOTAL.labels(
                backend="ollama", tenant=req.get("tenant_id", "anonymous")
            ).inc(cost)

            yield {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "estimated_cost_usd": cost,
                },
                "timing": {
                    "ttft_ms": ttft_ms if ttft_recorded else latency_ms,
                    "latency_ms": latency_ms,
                }
            }
            span.end()

        except Exception as e:
            span.record_exception(e)
            span.end()
            logger.error(f"Ollama streaming error: {e}")
            raise

    # ── Mock implementations ─────────────────────────────────────────────────

    async def _mock_generate(self, req: dict[str, Any]) -> dict[str, Any]:
        """Simulates Ollama with realistic local-model latency (fast, zero cost)."""
        with tracer.start_as_current_span("ollama_mock_generate") as span:
            model = req.get("model", self.model)
            span.set_attribute("llm.model", model)

            prompt_len = sum(len(m.get("content", "")) for m in req.get("messages", []))
            prompt_tokens = prompt_len // 4 + 5
            completion_tokens = 85

            await asyncio.sleep(0.18)  # local model: ~180ms

            content = "This is a mock response from local Ollama. Running llama3 in simulation mode — fast, free, and private."
            if "response_format" in req and req["response_format"].get("type") == "json_schema":
                content = '{"invoice_id": "OLLAMA-MOCK-001", "amount": 12.50}'

            return {
                "id": f"ollama-mock-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "estimated_cost_usd": 0.0,
                },
                "timing": {"ttft_ms": 180.0, "latency_ms": 180.0},
            }

    async def _mock_generate_stream(self, req: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Simulates Ollama streaming — fast inter-token delay, zero cost."""
        span = tracer.start_span("ollama_mock_generate_stream")
        model = req.get("model", self.model)
        span.set_attribute("llm.model", model)

        prompt_len = sum(len(m.get("content", "")) for m in req.get("messages", []))
        prompt_tokens = prompt_len // 4 + 5

        content = "This is a streaming mock from local Ollama. Confirming local-first routing, zero cost, and fast TTFT."
        if "response_format" in req and req["response_format"].get("type") == "json_schema":
            content = '{"invoice_id": "OLLAMA-STREAM-001", "amount": 8.00}'

        words = content.split(" ")
        start_time = time.time()
        await asyncio.sleep(0.08)  # 80ms TTFT — local model is fast
        ttft_ms = (time.time() - start_time) * 1000.0
        span.set_attribute("llm.ttft_seconds", ttft_ms / 1000.0)

        chunk_id = f"ollama-stream-mock-{int(time.time())}"
        yield {
            "id": chunk_id, "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
        }

        for i, word in enumerate(words):
            await asyncio.sleep(0.012)  # 12ms — local inference is fast
            yield {
                "id": chunk_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "delta": {"role": None, "content": word if i == 0 else " " + word}, "finish_reason": None}]
            }

        yield {
            "id": chunk_id, "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "delta": {"role": None, "content": None}, "finish_reason": "stop"}]
        }

        latency_ms = (time.time() - start_time) * 1000.0
        completion_tokens = len(words) * 2
        span.set_attribute("llm.latency_seconds", latency_ms / 1000.0)

        yield {
            "id": chunk_id, "object": "chat.completion.chunk", "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens, "estimated_cost_usd": 0.0,
            },
            "timing": {"ttft_ms": ttft_ms, "latency_ms": latency_ms},
        }
        span.end()
