"""
Google Gemini adapter for InferRoute.

Uses the google-generativeai SDK. Supports gemini-1.5-flash and gemini-1.5-pro.
Mock mode mirrors the vLLM adapter pattern — no real API calls needed for testing.
"""
import asyncio
import time
import logging
from typing import AsyncGenerator, Any

from inferroute.adapters.base import BaseAdapter
from inferroute.config import settings
from inferroute.observability import tracer, PROVIDER_COST_USD_TOTAL

logger = logging.getLogger("inferroute.adapters.gemini")

# Gemini pricing (USD per 1M tokens) — as of mid-2024
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-1.5-flash": {"input": 0.075 / 1e6, "output": 0.30 / 1e6},
    "gemini-1.5-pro":   {"input": 3.50 / 1e6,  "output": 10.50 / 1e6},
    "gemini-1.0-pro":   {"input": 0.50 / 1e6,  "output": 1.50 / 1e6},
}

# Lazy-import to avoid hard crash if package not installed
_genai = None

def _get_genai():
    global _genai
    if _genai is None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            _genai = genai
        except ImportError:
            logger.warning("google-generativeai not installed. Gemini adapter will use mock mode.")
    return _genai


class GeminiAdapter(BaseAdapter):
    def __init__(self):
        self.model_name = settings.GEMINI_MODEL
        self.mock_mode = settings.MOCK_GEMINI

    def _get_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        prices = MODEL_PRICING.get(model, MODEL_PRICING["gemini-1.5-flash"])
        return prompt_tokens * prices["input"] + completion_tokens * prices["output"]

    def _messages_to_gemini(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert OpenAI-style messages to Gemini content format."""
        system_prompt = ""
        history = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})
        return system_prompt, history

    async def generate(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.mock_mode:
            return await self._mock_generate(req)

        genai = _get_genai()
        if genai is None:
            return await self._mock_generate(req)

        model_name = req.get("model", self.model_name)
        if model_name not in MODEL_PRICING:
            model_name = self.model_name

        system_prompt, history = self._messages_to_gemini(req.get("messages", []))

        with tracer.start_as_current_span("gemini_generate") as span:
            span.set_attribute("llm.model", model_name)
            start_time = time.time()
            try:
                # Run sync SDK call in executor to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt or None
                )
                response = await loop.run_in_executor(
                    None,
                    lambda: model.generate_content(
                        history,
                        generation_config=genai.GenerationConfig(
                            max_output_tokens=req.get("max_output_tokens", 512),
                            temperature=req.get("temperature", 0.7),
                        )
                    )
                )
                latency = time.time() - start_time
                span.set_attribute("llm.latency_seconds", latency)

                prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                completion_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                cost = self._get_cost(model_name, prompt_tokens, completion_tokens)

                PROVIDER_COST_USD_TOTAL.labels(
                    backend="gemini", tenant=req.get("tenant_id", "anonymous")
                ).inc(cost)

                content = response.text if response.text else ""
                return {
                    "id": f"gemini-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
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
                logger.error(f"Gemini completion error: {e}")
                raise

    async def generate_stream(self, req: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        if self.mock_mode:
            async for chunk in self._mock_generate_stream(req):
                yield chunk
            return

        genai = _get_genai()
        if genai is None:
            async for chunk in self._mock_generate_stream(req):
                yield chunk
            return

        model_name = req.get("model", self.model_name)
        if model_name not in MODEL_PRICING:
            model_name = self.model_name

        system_prompt, history = self._messages_to_gemini(req.get("messages", []))
        span = tracer.start_span("gemini_generate_stream")
        span.set_attribute("llm.model", model_name)
        start_time = time.time()
        ttft_recorded = False
        ttft_ms = 0.0
        prompt_tokens = 0
        completion_tokens = 0

        try:
            loop = asyncio.get_event_loop()
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt or None
            )

            # Gemini streaming is synchronous; run in executor and yield chunks
            response_stream = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    history,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=req.get("max_output_tokens", 512),
                        temperature=req.get("temperature", 0.7),
                    ),
                    stream=True,
                )
            )

            chunk_id = f"gemini-stream-{int(time.time())}"
            for chunk in response_stream:
                if not ttft_recorded and chunk.text:
                    ttft_ms = (time.time() - start_time) * 1000.0
                    ttft_recorded = True
                    span.set_attribute("llm.ttft_seconds", ttft_ms / 1000.0)

                if chunk.usage_metadata:
                    prompt_tokens = chunk.usage_metadata.prompt_token_count or prompt_tokens
                    completion_tokens = chunk.usage_metadata.candidates_token_count or completion_tokens

                yield {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": None, "content": chunk.text or ""},
                        "finish_reason": None,
                    }]
                }

            latency_ms = (time.time() - start_time) * 1000.0
            cost = self._get_cost(model_name, prompt_tokens, completion_tokens)
            PROVIDER_COST_USD_TOTAL.labels(
                backend="gemini", tenant=req.get("tenant_id", "anonymous")
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
            logger.error(f"Gemini streaming error: {e}")
            raise

    # ── Mock implementations ─────────────────────────────────────────────────

    async def _mock_generate(self, req: dict[str, Any]) -> dict[str, Any]:
        """Simulates Gemini non-streaming with realistic latency."""
        with tracer.start_as_current_span("gemini_mock_generate") as span:
            model_name = req.get("model", self.model_name)
            span.set_attribute("llm.model", model_name)

            prompt_len = sum(len(m.get("content", "")) for m in req.get("messages", []))
            prompt_tokens = prompt_len // 4 + 5
            completion_tokens = 75

            await asyncio.sleep(0.25)  # ~250ms mock TTFT

            cost = self._get_cost(model_name, prompt_tokens, completion_tokens)
            PROVIDER_COST_USD_TOTAL.labels(
                backend="gemini", tenant=req.get("tenant_id", "anonymous")
            ).inc(cost)

            content = "This is a mock response from Google Gemini. The model is running in simulation mode."
            if "response_format" in req and req["response_format"].get("type") == "json_schema":
                content = '{"invoice_id": "GEMINI-MOCK-001", "amount": 55.00}'

            return {
                "id": f"gemini-mock-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "estimated_cost_usd": cost,
                },
                "timing": {"ttft_ms": 250.0, "latency_ms": 250.0},
            }

    async def _mock_generate_stream(self, req: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Simulates Gemini streaming token generation."""
        span = tracer.start_span("gemini_mock_generate_stream")
        model_name = req.get("model", self.model_name)
        span.set_attribute("llm.model", model_name)

        prompt_len = sum(len(m.get("content", "")) for m in req.get("messages", []))
        prompt_tokens = prompt_len // 4 + 5

        content = "This is a streaming mock from Gemini, confirming multi-provider routing and fallback paths."
        if "response_format" in req and req["response_format"].get("type") == "json_schema":
            content = '{"invoice_id": "GEMINI-STREAM-001", "amount": 77.50}'

        words = content.split(" ")
        start_time = time.time()
        await asyncio.sleep(0.10)  # 100ms TTFT
        ttft_ms = (time.time() - start_time) * 1000.0
        span.set_attribute("llm.ttft_seconds", ttft_ms / 1000.0)

        chunk_id = f"gemini-stream-mock-{int(time.time())}"
        yield {
            "id": chunk_id, "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
        }

        for i, word in enumerate(words):
            await asyncio.sleep(0.015)
            yield {
                "id": chunk_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model_name,
                "choices": [{"index": 0, "delta": {"role": None, "content": word if i == 0 else " " + word}, "finish_reason": None}]
            }

        yield {
            "id": chunk_id, "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model_name,
            "choices": [{"index": 0, "delta": {"role": None, "content": None}, "finish_reason": "stop"}]
        }

        latency_ms = (time.time() - start_time) * 1000.0
        completion_tokens = len(words) * 2
        cost = self._get_cost(model_name, prompt_tokens, completion_tokens)
        PROVIDER_COST_USD_TOTAL.labels(
            backend="gemini", tenant=req.get("tenant_id", "anonymous")
        ).inc(cost)

        yield {
            "id": chunk_id, "object": "chat.completion.chunk", "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens, "estimated_cost_usd": cost,
            },
            "timing": {"ttft_ms": ttft_ms, "latency_ms": latency_ms},
        }
        span.end()
