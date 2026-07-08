import time
import logging
from typing import AsyncGenerator, Any
from openai import AsyncOpenAI
from opentelemetry import trace
from inferroute.adapters.base import BaseAdapter
from inferroute.config import settings
from inferroute.observability import tracer, PROVIDER_COST_USD_TOTAL

logger = logging.getLogger("inferroute.adapters.openai")

MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15 / 1e6, "cached_input": 0.075 / 1e6, "output": 0.60 / 1e6},
    "gpt-4o": {"input": 5.0 / 1e6, "cached_input": 2.5 / 1e6, "output": 15.0 / 1e6},
    "gpt-3.5-turbo": {"input": 0.50 / 1e6, "cached_input": 0.50 / 1e6, "output": 1.50 / 1e6},
}

class OpenAIAdapter(BaseAdapter):
    def __init__(self):
        self.mock_mode = settings.MOCK_OPENAI
        if self.mock_mode:
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    def _get_cost(self, model: str, prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int = 0) -> float:
        prices = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
        cost = (
            (prompt_tokens - cached_prompt_tokens) * prices["input"]
            + cached_prompt_tokens * prices["cached_input"]
            + completion_tokens * prices["output"]
        )
        return cost

    async def generate(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.mock_mode:
            return await self._mock_generate(req)
            
        model = req.get("model", "gpt-4o-mini")
        if model not in MODEL_PRICING:
            # Fallback to mini pricing if custom model name
            model = "gpt-4o-mini"
            
        params = {
            "model": model,
            "messages": req["messages"],
            "temperature": req.get("temperature", 0.7),
            "max_tokens": req.get("max_output_tokens", 512),
        }
        if "response_format" in req:
            params["response_format"] = req["response_format"]

        with tracer.start_as_current_span("openai_generate") as span:
            span.set_attribute("llm.model", model)
            start_time = time.time()
            
            try:
                response = await self.client.chat.completions.create(**params)
                latency = time.time() - start_time
                span.set_attribute("llm.latency_seconds", latency)
                
                # Extract usage
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                cached_prompt_tokens = getattr(response.usage, "prompt_tokens_details", None)
                cached_tokens = 0
                if cached_prompt_tokens and hasattr(cached_prompt_tokens, "cached_tokens"):
                    cached_tokens = cached_prompt_tokens.cached_tokens
                
                cost = self._get_cost(model, prompt_tokens, completion_tokens, cached_tokens)
                
                # Accumulate cost metric
                PROVIDER_COST_USD_TOTAL.labels(backend="openai", tenant=req.get("tenant_id", "anonymous")).inc(cost)
                
                # Map to standard return format
                mapped_response = {
                    "id": response.id,
                    "object": "chat.completion",
                    "created": response.created,
                    "model": response.model,
                    "choices": [
                        {
                            "index": c.index,
                            "message": {
                                "role": c.message.role,
                                "content": c.message.content,
                            },
                            "finish_reason": c.finish_reason,
                        }
                        for c in response.choices
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "cached_prompt_tokens": cached_tokens,
                        "estimated_cost_usd": cost,
                    },
                    "timing": {
                        "ttft_ms": latency * 1000.0, # non-streaming ttft equals total latency
                        "latency_ms": latency * 1000.0,
                    }
                }
                return mapped_response
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                logger.error(f"OpenAI completion error: {e}")
                raise

    async def generate_stream(self, req: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        if self.mock_mode:
            async for chunk in self._mock_generate_stream(req):
                yield chunk
            return
            
        model = req.get("model", "gpt-4o-mini")
        if model not in MODEL_PRICING:
            model = "gpt-4o-mini"
            
        params = {
            "model": model,
            "messages": req["messages"],
            "temperature": req.get("temperature", 0.7),
            "max_tokens": req.get("max_output_tokens", 512),
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        if "response_format" in req:
            params["response_format"] = req["response_format"]

        span = tracer.start_span("openai_generate_stream")
        span.set_attribute("llm.model", model)
        
        start_time = time.time()
        ttft_recorded = False
        ttft_ms = 0.0
        
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        
        try:
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                if not ttft_recorded and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    ttft_ms = (time.time() - start_time) * 1000.0
                    ttft_recorded = True
                    span.set_attribute("llm.ttft_seconds", ttft_ms / 1000.0)
                
                # Check for usage at the end of the stream
                if hasattr(chunk, "usage") and chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    cached_prompt_tokens = getattr(chunk.usage, "prompt_tokens_details", None)
                    if cached_prompt_tokens and hasattr(cached_prompt_tokens, "cached_tokens"):
                        cached_tokens = cached_prompt_tokens.cached_tokens
                
                # Map chunk to standard format
                mapped_chunk = {
                    "id": chunk.id,
                    "object": "chat.completion.chunk",
                    "created": chunk.created,
                    "model": chunk.model,
                    "choices": [
                        {
                            "index": c.index,
                            "delta": {
                                "role": getattr(c.delta, "role", None),
                                "content": getattr(c.delta, "content", None),
                            },
                            "finish_reason": c.finish_reason,
                        }
                        for c in chunk.choices
                    ]
                }
                yield mapped_chunk
                
            latency_ms = (time.time() - start_time) * 1000.0
            span.set_attribute("llm.latency_seconds", latency_ms / 1000.0)
            
            # If usage details are missing, estimate them based on stream chunks
            if prompt_tokens == 0:
                # Fallback estimation if stream_options didn't provide usage
                prompt_tokens = sum(len(m["content"]) for m in req["messages"]) // 4 + 10
                completion_tokens = int(latency_ms / 50.0) # mock output tokens based on avg time
            
            cost = self._get_cost(model, prompt_tokens, completion_tokens, cached_tokens)
            PROVIDER_COST_USD_TOTAL.labels(backend="openai", tenant=req.get("tenant_id", "anonymous")).inc(cost)
            
            # Yield final statistics block
            stats_chunk = {
                "id": chunk.id if 'chunk' in locals() else "stream-end",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "cached_prompt_tokens": cached_tokens,
                    "estimated_cost_usd": cost,
                },
                "timing": {
                    "ttft_ms": ttft_ms if ttft_recorded else latency_ms,
                    "latency_ms": latency_ms,
                }
            }
            yield stats_chunk
            span.end()
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.end()
            logger.error(f"OpenAI streaming error: {e}")
            raise

    async def _mock_generate(self, req: dict[str, Any]) -> dict[str, Any]:
        from inferroute.adapters.mock_generator import generate_mock_reply
        import asyncio
        model = req.get("model", "gpt-4o-mini")
        prompt_len = sum(len(m.get("content", "")) for m in req.get("messages", []))
        prompt_tokens = prompt_len // 4 + 5
        completion_tokens = 60
        
        await asyncio.sleep(0.25)  # simulated 250ms cloud latency
        messages = req.get("messages", [])
        prompt = messages[-1].get("content", "") if messages else ""
        content = generate_mock_reply(prompt, "openai")
        cost = self._get_cost(model, prompt_tokens, completion_tokens, 0)
        
        return {
            "id": f"openai-mock-{int(time.time())}",
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
                "cached_prompt_tokens": 0,
                "estimated_cost_usd": cost,
            },
            "timing": {"ttft_ms": 250.0, "latency_ms": 250.0},
        }

    async def _mock_generate_stream(self, req: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        from inferroute.adapters.mock_generator import generate_mock_reply
        import asyncio
        model = req.get("model", "gpt-4o-mini")
        prompt_len = sum(len(m.get("content", "")) for m in req.get("messages", []))
        prompt_tokens = prompt_len // 4 + 5
        
        messages = req.get("messages", [])
        prompt = messages[-1].get("content", "") if messages else ""
        content = generate_mock_reply(prompt, "openai")
        words = content.split(" ")
        
        start_time = time.time()
        await asyncio.sleep(0.12)  # 120ms cloud TTFT
        ttft_ms = (time.time() - start_time) * 1000.0
        
        chunk_id = f"openai-stream-mock-{int(time.time())}"
        yield {
            "id": chunk_id, "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
        }
        
        for i, word in enumerate(words):
            await asyncio.sleep(0.015)  # 15ms delay per token
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
        cost = self._get_cost(model, prompt_tokens, completion_tokens, 0)
        
        yield {
            "id": chunk_id, "object": "chat.completion.chunk", "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens, "cached_prompt_tokens": 0,
                "estimated_cost_usd": cost,
            },
            "timing": {"ttft_ms": ttft_ms, "latency_ms": latency_ms},
        }
