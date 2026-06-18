import asyncio
import time
import logging
from typing import AsyncGenerator, Any
import httpx
from inferroute.adapters.base import BaseAdapter
from inferroute.config import settings
from inferroute.observability import tracer, PROVIDER_COST_USD_TOTAL

logger = logging.getLogger("inferroute.adapters.vllm")

# Local GPU/CPU amortized cost estimate per token
LOCAL_INPUT_COST = 0.000005 / 1000  # $0.005 per 1K tokens
LOCAL_OUTPUT_COST = 0.000015 / 1000 # $0.015 per 1K tokens

class VLLMAdapter(BaseAdapter):
    def __init__(self):
        self.api_url = settings.VLLM_API_URL
        self.api_key = settings.VLLM_API_KEY
        self.mock_mode = settings.MOCK_VLLM
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )

    def _get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # Local model cost is derived from hardware amortization estimates
        return (prompt_tokens * LOCAL_INPUT_COST) + (completion_tokens * LOCAL_OUTPUT_COST)

    async def generate(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.mock_mode:
            return await self._mock_generate(req)
            
        model = req.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
        params = {
            "model": model,
            "messages": req["messages"],
            "temperature": req.get("temperature", 0.7),
            "max_tokens": req.get("max_output_tokens", 512),
            "stream": False
        }
        if "response_format" in req:
            params["response_format"] = req["response_format"]

        with tracer.start_as_current_span("vllm_generate") as span:
            span.set_attribute("llm.model", model)
            start_time = time.time()
            try:
                response = await self.client.post(
                    f"{self.api_url}/chat/completions",
                    json=params,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                latency = time.time() - start_time
                span.set_attribute("llm.latency_seconds", latency)
                
                # Extract usage and cost
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                cost = self._get_cost(prompt_tokens, completion_tokens)
                
                PROVIDER_COST_USD_TOTAL.labels(backend="vllm", tenant=req.get("tenant_id", "anonymous")).inc(cost)
                
                # Map to consistent output
                choices = data.get("choices", [])
                mapped_choices = []
                for idx, c in enumerate(choices):
                    msg = c.get("message", {})
                    mapped_choices.append({
                        "index": c.get("index", idx),
                        "message": {
                            "role": msg.get("role", "assistant"),
                            "content": msg.get("content", ""),
                        },
                        "finish_reason": c.get("finish_reason", "stop")
                    })
                
                return {
                    "id": data.get("id", "vllm-completion"),
                    "object": "chat.completion",
                    "created": data.get("created", int(time.time())),
                    "model": data.get("model", model),
                    "choices": mapped_choices,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "cached_prompt_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                        "estimated_cost_usd": cost,
                    },
                    "timing": {
                        "ttft_ms": latency * 1000.0,
                        "latency_ms": latency * 1000.0,
                    }
                }
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                logger.error(f"vLLM API HTTP request failed: {e}")
                raise

    async def generate_stream(self, req: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        if self.mock_mode:
            async for chunk in self._mock_generate_stream(req):
                yield chunk
            return

        model = req.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
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

        span = tracer.start_span("vllm_generate_stream")
        span.set_attribute("llm.model", model)
        
        start_time = time.time()
        ttft_recorded = False
        ttft_ms = 0.0
        
        prompt_tokens = 0
        completion_tokens = 0
        
        try:
            # We use a custom event stream reader
            async with self.client.stream("POST", f"{self.api_url}/chat/completions", json=params, timeout=30.0) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        break
                    if line.startswith("data: "):
                        import json
                        try:
                            chunk_data = json.loads(line[6:])
                            choices = chunk_data.get("choices", [])
                            
                            if not ttft_recorded and len(choices) > 0 and choices[0].get("delta", {}).get("content"):
                                ttft_ms = (time.time() - start_time) * 1000.0
                                ttft_recorded = True
                                span.set_attribute("llm.ttft_seconds", ttft_ms / 1000.0)
                            
                            usage = chunk_data.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                            
                            # Standardize chunk
                            mapped_choices = []
                            for idx, c in enumerate(choices):
                                delta = c.get("delta", {})
                                mapped_choices.append({
                                    "index": c.get("index", idx),
                                    "delta": {
                                        "role": delta.get("role"),
                                        "content": delta.get("content"),
                                    },
                                    "finish_reason": c.get("finish_reason")
                                })
                                
                            mapped_chunk = {
                                "id": chunk_data.get("id"),
                                "object": "chat.completion.chunk",
                                "created": chunk_data.get("created"),
                                "model": chunk_data.get("model", model),
                                "choices": mapped_choices
                            }
                            yield mapped_chunk
                        except Exception as e:
                            logger.error(f"Error parsing vLLM stream line: {line}. Error: {e}")
                            
            latency_ms = (time.time() - start_time) * 1000.0
            span.set_attribute("llm.latency_seconds", latency_ms / 1000.0)
            
            if prompt_tokens == 0:
                prompt_tokens = sum(len(m["content"]) for m in req["messages"]) // 4 + 10
                completion_tokens = int(latency_ms / 30.0)
                
            cost = self._get_cost(prompt_tokens, completion_tokens)
            PROVIDER_COST_USD_TOTAL.labels(backend="vllm", tenant=req.get("tenant_id", "anonymous")).inc(cost)
            
            stats_chunk = {
                "id": "vllm-stream-end",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "cached_prompt_tokens": 0,
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
            logger.error(f"vLLM streaming error: {e}")
            raise

    # --- MOCK MODE IMPLEMENTATIONS ---
    
    async def _mock_generate(self, req: dict[str, Any]) -> dict[str, Any]:
        """Simulates non-streaming generation with mock latency."""
        with tracer.start_as_current_span("vllm_mock_generate") as span:
            model = req.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
            span.set_attribute("llm.model", model)
            
            # Simulate processing delay: ~150ms TTFT + ~15ms per token for 100 tokens -> ~1.65 seconds
            prompt_len = sum(len(m["content"]) for m in req["messages"])
            prompt_tokens = prompt_len // 4 + 5
            completion_tokens = 80
            
            # Simulated delay
            await asyncio.sleep(0.3) 
            
            cost = self._get_cost(prompt_tokens, completion_tokens)
            PROVIDER_COST_USD_TOTAL.labels(backend="vllm", tenant=req.get("tenant_id", "anonymous")).inc(cost)
            
            content = "This is a mock response from local vLLM. It matches your structural design patterns and is fully functional."
            if "response_format" in req and req["response_format"].get("type") == "json_schema":
                content = '{"invoice_id": "MOCK-12345", "amount": 99.99}'
                
            return {
                "id": "mock-vllm-completion-123",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "cached_prompt_tokens": 0,
                    "estimated_cost_usd": cost,
                },
                "timing": {
                    "ttft_ms": 150.0,
                    "latency_ms": 300.0,
                }
            }

    async def _mock_generate_stream(self, req: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Simulates streaming token generation with mock interval delays."""
        span = tracer.start_span("vllm_mock_generate_stream")
        model = req.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
        span.set_attribute("llm.model", model)
        
        prompt_len = sum(len(m["content"]) for m in req["messages"])
        prompt_tokens = prompt_len // 4 + 5
        
        # Stream chunks simulation
        content = "This is a streaming mock response from local vLLM. It confirms execution paths and fallback routes."
        if "response_format" in req and req["response_format"].get("type") == "json_schema":
            content = '{"invoice_id": "MOCK-STREAM-99", "amount": 42.50}'
            
        words = content.split(" ")
        
        # 1. Simulate TTFT (initial wait)
        start_time = time.time()
        await asyncio.sleep(0.12) # 120ms wait
        ttft_ms = (time.time() - start_time) * 1000.0
        span.set_attribute("llm.ttft_seconds", ttft_ms / 1000.0)
        
        # Send first delta block
        yield {
            "id": "mock-vllm-stream-chunk",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None
            }]
        }
        
        # 2. Stream word by word
        for i, word in enumerate(words):
            await asyncio.sleep(0.02) # 20ms between tokens
            chunk_content = word if i == 0 else " " + word
            yield {
                "id": "mock-vllm-stream-chunk",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": None, "content": chunk_content},
                    "finish_reason": None
                }]
            }
            
        # 3. Stream finalize Choice
        yield {
            "id": "mock-vllm-stream-chunk",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": None, "content": None},
                "finish_reason": "stop"
            }]
        }
        
        latency_ms = (time.time() - start_time) * 1000.0
        span.set_attribute("llm.latency_seconds", latency_ms / 1000.0)
        completion_tokens = len(words) * 2 # estimate 2 tokens per word
        
        cost = self._get_cost(prompt_tokens, completion_tokens)
        PROVIDER_COST_USD_TOTAL.labels(backend="vllm", tenant=req.get("tenant_id", "anonymous")).inc(cost)
        
        # Yield statistics block
        yield {
            "id": "mock-vllm-stream-chunk",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cached_prompt_tokens": 0,
                "estimated_cost_usd": cost,
            },
            "timing": {
                "ttft_ms": ttft_ms,
                "latency_ms": latency_ms,
            }
        }
        span.end()
