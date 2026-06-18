import json
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Depends, HTTPException, Security, status, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from edgeserve.config import settings
from edgeserve.database import init_db, get_db, async_session
from edgeserve.models import RequestLog
from edgeserve.auth import verify_api_key, check_rate_limit
from edgeserve.cache import CacheLayer
from edgeserve.router import Router
from edgeserve.validator import OutputValidator
from edgeserve.adapters.openai import OpenAIAdapter
from edgeserve.adapters.vllm import VLLMAdapter
from edgeserve.observability import (
    setup_observability,
    get_metrics_response,
    REQUESTS_TOTAL,
    REQUEST_LATENCY,
    TTFT_LATENCY,
    INTER_TOKEN_LATENCY,
    FALLBACK_TOTAL,
    QUEUE_DEPTH
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edgeserve.main")

# Instantiate adapters
openai_adapter = OpenAIAdapter()
vllm_adapter = VLLMAdapter()

ADAPTERS = {
    "openai": openai_adapter,
    "vllm": vllm_adapter
}

router_engine = Router()
validator = OutputValidator()
cache_layer = CacheLayer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB & Redis
    logger.info("Initializing databases and external services...")
    await init_db()
    
    # Initialize global Redis client in auth module
    import edgeserve.auth as auth
    auth.redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info("Lifespan setup completed successfully.")
    yield
    # Shutdown: Close connections
    if auth.redis_client:
        await auth.redis_client.close()
        logger.info("Redis connection closed.")

app = FastAPI(
    title="EdgeServe AI Gateway",
    description="Production-grade low latency LLM inference router and observability gateway",
    version="0.1.0",
    lifespan=lifespan
)

# Setup OTel instrumentation
setup_observability(app)

# Helper function to write logs to PostgreSQL in the background
async def db_log_request(
    tenant_id: str,
    model: str,
    logical_model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    cache_hit: bool,
    cache_type: str | None,
    primary_backend: str,
    selected_backend: str,
    fallback_count: int,
    status_str: str,
    error_message: str | None,
    queue_ms: float,
    ttft_ms: float,
    latency_ms: float
):
    async with async_session() as session:
        try:
            log_entry = RequestLog(
                tenant_id=tenant_id,
                model=model,
                logical_model=logical_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=cost_usd,
                cache_hit=cache_hit,
                cache_type=cache_type,
                primary_backend=primary_backend,
                selected_backend=selected_backend,
                fallback_count=fallback_count,
                status=status_str,
                error_message=error_message,
                timing_queue_ms=queue_ms,
                timing_ttft_ms=ttft_ms,
                timing_latency_ms=latency_ms
            )
            session.add(log_entry)
            await session.commit()
        except Exception as e:
            logger.error(f"Failed to write request log to PostgreSQL: {e}")

@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/readyz")
async def readyz():
    import edgeserve.auth as auth
    status_info = {"postgres": "unhealthy", "redis": "unhealthy"}
    
    # Check Postgres
    try:
        async with async_session() as session:
            await session.execute("SELECT 1")
            status_info["postgres"] = "healthy"
    except Exception as e:
        logger.error(f"Readyz Postgres healthcheck failed: {e}")
        
    # Check Redis
    try:
        if auth.redis_client:
            await auth.redis_client.ping()
            status_info["redis"] = "healthy"
    except Exception as e:
        logger.error(f"Readyz Redis healthcheck failed: {e}")
        
    if "unhealthy" in status_info.values():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "degraded", "components": status_info}
        )
        
    return {"status": "ready", "components": status_info}

@app.get("/metrics")
async def metrics():
    content, media_type = get_metrics_response()
    from fastapi.responses import Response
    return Response(content=content, media_type=media_type)

@app.get("/v1/models")
async def list_models(tenant_id: str = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {"id": "edge/auto", "object": "model", "owned_by": "edgeserve"},
            {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
            {"id": "meta-llama/Meta-Llama-3-8B-Instruct", "object": "model", "owned_by": "meta"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(verify_api_key)
):
    # Retrieve body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    # Rate limit check
    await check_rate_limit(tenant_id)
    
    # Inject tenant_id into request dict for adapters/logging usage
    body["tenant_id"] = tenant_id
    
    model_req = body.get("model", "edge/auto")
    stream_req = body.get("stream", False)
    
    # 1. Exact Cache Lookup
    cached_resp = await cache_layer.lookup_exact(body)
    if cached_resp:
        REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend="cache", status="completed").inc()
        
        # Log to PostgreSQL asynchronously
        background_tasks.add_task(
            db_log_request,
            tenant_id=tenant_id,
            model=cached_resp.get("model", model_req),
            logical_model=model_req,
            prompt_tokens=cached_resp.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=cached_resp.get("usage", {}).get("completion_tokens", 0),
            cost_usd=0.0, # cache hit has no token costs
            cache_hit=True,
            cache_type="exact",
            primary_backend="cache",
            selected_backend="cache",
            fallback_count=0,
            status_str="completed",
            error_message=None,
            queue_ms=0.0,
            ttft_ms=5.0, # mock minimum lookup latency
            latency_ms=5.0
        )
        
        if stream_req:
            # Yield cached response structured as chunks
            async def cache_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4()}"
                choices = cached_resp.get("choices", [])
                content = choices[0].get("message", {}).get("content", "") if choices else ""
                
                # Yield initial role delta
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
                
                # Yield full content
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': None, 'content': content}, 'finish_reason': 'stop'}]})}\n\n"
                
                # Yield statistics
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [], 'usage': cached_resp.get('usage', {})})}\n\n"
                yield "data: [DONE]\n\n"
                
            return StreamingResponse(
                cache_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            return cached_resp

    # 2. Routing Decision
    primary_backend, fallback_backend, route_reason = await router_engine.choose_backend(body)
    
    # 3. Execution (Non-Streaming vs Streaming)
    if stream_req:
        return await handle_streaming_flow(
            body, tenant_id, primary_backend, fallback_backend, background_tasks
        )
    else:
        return await handle_blocking_flow(
            body, tenant_id, primary_backend, fallback_backend, background_tasks
        )

# --- FLOW ORCHESTRATION IMPLEMENTATIONS ---

async def handle_blocking_flow(
    req: dict[str, Any],
    tenant_id: str,
    primary: str,
    fallback: Optional[str],
    background_tasks: BackgroundTasks
) -> dict[str, Any]:
    model_req = req.get("model", "edge/auto")
    
    # Track metrics
    start_time = time.time()
    fallback_count = 0
    selected_backend = primary
    error_msg = None
    status_str = "completed"
    
    # Update queue depths
    QUEUE_DEPTH.labels(backend=primary).inc()
    
    try:
        # Step 1: Call primary backend
        logger.info(f"Invoking primary backend: {primary}")
        resp = await ADAPTERS[primary].generate(req)
        
        # Step 2: Validate Output
        val_res = validator.validate_response(req, resp)
        if not val_res.ok:
            raise ValueError(f"validation_failed: {val_res.reason}")
            
    except Exception as primary_exc:
        # Primary failed, attempt fallback if available
        logger.warning(f"Primary backend {primary} failed. Error: {primary_exc}")
        if fallback:
            logger.info(f"Attempting fallback backend: {fallback}")
            fallback_count += 1
            selected_backend = fallback
            FALLBACK_TOTAL.labels(from_backend=primary, to_backend=fallback, reason=str(primary_exc)).inc()
            
            QUEUE_DEPTH.labels(backend=primary).dec()
            QUEUE_DEPTH.labels(backend=fallback).inc()
            
            try:
                resp = await ADAPTERS[fallback].generate(req)
                val_res = validator.validate_response(req, resp)
                if not val_res.ok:
                    status_str = "validation_failed"
                    error_msg = f"Fallback validation failed: {val_res.reason}"
                    raise HTTPException(status_code=422, detail=error_msg)
            except Exception as fallback_exc:
                status_str = "failed"
                error_msg = f"Fallback failed: {fallback_exc}"
                logger.error(f"Fallback backend {fallback} also failed: {fallback_exc}")
                
                # Log execution record
                latency_ms = (time.time() - start_time) * 1000.0
                background_tasks.add_task(
                    db_log_request, tenant_id=tenant_id, model=model_req, logical_model=model_req,
                    prompt_tokens=0, completion_tokens=0, cost_usd=0.0, cache_hit=False, cache_type=None,
                    primary_backend=primary, selected_backend=selected_backend, fallback_count=fallback_count,
                    status_str=status_str, error_message=error_msg, queue_ms=0.0, ttft_ms=0.0, latency_ms=latency_ms
                )
                
                # Update metrics
                REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=selected_backend, status=status_str).inc()
                await router_engine.record_metrics(selected_backend, 0.0, latency_ms, success=False)
                
                QUEUE_DEPTH.labels(backend=selected_backend).dec()
                raise HTTPException(status_code=502, detail=error_msg)
        else:
            # No fallback, fail request
            status_str = "validation_failed" if "validation_failed" in str(primary_exc) else "failed"
            error_msg = str(primary_exc)
            latency_ms = (time.time() - start_time) * 1000.0
            
            background_tasks.add_task(
                db_log_request, tenant_id=tenant_id, model=model_req, logical_model=model_req,
                prompt_tokens=0, completion_tokens=0, cost_usd=0.0, cache_hit=False, cache_type=None,
                primary_backend=primary, selected_backend=selected_backend, fallback_count=fallback_count,
                status_str=status_str, error_message=error_msg, queue_ms=0.0, ttft_ms=0.0, latency_ms=latency_ms
            )
            
            REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=selected_backend, status=status_str).inc()
            await router_engine.record_metrics(selected_backend, 0.0, latency_ms, success=False)
            
            QUEUE_DEPTH.labels(backend=primary).dec()
            
            status_code = 422 if status_str == "validation_failed" else 502
            raise HTTPException(status_code=status_code, detail=error_msg)
            
    # Success execution
    latency_ms = (time.time() - start_time) * 1000.0
    ttft_ms = resp.get("timing", {}).get("ttft_ms", latency_ms)
    
    # Store in Cache
    await cache_layer.store_exact(req, resp)
    
    # Dec queue depth
    QUEUE_DEPTH.labels(backend=selected_backend).dec()
    
    # Record stats
    background_tasks.add_task(
        db_log_request,
        tenant_id=tenant_id,
        model=resp.get("model", model_req),
        logical_model=model_req,
        prompt_tokens=resp.get("usage", {}).get("prompt_tokens", 0),
        completion_tokens=resp.get("usage", {}).get("completion_tokens", 0),
        cost_usd=resp.get("usage", {}).get("estimated_cost_usd", 0.0),
        cache_hit=False,
        cache_type=None,
        primary_backend=primary,
        selected_backend=selected_backend,
        fallback_count=fallback_count,
        status_str=status_str,
        error_message=None,
        queue_ms=0.0,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms
    )
    
    # Update Prometheus metrics
    REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=selected_backend, status="completed").inc()
    REQUEST_LATENCY.labels(tenant=tenant_id, model=model_req, backend=selected_backend).observe(latency_ms / 1000.0)
    TTFT_LATENCY.labels(tenant=tenant_id, model=model_req, backend=selected_backend).observe(ttft_ms / 1000.0)
    
    # Record to router history
    await router_engine.record_metrics(selected_backend, ttft_ms, latency_ms, success=True)
    
    # Add routing details into standard response metadata
    resp["logical_model"] = model_req
    resp["route"] = {
        "selected_backend": selected_backend,
        "fallback_count": fallback_count,
        "cache_hit": False,
        "policy": req.get("routing", {}).get("policy", "latency")
    }
    return resp


async def handle_streaming_flow(
    req: dict[str, Any],
    tenant_id: str,
    primary: str,
    fallback: Optional[str],
    background_tasks: BackgroundTasks
) -> StreamingResponse:
    model_req = req.get("model", "edge/auto")
    
    async def event_generator():
        start_time = time.time()
        fallback_count = 0
        selected_backend = primary
        
        ttft_ms = 0.0
        ttft_recorded = False
        accumulated_content = []
        
        prompt_tokens = 0
        completion_tokens = 0
        cost_usd = 0.0
        
        QUEUE_DEPTH.labels(backend=primary).inc()
        
        try:
            # Stream from primary backend
            logger.info(f"Streaming from primary backend: {primary}")
            async for chunk in ADAPTERS[primary].generate_stream(req):
                # Intercept metadata chunk at the end of the stream
                if "usage" in chunk:
                    prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    completion_tokens = chunk["usage"].get("completion_tokens", 0)
                    cost_usd = chunk["usage"].get("estimated_cost_usd", 0.0)
                    if "timing" in chunk:
                        ttft_ms = chunk["timing"].get("ttft_ms", ttft_ms)
                    continue
                    
                # Record TTFT when first chunk content arrives
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    if not ttft_recorded:
                        ttft_ms = (time.time() - start_time) * 1000.0
                        ttft_recorded = True
                        TTFT_LATENCY.labels(tenant=tenant_id, model=model_req, backend=primary).observe(ttft_ms / 1000.0)
                    accumulated_content.append(choices[0]["delta"]["content"])
                    
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                
        except Exception as primary_exc:
            logger.warning(f"Streaming failed on primary backend {primary}: {primary_exc}")
            
            if fallback:
                logger.info(f"Streaming: falling back to {fallback}")
                fallback_count += 1
                selected_backend = fallback
                FALLBACK_TOTAL.labels(from_backend=primary, to_backend=fallback, reason=str(primary_exc)).inc()
                
                QUEUE_DEPTH.labels(backend=primary).dec()
                QUEUE_DEPTH.labels(backend=fallback).inc()
                
                # Clean accumulator if we restart stream entirely
                # Or append a clear boundary
                accumulated_content = [] 
                ttft_recorded = False
                
                try:
                    async for chunk in ADAPTERS[fallback].generate_stream(req):
                        if "usage" in chunk:
                            prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                            completion_tokens = chunk["usage"].get("completion_tokens", 0)
                            cost_usd = chunk["usage"].get("estimated_cost_usd", 0.0)
                            if "timing" in chunk:
                                ttft_ms = chunk["timing"].get("ttft_ms", ttft_ms)
                            continue
                            
                        choices = chunk.get("choices", [])
                        if choices and choices[0].get("delta", {}).get("content"):
                            if not ttft_recorded:
                                ttft_ms = (time.time() - start_time) * 1000.0
                                ttft_recorded = True
                                TTFT_LATENCY.labels(tenant=tenant_id, model=model_req, backend=fallback).observe(ttft_ms / 1000.0)
                            accumulated_content.append(choices[0]["delta"]["content"])
                            
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                except Exception as fallback_exc:
                    logger.error(f"Fallback streaming failed on {fallback}: {fallback_exc}")
                    latency_ms = (time.time() - start_time) * 1000.0
                    
                    background_tasks.add_task(
                        db_log_request, tenant_id=tenant_id, model=model_req, logical_model=model_req,
                        prompt_tokens=0, completion_tokens=0, cost_usd=0.0, cache_hit=False, cache_type=None,
                        primary_backend=primary, selected_backend=selected_backend, fallback_count=fallback_count,
                        status_str="failed", error_message=str(fallback_exc), queue_ms=0.0, ttft_ms=0.0, latency_ms=latency_ms
                    )
                    REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=selected_backend, status="failed").inc()
                    await router_engine.record_metrics(selected_backend, 0.0, latency_ms, success=False)
                    QUEUE_DEPTH.labels(backend=selected_backend).dec()
                    yield f"data: {json.dumps({'error': f'Streaming error occurred: {fallback_exc}'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            else:
                latency_ms = (time.time() - start_time) * 1000.0
                background_tasks.add_task(
                    db_log_request, tenant_id=tenant_id, model=model_req, logical_model=model_req,
                    prompt_tokens=0, completion_tokens=0, cost_usd=0.0, cache_hit=False, cache_type=None,
                    primary_backend=primary, selected_backend=selected_backend, fallback_count=fallback_count,
                    status_str="failed", error_message=str(primary_exc), queue_ms=0.0, ttft_ms=0.0, latency_ms=latency_ms
                )
                REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=selected_backend, status="failed").inc()
                await router_engine.record_metrics(selected_backend, 0.0, latency_ms, success=False)
                QUEUE_DEPTH.labels(backend=primary).dec()
                yield f"data: {json.dumps({'error': f'Streaming error occurred: {primary_exc}'})}\n\n"
                yield "data: [DONE]\n\n"
                return
                
        # Stream finished successfully, perform validation
        latency_ms = (time.time() - start_time) * 1000.0
        final_text = "".join(accumulated_content)
        
        # Schema validation check
        val_res = validator.validate_stream_chunk(req, final_text)
        status_str = "completed"
        error_msg = None
        
        if not val_res.ok:
            status_str = "validation_failed"
            error_msg = f"Output schema validation failed: {val_res.reason}"
            logger.error(error_msg)
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
        else:
            # Successful validation, Cache results (non-streaming shape)
            full_response_shape = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "openai/gpt-4o-mini" if selected_backend == "openai" else "meta-llama/Meta-Llama-3-8B-Instruct",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": final_text},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "estimated_cost_usd": cost_usd
                }
            }
            await cache_layer.store_exact(req, full_response_shape)
            
            # Send standard final statistics chunk to client
            stats_chunk = {
                "id": "edgeserve-stream-end",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "estimated_cost_usd": cost_usd,
                },
                "timing": {
                    "ttft_ms": ttft_ms,
                    "latency_ms": latency_ms
                }
            }
            yield f"data: {json.dumps(stats_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            
        # Record stats
        background_tasks.add_task(
            db_log_request,
            tenant_id=tenant_id,
            model="openai/gpt-4o-mini" if selected_backend == "openai" else "meta-llama/Meta-Llama-3-8B-Instruct",
            logical_model=model_req,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            cache_hit=False,
            cache_type=None,
            primary_backend=primary,
            selected_backend=selected_backend,
            fallback_count=fallback_count,
            status_str=status_str,
            error_message=error_msg,
            queue_ms=0.0,
            ttft_ms=ttft_ms,
            latency_ms=latency_ms
        )
        
        REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=selected_backend, status=status_str).inc()
        REQUEST_LATENCY.labels(tenant=tenant_id, model=model_req, backend=selected_backend).observe(latency_ms / 1000.0)
        await router_engine.record_metrics(selected_backend, ttft_ms, latency_ms, success=(status_str == "completed"))
        
        QUEUE_DEPTH.labels(backend=selected_backend).dec()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
