"""
InferRoute — FastAPI inference gateway entry point.

Supports 4 providers: OpenAI, Google Gemini, vLLM (local), Ollama (local).
Features: SLO-aware routing, circuit breakers, exact + prefix caching,
request deduplication, streaming SSE, OpenTelemetry tracing, Prometheus metrics,
and PostgreSQL audit logging.
"""
import json
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Depends, HTTPException, Security, status, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, Response
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from inferroute.config import settings
from inferroute.database import init_db, get_db, async_session
from inferroute.models import RequestLog
from inferroute.auth import verify_api_key, check_rate_limit
from inferroute.cache import CacheLayer
from inferroute.router import Router, ALL_BACKENDS, BASELINES
from inferroute.validator import OutputValidator
from inferroute import circuit_breaker
from inferroute.adapters.openai import OpenAIAdapter
from inferroute.adapters.gemini import GeminiAdapter
from inferroute.adapters.vllm import VLLMAdapter
from inferroute.adapters.ollama import OllamaAdapter
from inferroute.observability import (
    setup_observability,
    get_metrics_response,
    REQUESTS_TOTAL,
    REQUEST_LATENCY,
    TTFT_LATENCY,
    FALLBACK_TOTAL,
    QUEUE_DEPTH,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inferroute.main")

# ── Adapter registry ──────────────────────────────────────────────────────────
ADAPTERS: dict[str, Any] = {
    "openai": OpenAIAdapter(),
    "gemini": GeminiAdapter(),
    "vllm":   VLLMAdapter(),
    "ollama": OllamaAdapter(),
}

router_engine = Router()
validator = OutputValidator()
cache_layer = CacheLayer()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting InferRoute gateway…")
    await init_db()

    import inferroute.auth as auth
    redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    auth.redis_client = redis_client

    # Inject Redis into circuit breakers
    circuit_breaker.initialize_circuit_breakers(redis_client, list(ADAPTERS.keys()))

    logger.info("InferRoute gateway ready.")
    yield

    if auth.redis_client:
        await auth.redis_client.aclose()
        logger.info("Redis connection closed.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="InferRoute Gateway",
    description=(
        "Kubernetes-ready LLM inference router supporting OpenAI, Google Gemini, "
        "vLLM, and local Ollama. Features SLO-aware routing, circuit breakers, "
        "prefix caching, and full observability."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

setup_observability(app)


# ── DB logging helper ─────────────────────────────────────────────────────────
async def db_log_request(
    tenant_id: str,
    model: str,
    logical_model: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    cache_hit: bool,
    cache_type: Optional[str],
    prefix_cache_hit: bool,
    dedup_hit: bool,
    primary_backend: str,
    selected_backend: str,
    fallback_count: int,
    routing_policy: str,
    circuit_state: str,
    slo_met: bool,
    status_str: str,
    error_message: Optional[str],
    queue_ms: float,
    ttft_ms: float,
    latency_ms: float,
) -> None:
    async with async_session() as session:
        try:
            log_entry = RequestLog(
                tenant_id=tenant_id,
                model=model,
                logical_model=logical_model,
                provider=provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=cost_usd,
                cache_hit=cache_hit,
                cache_type=cache_type,
                prefix_cache_hit=prefix_cache_hit,
                dedup_hit=dedup_hit,
                primary_backend=primary_backend,
                selected_backend=selected_backend,
                fallback_count=fallback_count,
                routing_policy=routing_policy,
                circuit_state=circuit_state,
                slo_met=slo_met,
                slo_p95_target_ms=settings.SLO_P95_MS,
                status=status_str,
                error_message=error_message,
                timing_queue_ms=queue_ms,
                timing_ttft_ms=ttft_ms,
                timing_latency_ms=latency_ms,
            )
            session.add(log_entry)
            await session.commit()
        except Exception as e:
            logger.error(f"Failed to write request log: {e}")


# ── Health / observability endpoints ─────────────────────────────────────────
@app.get("/healthz", status_code=status.HTTP_200_OK, tags=["ops"])
async def healthz():
    return {"status": "ok", "timestamp": time.time(), "version": "0.2.0"}


@app.get("/readyz", tags=["ops"])
async def readyz():
    import inferroute.auth as auth
    components = {"postgres": "unhealthy", "redis": "unhealthy"}

    try:
        async with async_session() as session:
            await session.execute("SELECT 1")
            components["postgres"] = "healthy"
    except Exception as e:
        logger.error(f"Readyz Postgres check failed: {e}")

    try:
        if auth.redis_client:
            await auth.redis_client.ping()
            components["redis"] = "healthy"
    except Exception as e:
        logger.error(f"Readyz Redis check failed: {e}")

    if "unhealthy" in components.values():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "degraded", "components": components},
        )
    return {"status": "ready", "components": components}


@app.get("/metrics", tags=["ops"])
async def metrics():
    content, media_type = get_metrics_response()
    return Response(content=content, media_type=media_type)


@app.get("/v1/models", tags=["inference"])
async def list_models(tenant_id: str = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {"id": "edge/auto",                              "object": "model", "owned_by": "inferroute"},
            {"id": "gpt-4o-mini",                           "object": "model", "owned_by": "openai"},
            {"id": "gpt-4o",                                "object": "model", "owned_by": "openai"},
            {"id": "gemini-1.5-flash",                      "object": "model", "owned_by": "google"},
            {"id": "gemini-1.5-pro",                        "object": "model", "owned_by": "google"},
            {"id": "meta-llama/Meta-Llama-3-8B-Instruct",   "object": "model", "owned_by": "meta"},
            {"id": "llama3",                                 "object": "model", "owned_by": "ollama"},
            {"id": "mistral",                                "object": "model", "owned_by": "ollama"},
        ],
    }


@app.get("/v1/providers", tags=["ops"])
async def list_providers():
    """Returns real-time provider health and circuit breaker states."""
    providers = []
    for backend in ALL_BACKENDS:
        cb = circuit_breaker.get_circuit_breaker(backend)
        cb_status = await cb.get_status()
        baseline = BASELINES[backend]
        providers.append({
            "backend": backend,
            "mock_mode": getattr(settings, f"MOCK_{backend.upper()}", False),
            "circuit_breaker": cb_status,
            "baseline_ttft_ms": baseline["ttft_ms"],
            "cost_per_token": baseline["cost_per_token"],
        })
    return {"providers": providers}


@app.get("/v1/routing/status", tags=["ops"])
async def routing_status():
    """Real-time routing dashboard: CB states, percentile latencies, cache stats."""
    backends_info = []
    for backend in ALL_BACKENDS:
        cb = circuit_breaker.get_circuit_breaker(backend)
        cb_info = await cb.get_status()
        stats = await router_engine.get_backend_stats(backend)
        backends_info.append({
            "backend": backend,
            "circuit_breaker": cb_info,
            "latency": {
                "p50_ms": round(stats["ttft_ms"], 1),
                "p95_ms": round(stats["p95_ms"], 1),
                "p99_ms": round(stats["p99_ms"], 1),
            },
            "failure_risk": round(stats["failure_risk"], 4),
        })

    return {
        "slo_targets": {
            "p50_ms": settings.SLO_P50_MS,
            "p95_ms": settings.SLO_P95_MS,
            "p99_ms": settings.SLO_P99_MS,
        },
        "backends": backends_info,
        "timestamp": time.time(),
    }


# ── Main inference endpoint ───────────────────────────────────────────────────
@app.post("/v1/chat/completions", tags=["inference"])
async def chat_completions(
    request: Request,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(verify_api_key),
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    await check_rate_limit(tenant_id)
    body["tenant_id"] = tenant_id

    model_req = body.get("model", "edge/auto")
    stream_req = body.get("stream", False)

    # 1. Exact cache lookup
    cached_resp = await cache_layer.lookup_exact(body)
    if cached_resp:
        REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend="cache", status="completed").inc()
        background_tasks.add_task(
            db_log_request,
            tenant_id=tenant_id, model=cached_resp.get("model", model_req),
            logical_model=model_req, provider="cache",
            prompt_tokens=cached_resp.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=cached_resp.get("usage", {}).get("completion_tokens", 0),
            cost_usd=0.0, cache_hit=True, cache_type="exact",
            prefix_cache_hit=False, dedup_hit=False,
            primary_backend="cache", selected_backend="cache", fallback_count=0,
            routing_policy="cache", circuit_state="CLOSED", slo_met=True,
            status_str="completed", error_message=None,
            queue_ms=0.0, ttft_ms=1.0, latency_ms=1.0,
        )
        if stream_req:
            return _stream_cached_response(cached_resp)
        return cached_resp

    # 2. Prefix cache lookup (hint — may trigger a miss in exact cache)
    prefix_resp = await cache_layer.lookup_prefix(body)
    if prefix_resp and not stream_req:
        REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend="cache", status="completed").inc()
        background_tasks.add_task(
            db_log_request,
            tenant_id=tenant_id, model=prefix_resp.get("model", model_req),
            logical_model=model_req, provider="cache",
            prompt_tokens=prefix_resp.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=prefix_resp.get("usage", {}).get("completion_tokens", 0),
            cost_usd=0.0, cache_hit=True, cache_type="prefix",
            prefix_cache_hit=True, dedup_hit=False,
            primary_backend="cache", selected_backend="cache", fallback_count=0,
            routing_policy="cache", circuit_state="CLOSED", slo_met=True,
            status_str="completed", error_message=None,
            queue_ms=0.0, ttft_ms=1.0, latency_ms=1.0,
        )
        return prefix_resp

    # 3. Request deduplication — check if identical request is in-flight
    is_owner = await cache_layer.try_acquire_dedup_lock(body)
    if not is_owner:
        dedup_resp = await cache_layer.wait_for_dedup_result(body)
        if dedup_resp:
            REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend="cache", status="completed").inc()
            if stream_req:
                return _stream_cached_response(dedup_resp)
            return dedup_resp
        # Wait timed out or owner failed — fall through to normal processing

    # 4. Route
    decision = await router_engine.choose_backend(body)
    primary_backend = decision.primary
    fallback_backend = decision.fallback
    routing_policy = decision.policy

    # 5. Execute
    try:
        if stream_req:
            return await handle_streaming_flow(
                body, tenant_id, primary_backend, fallback_backend, routing_policy, decision.slo_compliant, background_tasks
            )
        else:
            return await handle_blocking_flow(
                body, tenant_id, primary_backend, fallback_backend, routing_policy, decision.slo_compliant, background_tasks
            )
    finally:
        if is_owner:
            await cache_layer.release_dedup_lock(body)


def _stream_cached_response(cached_resp: dict[str, Any]) -> StreamingResponse:
    """Convert a cached blocking response into an SSE stream."""
    async def _gen():
        chunk_id = f"chatcmpl-{uuid.uuid4()}"
        choices = cached_resp.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': None, 'content': content}, 'finish_reason': 'stop'}]})}\n\n"
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [], 'usage': cached_resp.get('usage', {})})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


# ── Blocking flow ─────────────────────────────────────────────────────────────
async def handle_blocking_flow(
    req: dict[str, Any],
    tenant_id: str,
    primary: str,
    fallback: Optional[str],
    routing_policy: str,
    slo_compliant: bool,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    model_req = req.get("model", "edge/auto")
    start_time = time.time()
    fallback_count = 0
    selected_backend = primary
    error_msg: Optional[str] = None
    status_str = "completed"
    cb_primary = circuit_breaker.get_circuit_breaker(primary)

    QUEUE_DEPTH.labels(backend=primary).inc()

    resp: Optional[dict[str, Any]] = None

    try:
        logger.info(f"[Gateway] Invoking primary backend: {primary}")
        resp = await ADAPTERS[primary].generate(req)

        val_res = validator.validate_response(req, resp)
        if not val_res.ok:
            raise ValueError(f"validation_failed: {val_res.reason}")

        await cb_primary.record_success()

    except Exception as primary_exc:
        await cb_primary.record_failure()
        logger.warning(f"[Gateway] Primary {primary} failed: {primary_exc}")

        if fallback:
            fallback_count = 1
            selected_backend = fallback
            FALLBACK_TOTAL.labels(from_backend=primary, to_backend=fallback, reason=str(primary_exc)[:100]).inc()
            QUEUE_DEPTH.labels(backend=primary).dec()
            QUEUE_DEPTH.labels(backend=fallback).inc()
            cb_fallback = circuit_breaker.get_circuit_breaker(fallback)

            try:
                resp = await ADAPTERS[fallback].generate(req)
                val_res = validator.validate_response(req, resp)
                if not val_res.ok:
                    status_str = "validation_failed"
                    error_msg = f"Fallback validation failed: {val_res.reason}"
                    raise HTTPException(422, error_msg)
                await cb_fallback.record_success()

            except HTTPException:
                raise
            except Exception as fallback_exc:
                await cb_fallback.record_failure()
                status_str = "failed"
                error_msg = f"Fallback {fallback} failed: {fallback_exc}"
                logger.error(error_msg)
                latency_ms = (time.time() - start_time) * 1000.0
                background_tasks.add_task(
                    db_log_request,
                    tenant_id=tenant_id, model=model_req, logical_model=model_req,
                    provider=selected_backend,
                    prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
                    cache_hit=False, cache_type=None, prefix_cache_hit=False, dedup_hit=False,
                    primary_backend=primary, selected_backend=selected_backend,
                    fallback_count=fallback_count, routing_policy=routing_policy,
                    circuit_state="OPEN", slo_met=False,
                    status_str=status_str, error_message=error_msg,
                    queue_ms=0.0, ttft_ms=0.0, latency_ms=latency_ms,
                )
                REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=selected_backend, status=status_str).inc()
                await router_engine.record_metrics(selected_backend, 0.0, latency_ms, success=False)
                QUEUE_DEPTH.labels(backend=selected_backend).dec()
                # Notify dedup waiters of failure
                await cache_layer.publish_dedup_result(req, None)
                raise HTTPException(502, error_msg)
        else:
            status_str = "validation_failed" if "validation_failed" in str(primary_exc) else "failed"
            error_msg = str(primary_exc)
            latency_ms = (time.time() - start_time) * 1000.0
            background_tasks.add_task(
                db_log_request,
                tenant_id=tenant_id, model=model_req, logical_model=model_req,
                provider=primary,
                prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
                cache_hit=False, cache_type=None, prefix_cache_hit=False, dedup_hit=False,
                primary_backend=primary, selected_backend=primary,
                fallback_count=0, routing_policy=routing_policy,
                circuit_state="CLOSED", slo_met=False,
                status_str=status_str, error_message=error_msg,
                queue_ms=0.0, ttft_ms=0.0, latency_ms=latency_ms,
            )
            REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=primary, status=status_str).inc()
            await router_engine.record_metrics(primary, 0.0, latency_ms, success=False)
            QUEUE_DEPTH.labels(backend=primary).dec()
            await cache_layer.publish_dedup_result(req, None)
            raise HTTPException(422 if status_str == "validation_failed" else 502, error_msg)

    # ── Success path ─────────────────────────────────────────────────────────
    latency_ms = (time.time() - start_time) * 1000.0
    ttft_ms = resp.get("timing", {}).get("ttft_ms", latency_ms) if resp else latency_ms
    slo_met = latency_ms <= settings.SLO_P95_MS

    await cache_layer.store_exact(req, resp)
    await cache_layer.publish_dedup_result(req, resp)
    QUEUE_DEPTH.labels(backend=selected_backend).dec()

    usage = resp.get("usage", {}) if resp else {}
    background_tasks.add_task(
        db_log_request,
        tenant_id=tenant_id,
        model=resp.get("model", model_req) if resp else model_req,
        logical_model=model_req,
        provider=selected_backend,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        cost_usd=usage.get("estimated_cost_usd", 0.0),
        cache_hit=False, cache_type=None, prefix_cache_hit=False, dedup_hit=False,
        primary_backend=primary, selected_backend=selected_backend,
        fallback_count=fallback_count, routing_policy=routing_policy,
        circuit_state="CLOSED", slo_met=slo_met,
        status_str=status_str, error_message=None,
        queue_ms=0.0, ttft_ms=ttft_ms, latency_ms=latency_ms,
    )

    REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=selected_backend, status="completed").inc()
    REQUEST_LATENCY.labels(tenant=tenant_id, model=model_req, backend=selected_backend).observe(latency_ms / 1000.0)
    TTFT_LATENCY.labels(tenant=tenant_id, model=model_req, backend=selected_backend).observe(ttft_ms / 1000.0)
    await router_engine.record_metrics(selected_backend, ttft_ms, latency_ms, success=True)

    resp["logical_model"] = model_req
    resp["route"] = {
        "selected_backend": selected_backend,
        "fallback_count": fallback_count,
        "cache_hit": False,
        "policy": routing_policy,
        "slo_met": slo_met,
    }
    return resp


# ── Streaming flow ────────────────────────────────────────────────────────────
async def handle_streaming_flow(
    req: dict[str, Any],
    tenant_id: str,
    primary: str,
    fallback: Optional[str],
    routing_policy: str,
    slo_compliant: bool,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    model_req = req.get("model", "edge/auto")

    async def event_generator():
        start_time = time.time()
        fallback_count = 0
        selected_backend = primary
        ttft_ms = 0.0
        ttft_recorded = False
        accumulated_content: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        cost_usd = 0.0

        QUEUE_DEPTH.labels(backend=primary).inc()
        cb = circuit_breaker.get_circuit_breaker(primary)

        try:
            logger.info(f"[Gateway] Streaming from primary backend: {primary}")
            async for chunk in ADAPTERS[primary].generate_stream(req):
                if "usage" in chunk:
                    usage = chunk["usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    cost_usd = usage.get("estimated_cost_usd", 0.0)
                    if "timing" in chunk:
                        ttft_ms = chunk["timing"].get("ttft_ms", ttft_ms)
                    continue
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    if not ttft_recorded:
                        ttft_ms = (time.time() - start_time) * 1000.0
                        ttft_recorded = True
                        TTFT_LATENCY.labels(tenant=tenant_id, model=model_req, backend=primary).observe(ttft_ms / 1000.0)
                    accumulated_content.append(choices[0]["delta"]["content"])
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            await cb.record_success()

        except Exception as primary_exc:
            await cb.record_failure()
            logger.warning(f"[Gateway] Streaming primary {primary} failed: {primary_exc}")

            if fallback:
                fallback_count = 1
                selected_backend = fallback
                FALLBACK_TOTAL.labels(from_backend=primary, to_backend=fallback, reason=str(primary_exc)[:100]).inc()
                QUEUE_DEPTH.labels(backend=primary).dec()
                QUEUE_DEPTH.labels(backend=fallback).inc()
                accumulated_content = []
                ttft_recorded = False
                cb_fb = circuit_breaker.get_circuit_breaker(fallback)

                try:
                    async for chunk in ADAPTERS[fallback].generate_stream(req):
                        if "usage" in chunk:
                            usage = chunk["usage"]
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)
                            cost_usd = usage.get("estimated_cost_usd", 0.0)
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
                    await cb_fb.record_success()

                except Exception as fallback_exc:
                    await cb_fb.record_failure()
                    latency_ms = (time.time() - start_time) * 1000.0
                    background_tasks.add_task(
                        db_log_request, tenant_id=tenant_id, model=model_req, logical_model=model_req,
                        provider=selected_backend, prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
                        cache_hit=False, cache_type=None, prefix_cache_hit=False, dedup_hit=False,
                        primary_backend=primary, selected_backend=selected_backend, fallback_count=fallback_count,
                        routing_policy=routing_policy, circuit_state="OPEN", slo_met=False,
                        status_str="failed", error_message=str(fallback_exc),
                        queue_ms=0.0, ttft_ms=0.0, latency_ms=latency_ms,
                    )
                    REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=selected_backend, status="failed").inc()
                    await router_engine.record_metrics(selected_backend, 0.0, latency_ms, success=False)
                    QUEUE_DEPTH.labels(backend=selected_backend).dec()
                    await cache_layer.publish_dedup_result(req, None)
                    yield f"data: {json.dumps({'error': f'All backends failed: {fallback_exc}'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            else:
                latency_ms = (time.time() - start_time) * 1000.0
                REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=primary, status="failed").inc()
                await router_engine.record_metrics(primary, 0.0, latency_ms, success=False)
                QUEUE_DEPTH.labels(backend=primary).dec()
                await cache_layer.publish_dedup_result(req, None)
                yield f"data: {json.dumps({'error': f'Primary failed: {primary_exc}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

        # ── Post-stream success ───────────────────────────────────────────────
        latency_ms = (time.time() - start_time) * 1000.0
        final_text = "".join(accumulated_content)
        slo_met = latency_ms <= settings.SLO_P95_MS

        val_res = validator.validate_stream_chunk(req, final_text)
        status_str = "completed"
        error_msg: Optional[str] = None

        if not val_res.ok:
            status_str = "validation_failed"
            error_msg = f"Stream validation failed: {val_res.reason}"
            logger.error(error_msg)
            await cache_layer.publish_dedup_result(req, None)
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
        else:
            full_resp = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": selected_backend,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": final_text}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens, "estimated_cost_usd": cost_usd,
                },
            }
            await cache_layer.store_exact(req, full_resp)
            await cache_layer.publish_dedup_result(req, full_resp)

            stats_chunk = {
                "id": "inferroute-stream-end",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens, "estimated_cost_usd": cost_usd,
                },
                "timing": {"ttft_ms": ttft_ms, "latency_ms": latency_ms},
                "route": {"selected_backend": selected_backend, "fallback_count": fallback_count, "slo_met": slo_met},
            }
            yield f"data: {json.dumps(stats_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        background_tasks.add_task(
            db_log_request,
            tenant_id=tenant_id, model=selected_backend, logical_model=model_req,
            provider=selected_backend,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost_usd=cost_usd,
            cache_hit=False, cache_type=None, prefix_cache_hit=False, dedup_hit=False,
            primary_backend=primary, selected_backend=selected_backend, fallback_count=fallback_count,
            routing_policy=routing_policy, circuit_state="CLOSED", slo_met=slo_met,
            status_str=status_str, error_message=error_msg,
            queue_ms=0.0, ttft_ms=ttft_ms, latency_ms=latency_ms,
        )
        REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=selected_backend, status=status_str).inc()
        REQUEST_LATENCY.labels(tenant=tenant_id, model=model_req, backend=selected_backend).observe(latency_ms / 1000.0)
        await router_engine.record_metrics(selected_backend, ttft_ms, latency_ms, success=(status_str == "completed"))
        QUEUE_DEPTH.labels(backend=selected_backend).dec()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
