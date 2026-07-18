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
from fastapi.responses import StreamingResponse, Response, HTMLResponse
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
    DEDUP_HIT_TOTAL,
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

BACKEND_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "vllm": "meta-llama/Meta-Llama-3-8B-Instruct",
    "ollama": "llama3"
}

router_engine = Router()
validator = OutputValidator()
cache_layer = CacheLayer()

from inferroute.validator import ReliabilityScorer
from inferroute.prompt_adapter import adapt_prompt
reliability_scorer = ReliabilityScorer()

from inferroute.rate_limiter import AdaptiveConcurrencyLimiter
concurrency_limiter = AdaptiveConcurrencyLimiter()


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
            # Billing: deduct token cost from tenant's wallet
            if tenant_id != "admin":
                from sqlalchemy import select
                from inferroute.models import UserWallet, TransactionLedger
                
                q_wallet = select(UserWallet).where(UserWallet.tenant_id == tenant_id)
                wallet = (await session.execute(q_wallet)).scalar_one_or_none()
                if not wallet:
                    wallet = UserWallet(tenant_id=tenant_id, balance_usd=5.0)
                    session.add(wallet)
                    await session.flush()

                if cost_usd > 0.0:
                    wallet.balance_usd = max(0.0, wallet.balance_usd - cost_usd)
                    ledger_entry = TransactionLedger(
                        tenant_id=tenant_id,
                        amount_usd=-cost_usd,
                        transaction_type="deduction",
                        description=f"LLM usage: {model} ({provider})"
                    )
                    session.add(ledger_entry)

            session.add(log_entry)
            await session.commit()
        except Exception as e:
            logger.error(f"Failed to write request log & balance deduction: {e}")


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


@app.get("/", response_class=HTMLResponse, tags=["ui"])
async def get_playground():
    """Serves the interactive playground UI."""
    import os
    template_path = os.path.join(os.path.dirname(__file__), "templates", "playground.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Playground UI template not found")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/academic", response_class=HTMLResponse, tags=["ui"])
async def get_academic():
    """Serves the academic technical hub page from docs/index.html."""
    import os
    # Read index.html from workspace docs folder
    doc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "index.html")
    if not os.path.exists(doc_path):
        raise HTTPException(status_code=404, detail="Technical documentation hub index.html not found under docs/")
    with open(doc_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/v1/routing/metrics", tags=["ops"])
async def routing_metrics():
    """Fetch aggregated lifetime usage and cost savings from database."""
    from sqlalchemy import select, func
    async with async_session() as session:
        # 1. Total requests
        q_total = select(func.count(RequestLog.id))
        total_requests = (await session.execute(q_total)).scalar() or 0

        # 2. Cache hits (exact + prefix)
        q_cache = select(func.count(RequestLog.id)).where(RequestLog.cache_hit == True)
        cache_hits = (await session.execute(q_cache)).scalar() or 0

        # 3. Deduplication hits
        q_dedup = select(func.count(RequestLog.id)).where(RequestLog.dedup_hit == True)
        dedup_hits = (await session.execute(q_dedup)).scalar() or 0

        # 4. Total USD cost of executed cloud requests
        q_cost = select(func.sum(RequestLog.cost_usd))
        actual_cost = (await session.execute(q_cost)).scalar() or 0.0

        # 5. Tokens saved (served by cache or local Ollama/vLLM)
        q_saved_tokens = select(func.sum(RequestLog.prompt_tokens + RequestLog.completion_tokens))\
            .where((RequestLog.cache_hit == True) | (RequestLog.provider.in_(["ollama", "vllm"])))
        saved_tokens = (await session.execute(q_saved_tokens)).scalar() or 0

        # Estimate savings: 0.002 USD per 1K tokens (reasonable average for GPT-4o-mini / Gemini-Flash mix)
        estimated_saved_usd = (saved_tokens / 1000.0) * 0.002

    return {
        "total_requests": total_requests,
        "cache_hits": cache_hits,
        "dedup_hits": dedup_hits,
        "actual_cost_usd": float(actual_cost),
        "estimated_saved_usd": float(estimated_saved_usd),
        "saved_tokens": int(saved_tokens)
    }


@app.get("/v1/billing/balance", tags=["billing"])
async def get_balance(tenant_id: str = Depends(verify_api_key)):
    """Fetch user wallet balance."""
    from sqlalchemy import select
    from inferroute.models import UserWallet

    async with async_session() as session:
        result = await session.execute(
            select(UserWallet).where(UserWallet.tenant_id == tenant_id)
        )
        wallet = result.scalar_one_or_none()
        if not wallet:
            wallet = UserWallet(tenant_id=tenant_id, balance_usd=5.0)
            session.add(wallet)
            await session.commit()

        return {"tenant_id": tenant_id, "balance_usd": wallet.balance_usd}


@app.post("/v1/billing/recharge", tags=["billing"])
async def recharge_wallet(amount: float = 10.0, tenant_id: str = Depends(verify_api_key)):
    """Recharge wallet with simulated USD funds."""
    from sqlalchemy import select
    from inferroute.models import UserWallet, TransactionLedger

    if amount <= 0:
        raise HTTPException(status_code=400, detail="Recharge amount must be positive")

    async with async_session() as session:
        result = await session.execute(
            select(UserWallet).where(UserWallet.tenant_id == tenant_id)
        )
        wallet = result.scalar_one_or_none()
        if not wallet:
            wallet = UserWallet(tenant_id=tenant_id, balance_usd=5.0)
            session.add(wallet)

        wallet.balance_usd += amount
        ledger_entry = TransactionLedger(
            tenant_id=tenant_id,
            amount_usd=amount,
            transaction_type="recharge",
            description=f"Simulated wallet recharge"
        )
        session.add(ledger_entry)
        await session.commit()

        return {
            "status": "success",
            "recharged_amount": amount,
            "new_balance_usd": wallet.balance_usd
        }


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
    start_time = time.time()
    is_owner = False
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

    # 2. Request deduplication — check if identical request is in-flight
    is_owner = await cache_layer.try_acquire_dedup_lock(body)
    if not is_owner:
        if stream_req:
            logger.info("[Gateway] Joining active in-flight stream deduplication...")
            async def dedup_stream_generator():
                DEDUP_HIT_TOTAL.labels(backend=model_req).inc()
                try:
                    async for chunk in cache_layer.wait_for_stream_dedup(body):
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"[Gateway] Stream dedup consumer failed: {e}")
                    yield f"data: {json.dumps({'error': f'Stream deduplication failed: {e}'})}\n\n"
                    yield "data: [DONE]\n\n"
            return StreamingResponse(
                dedup_stream_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            dedup_resp = await cache_layer.wait_for_dedup_result(body)
            if dedup_resp:
                REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend="cache", status="completed").inc()
                return dedup_resp
            # Wait timed out or owner failed — fall through to normal processing

    # We are the owner! Check concurrency limits before executing
    slot_acquired = await concurrency_limiter.acquire()
    if not slot_acquired:
        await cache_layer.release_dedup_lock(body)
        raise HTTPException(status_code=429, detail="Upstream concurrency limit reached. Please retry later.")

    # 4. Route
    decision = await router_engine.choose_backend(body)
    primary_backend = decision.primary
    fallback_backend = decision.fallback
    routing_policy = decision.policy

    # 5. Execute
    try:
        if routing_policy in ("cascade", "frugalgpt", "routing_survey"):
            try:
                cascade_data = json.loads(decision.reason)
                cascade_chain = cascade_data["cascade_chain"]
                acceptance_threshold = cascade_data["acceptance_threshold"]
            except Exception:
                cascade_chain = [primary_backend]
                if fallback_backend:
                    cascade_chain.append(fallback_backend)
                acceptance_threshold = 0.6
                
            if stream_req:
                return await handle_cascade_streaming_flow(
                    body, tenant_id, cascade_chain, acceptance_threshold, background_tasks
                )
            else:
                return await handle_cascade_blocking_flow(
                    body, tenant_id, cascade_chain, acceptance_threshold, background_tasks
                )

        elif routing_policy == "router_r1":
            from inferroute.agentic_router import agentic_router
            validate_fn = lambda prompt, output: reliability_scorer.evaluate_reliability({"messages": [{"role": "user", "content": prompt}]}, output)
            
            async def invoke_adapter_fn(backend: str, payload: dict[str, Any]):
                payload_copy = payload.copy()
                payload_copy["model"] = BACKEND_DEFAULT_MODELS.get(backend, backend)
                adapted_messages = adapt_prompt(payload_copy.get("messages", []), backend)
                payload_copy["messages"] = adapted_messages
                return await ADAPTERS[backend].generate(payload_copy)

            local_backends = ["ollama", "vllm"]
            cloud_backends = ["openai", "gemini"]
            local_cand = [b for b in ADAPTERS.keys() if b in local_backends]
            cloud_cand = [b for b in ADAPTERS.keys() if b in cloud_backends]
            l_backend = primary_backend if primary_backend in local_cand else (local_cand[0] if local_cand else "ollama")
            c_backend = fallback_backend if fallback_backend in cloud_cand else (cloud_cand[0] if cloud_cand else "openai")

            resp = await agentic_router.run_agentic_flow(
                body, l_backend, c_backend, invoke_adapter_fn, validate_fn,
                acceptance_threshold=body.get("routing", {}).get("acceptance_threshold", 0.7)
            )
            
            latency_ms = (time.time() - start_time) * 1000.0
            resp["logical_model"] = model_req
            fb_count = 1 if "<think>" in resp.get("choices", [{}])[0].get("message", {}).get("content", "") and "failed" in resp.get("choices", [{}])[0].get("message", {}).get("content", "") else 0
            resp["route"] = {
                "selected_backend": resp.get("model", primary_backend),
                "fallback_count": fb_count,
                "cache_hit": False,
                "policy": "router_r1",
                "slo_met": latency_ms <= settings.SLO_P95_MS
            }
            await cache_layer.store_exact(body, resp)
            await cache_layer.publish_dedup_result(body, resp)
            
            background_tasks.add_task(
                db_log_request,
                tenant_id=tenant_id, model=resp.get("model", model_req), logical_model=model_req,
                provider=resp["route"]["selected_backend"],
                prompt_tokens=resp.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=resp.get("usage", {}).get("completion_tokens", 0),
                cost_usd=resp.get("usage", {}).get("estimated_cost_usd", 0.0),
                cache_hit=False, cache_type=None, prefix_cache_hit=False, dedup_hit=False,
                primary_backend=l_backend, selected_backend=resp["route"]["selected_backend"],
                fallback_count=fb_count, routing_policy="router_r1",
                circuit_state="CLOSED", slo_met=resp["route"]["slo_met"],
                status_str="completed", error_message=None,
                queue_ms=0.0, ttft_ms=latency_ms, latency_ms=latency_ms
            )
            return resp

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
            latency_ms = (time.time() - start_time) * 1000.0
            await concurrency_limiter.release(latency_ms)



async def handle_cascade_blocking_flow(
    req: dict[str, Any],
    tenant_id: str,
    cascade_chain: list[str],
    acceptance_threshold: float,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    model_req = req.get("model", "edge/auto")
    start_time = time.time()
    
    cumulative_cost = 0.0
    cumulative_prompt_tokens = 0
    cumulative_completion_tokens = 0
    
    selected_backend = None
    final_resp = None
    step_details = []
    
    for idx, backend in enumerate(cascade_chain):
        # 1. Adapt prompt for cheap/local models (Prompt Adaptation)
        adapted_messages = adapt_prompt(req.get("messages", []), backend)
        backend_req = req.copy()
        backend_req["messages"] = adapted_messages
        backend_req["model"] = BACKEND_DEFAULT_MODELS.get(backend, backend)
        
        QUEUE_DEPTH.labels(backend=backend).inc()
        cb = circuit_breaker.get_circuit_breaker(backend)
        
        logger.info(f"[Cascade] Step {idx+1}/{len(cascade_chain)}: Trying backend={backend}")
        
        try:
            # Execute request on target backend
            resp = await ADAPTERS[backend].generate(backend_req)
            await cb.record_success()
            
            # Extract content and evaluate reliability
            choices = resp.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            
            # Score response
            score = reliability_scorer.evaluate_reliability(req, content)
            logger.info(f"[Cascade] Backend {backend} produced score {score:.2f} (threshold: {acceptance_threshold:.2f})")
            
            # Add step details for front-end rendering
            step_details.append({
                "backend": backend,
                "score": score,
                "accepted": score >= acceptance_threshold
            })
            
            # Keep track of cumulative tokens & cost
            usage = resp.get("usage", {})
            cumulative_prompt_tokens += usage.get("prompt_tokens", 0)
            cumulative_completion_tokens += usage.get("completion_tokens", 0)
            cumulative_cost += usage.get("estimated_cost_usd", 0.0)
            
            # Check acceptance condition
            if score >= acceptance_threshold or idx == len(cascade_chain) - 1:
                # Accepted or reached the end of the chain
                selected_backend = backend
                final_resp = resp
                QUEUE_DEPTH.labels(backend=backend).dec()
                break
                
            logger.info(f"[Cascade] Score {score:.2f} < threshold {acceptance_threshold:.2f}. Escalating to next tier...")
            
        except Exception as e:
            await cb.record_failure()
            logger.warning(f"[Cascade] Backend {backend} failed: {e}")
            step_details.append({
                "backend": backend,
                "score": 0.0,
                "accepted": False,
                "error": str(e)
            })
            
        QUEUE_DEPTH.labels(backend=backend).dec()
        
    if not final_resp:
        raise HTTPException(status_code=502, detail="All cascade models failed to respond.")
        
    latency_ms = (time.time() - start_time) * 1000.0
    slo_met = latency_ms <= settings.SLO_P95_MS
    
    # Force output model name to show which model accepted
    final_resp["model"] = selected_backend
    
    # Store exact cache and publish dedup
    await cache_layer.store_exact(req, final_resp)
    await cache_layer.publish_dedup_result(req, final_resp)
    
    # Log request in database with cumulative costs & tokens
    background_tasks.add_task(
        db_log_request,
        tenant_id=tenant_id,
        model=final_resp.get("model", model_req),
        logical_model=model_req,
        provider=selected_backend,
        prompt_tokens=cumulative_prompt_tokens,
        completion_tokens=cumulative_completion_tokens,
        cost_usd=cumulative_cost,
        cache_hit=False, cache_type=None, prefix_cache_hit=False, dedup_hit=False,
        primary_backend=cascade_chain[0],
        selected_backend=selected_backend,
        fallback_count=len(step_details) - 1,
        routing_policy="cascade",
        circuit_state="CLOSED", slo_met=slo_met,
        status_str="completed", error_message=None,
        queue_ms=0.0, ttft_ms=latency_ms, latency_ms=latency_ms,
    )
    
    REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=selected_backend, status="completed").inc()
    REQUEST_LATENCY.labels(tenant=tenant_id, model=model_req, backend=selected_backend).observe(latency_ms / 1000.0)
    await router_engine.record_metrics(selected_backend, latency_ms, latency_ms, success=True)
    
    # Enrich response payload for frontend visualization
    final_resp["logical_model"] = model_req
    final_resp["usage"] = {
        "prompt_tokens": cumulative_prompt_tokens,
        "completion_tokens": cumulative_completion_tokens,
        "total_tokens": cumulative_prompt_tokens + cumulative_completion_tokens,
        "estimated_cost_usd": cumulative_cost
    }
    final_resp["route"] = {
        "selected_backend": selected_backend,
        "fallback_count": len(step_details) - 1,
        "cache_hit": False,
        "policy": "cascade",
        "slo_met": slo_met,
        "cascade_steps": step_details
    }
    
    return final_resp


async def handle_cascade_streaming_flow(
    req: dict[str, Any],
    tenant_id: str,
    cascade_chain: list[str],
    acceptance_threshold: float,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    model_req = req.get("model", "edge/auto")

    async def event_generator():
        start_time = time.time()
        cumulative_cost = 0.0
        cumulative_prompt_tokens = 0
        cumulative_completion_tokens = 0
        
        selected_backend = None
        step_details = []
        final_text = ""
        ttft_ms = 0.0
        ttft_recorded = False
        
        for idx, backend in enumerate(cascade_chain):
            # 1. Adapt prompt for cheap/local models (Prompt Adaptation)
            adapted_messages = adapt_prompt(req.get("messages", []), backend)
            backend_req = req.copy()
            backend_req["messages"] = adapted_messages
            backend_req["model"] = BACKEND_DEFAULT_MODELS.get(backend, backend)
            
            QUEUE_DEPTH.labels(backend=backend).inc()
            cb = circuit_breaker.get_circuit_breaker(backend)
            
            logger.info(f"[Cascade Stream] Step {idx+1}/{len(cascade_chain)}: Trying backend={backend}")
            
            buffered_events = []
            backend_text = ""
            backend_prompt_tokens = 0
            backend_completion_tokens = 0
            backend_cost = 0.0
            
            try:
                async for chunk in ADAPTERS[backend].generate_stream(backend_req):
                    if "usage" in chunk:
                        usage = chunk["usage"]
                        backend_prompt_tokens = usage.get("prompt_tokens", 0)
                        backend_completion_tokens = usage.get("completion_tokens", 0)
                        backend_cost = usage.get("estimated_cost_usd", 0.0)
                        continue
                        
                    choices = chunk.get("choices", [])
                    content_chunk = ""
                    if choices and choices[0].get("delta", {}).get("content"):
                        content_chunk = choices[0]["delta"]["content"]
                        backend_text += content_chunk
                        if not ttft_recorded:
                            ttft_ms = (time.time() - start_time) * 1000.0
                            ttft_recorded = True
                            TTFT_LATENCY.labels(tenant=tenant_id, model=model_req, backend=backend).observe(ttft_ms / 1000.0)
                    
                    buffered_events.append(chunk)
                    
                await cb.record_success()
                
                # Score response
                score = reliability_scorer.evaluate_reliability(req, backend_text)
                logger.info(f"[Cascade Stream] Backend {backend} produced score {score:.2f} (threshold: {acceptance_threshold:.2f})")
                
                step_details.append({
                    "backend": backend,
                    "score": score,
                    "accepted": score >= acceptance_threshold
                })
                
                cumulative_prompt_tokens += backend_prompt_tokens
                cumulative_completion_tokens += backend_completion_tokens
                cumulative_cost += backend_cost
                
                # Check acceptance condition
                if score >= acceptance_threshold or idx == len(cascade_chain) - 1:
                    # Accept and output stream!
                    selected_backend = backend
                    final_text = backend_text
                    
                    # Yield all buffered events to client
                    chunk_idx = 0
                    for chunk in buffered_events:
                        chunk["model"] = selected_backend
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        await cache_layer.push_stream_chunk(req, chunk_idx, chunk)
                        chunk_idx += 1
                        
                    # Store cache & publish dedup
                    full_resp = {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": selected_backend,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": final_text}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": cumulative_prompt_tokens,
                            "completion_tokens": cumulative_completion_tokens,
                            "total_tokens": cumulative_prompt_tokens + cumulative_completion_tokens,
                            "estimated_cost_usd": cumulative_cost
                        }
                    }
                    await cache_layer.store_exact(req, full_resp)
                    await cache_layer.publish_dedup_result(req, full_resp)
                    
                    # Yield final end-of-stream stats chunk
                    latency_ms = (time.time() - start_time) * 1000.0
                    slo_met = latency_ms <= settings.SLO_P95_MS
                    
                    stats_chunk = {
                        "id": "inferroute-stream-end",
                        "object": "chat.completion.chunk",
                        "model": selected_backend,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": cumulative_prompt_tokens,
                            "completion_tokens": cumulative_completion_tokens,
                            "total_tokens": cumulative_prompt_tokens + cumulative_completion_tokens,
                            "estimated_cost_usd": cumulative_cost
                        },
                        "timing": {"ttft_ms": ttft_ms, "latency_ms": latency_ms},
                        "route": {
                            "selected_backend": selected_backend,
                            "fallback_count": len(step_details) - 1,
                            "slo_met": slo_met,
                            "policy": "cascade",
                            "cascade_steps": step_details
                        }
                    }
                    yield f"data: {json.dumps(stats_chunk, ensure_ascii=False)}\n\n"
                    await cache_layer.push_stream_chunk(req, chunk_idx, stats_chunk)
                    chunk_idx += 1
                    await cache_layer.publish_stream_end(req, chunk_idx)
                    yield "data: [DONE]\n\n"
                    
                    # Database log
                    background_tasks.add_task(
                        db_log_request,
                        tenant_id=tenant_id, model=selected_backend, logical_model=model_req,
                        provider=selected_backend,
                        prompt_tokens=cumulative_prompt_tokens,
                        completion_tokens=cumulative_completion_tokens,
                        cost_usd=cumulative_cost,
                        cache_hit=False, cache_type=None, prefix_cache_hit=False, dedup_hit=False,
                        primary_backend=cascade_chain[0], selected_backend=selected_backend,
                        fallback_count=len(step_details) - 1,
                        routing_policy="cascade", circuit_state="CLOSED", slo_met=slo_met,
                        status_str="completed", error_message=None,
                        queue_ms=0.0, ttft_ms=ttft_ms, latency_ms=latency_ms,
                    )
                    REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=selected_backend, status="completed").inc()
                    REQUEST_LATENCY.labels(tenant=tenant_id, model=model_req, backend=selected_backend).observe(latency_ms / 1000.0)
                    await router_engine.record_metrics(selected_backend, ttft_ms, latency_ms, success=True)
                    
                    QUEUE_DEPTH.labels(backend=backend).dec()
                    break
                    
                logger.info(f"[Cascade Stream] Score {score:.2f} < threshold {acceptance_threshold:.2f}. Escalating stream...")
                
            except Exception as e:
                await cb.record_failure()
                logger.warning(f"[Cascade Stream] Backend {backend} stream failed: {e}")
                step_details.append({
                    "backend": backend,
                    "score": 0.0,
                    "accepted": False,
                    "error": str(e)
                })
                
            QUEUE_DEPTH.labels(backend=backend).dec()
            
        else:
            yield f"data: {json.dumps({'error': 'All cascade models failed.'})}\n\n"
            yield "data: [DONE]\n\n"
            
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


def _stream_cached_response(cached_resp: dict[str, Any]) -> StreamingResponse:
    """Convert a cached blocking response into an SSE stream."""
    async def _gen():
        chunk_id = f"chatcmpl-{uuid.uuid4()}"
        choices = cached_resp.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': None, 'content': content}, 'finish_reason': 'stop'}]})}\n\n"
        yield f"data: {json.dumps({
            'id': 'inferroute-stream-end',
            'object': 'chat.completion.chunk',
            'choices': [],
            'usage': cached_resp.get('usage', {}),
            'timing': {'ttft_ms': 1.0, 'latency_ms': 1.0},
            'route': {
                'selected_backend': 'cache',
                'fallback_count': 0,
                'slo_met': True,
                'cache_hit': True,
                'cache_type': 'exact'
            }
        }, ensure_ascii=False)}\n\n"
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

    # Adapt prompt for primary backend
    from inferroute.prompt_adapter import adapt_prompt, apply_r2_constraints
    lambda_val = req.get("routing", {}).get("lambda", 1.0)
    adapted_primary_messages = adapt_prompt(req.get("messages", []), primary)
    if routing_policy == "r2_router":
        adapted_primary_messages = apply_r2_constraints(adapted_primary_messages, primary, lambda_val)
    req_primary = req.copy()
    req_primary["messages"] = adapted_primary_messages
    req_primary["model"] = BACKEND_DEFAULT_MODELS.get(primary, primary)

    try:
        logger.info(f"[Gateway] Invoking primary backend: {primary}")
        resp = await ADAPTERS[primary].generate(req_primary)

        if routing_policy == "speculative":
            choices = resp.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            quality_res = validator.validate_speculative_quality(content)
            if not quality_res.ok:
                raise ValueError(f"speculative_validation_failed: {quality_res.reason}")

        val_res = validator.validate_response(req_primary, resp)
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

            # Adapt prompt for fallback backend
            adapted_fallback_messages = adapt_prompt(req.get("messages", []), fallback)
            if routing_policy == "r2_router":
                adapted_fallback_messages = apply_r2_constraints(adapted_fallback_messages, fallback, lambda_val)
            req_fallback = req.copy()
            req_fallback["messages"] = adapted_fallback_messages
            req_fallback["model"] = BACKEND_DEFAULT_MODELS.get(fallback, fallback)

            try:
                resp = await ADAPTERS[fallback].generate(req_fallback)
                val_res = validator.validate_response(req_fallback, resp)
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

    prompt_text = " ".join(m.get("content", "") for m in req.get("messages", []))
    background_tasks.add_task(
        router_engine.trie_router.register_host_prefix,
        host=selected_backend,
        prompt_text=prompt_text
    )

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
        chunk_idx = 0

        is_spec = routing_policy == "speculative"
        buffered_chunks = []
        buffered_text = ""
        buffer_limit = 15
        buffer_validated = False

        QUEUE_DEPTH.labels(backend=primary).inc()
        cb = circuit_breaker.get_circuit_breaker(primary)

        # Adapt prompt for primary backend
        from inferroute.prompt_adapter import adapt_prompt, apply_r2_constraints
        lambda_val = req.get("routing", {}).get("lambda", 1.0)
        adapted_primary_messages = adapt_prompt(req.get("messages", []), primary)
        if routing_policy == "r2_router":
            adapted_primary_messages = apply_r2_constraints(adapted_primary_messages, primary, lambda_val)
        req_primary = req.copy()
        req_primary["messages"] = adapted_primary_messages
        req_primary["model"] = BACKEND_DEFAULT_MODELS.get(primary, primary)

        try:
            logger.info(f"[Gateway] Streaming from primary backend: {primary}")
            async for chunk in ADAPTERS[primary].generate_stream(req_primary):
                if "usage" in chunk:
                    usage = chunk["usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    cost_usd = usage.get("estimated_cost_usd", 0.0)
                    if "timing" in chunk:
                        ttft_ms = chunk["timing"].get("ttft_ms", ttft_ms)
                    continue
                choices = chunk.get("choices", [])
                content_chunk = ""
                if choices and choices[0].get("delta", {}).get("content"):
                    content_chunk = choices[0]["delta"]["content"]
                    if not ttft_recorded:
                        ttft_ms = (time.time() - start_time) * 1000.0
                        ttft_recorded = True
                        TTFT_LATENCY.labels(tenant=tenant_id, model=model_req, backend=primary).observe(ttft_ms / 1000.0)
                    accumulated_content.append(content_chunk)
                
                if is_spec and not buffer_validated:
                    buffered_chunks.append(chunk)
                    buffered_text += content_chunk
                    if len(buffered_chunks) >= buffer_limit:
                        val = validator.validate_speculative_quality(buffered_text)
                        if not val.ok:
                            logger.warning(f"[Gateway] Speculative quality check failed on primary: {val.reason}. Cascading to cloud backend.")
                            raise ValueError(f"speculative_validation_failed: {val.reason}")
                        else:
                            buffer_validated = True
                            for bc in buffered_chunks:
                                yield f"data: {json.dumps(bc, ensure_ascii=False)}\n\n"
                                await cache_layer.push_stream_chunk(req, chunk_idx, bc)
                                chunk_idx += 1
                    continue

                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await cache_layer.push_stream_chunk(req, chunk_idx, chunk)
                chunk_idx += 1

            if is_spec and not buffer_validated:
                val = validator.validate_speculative_quality(buffered_text)
                if not val.ok:
                    logger.warning(f"[Gateway] Speculative quality check failed (short stream): {val.reason}. Cascading to cloud backend.")
                    raise ValueError(f"speculative_validation_failed: {val.reason}")
                else:
                    for bc in buffered_chunks:
                        yield f"data: {json.dumps(bc, ensure_ascii=False)}\n\n"
                        await cache_layer.push_stream_chunk(req, chunk_idx, bc)
                        chunk_idx += 1

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
                # Adapt prompt for fallback backend
                adapted_fallback_messages = adapt_prompt(req.get("messages", []), fallback)
                if routing_policy == "r2_router":
                    adapted_fallback_messages = apply_r2_constraints(adapted_fallback_messages, fallback, lambda_val)
                req_fallback = req.copy()
                req_fallback["messages"] = adapted_fallback_messages
                req_fallback["model"] = BACKEND_DEFAULT_MODELS.get(fallback, fallback)

                try:
                    async for chunk in ADAPTERS[fallback].generate_stream(req_fallback):
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
                        await cache_layer.push_stream_chunk(req, chunk_idx, chunk)
                        chunk_idx += 1
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
                    await cache_layer.publish_stream_error(req, str(fallback_exc))
                    yield f"data: {json.dumps({'error': f'All backends failed: {fallback_exc}'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            else:
                latency_ms = (time.time() - start_time) * 1000.0
                REQUESTS_TOTAL.labels(tenant=tenant_id, model=model_req, backend=primary, status="failed").inc()
                await router_engine.record_metrics(primary, 0.0, latency_ms, success=False)
                QUEUE_DEPTH.labels(backend=primary).dec()
                await cache_layer.publish_stream_error(req, str(primary_exc))
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
            await cache_layer.publish_stream_error(req, error_msg)
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

            prompt_text = " ".join(m.get("content", "") for m in req.get("messages", []))
            background_tasks.add_task(
                router_engine.trie_router.register_host_prefix,
                host=selected_backend,
                prompt_text=prompt_text
            )

            stats_chunk = {
                "id": "inferroute-stream-end",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens, "estimated_cost_usd": cost_usd,
                },
                "timing": {"ttft_ms": ttft_ms, "latency_ms": latency_ms},
                "route": {
                    "selected_backend": selected_backend,
                    "fallback_count": fallback_count,
                    "slo_met": slo_met,
                    "policy": routing_policy,
                },
            }
            yield f"data: {json.dumps(stats_chunk, ensure_ascii=False)}\n\n"
            await cache_layer.push_stream_chunk(req, chunk_idx, stats_chunk)
            chunk_idx += 1
            await cache_layer.publish_stream_end(req, chunk_idx)
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
