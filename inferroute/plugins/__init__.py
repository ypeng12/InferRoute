"""
InferRoute Plugins Package.
Provides specific high-level API endpoints for Image OCR, Quant.ai, Paper Reader, Text Summarization, and Translation.
"""
import base64
import time
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from inferroute.auth import verify_api_key
from inferroute.adapters.gemini import GeminiAdapter
from inferroute.config import settings

from inferroute.plugins.quant import router as quant_router
from inferroute.plugins.paper import router as paper_router

logger = logging.getLogger("inferroute.plugins")

router = APIRouter(prefix="/v1/plugins", tags=["AI Plugins"])
router.include_router(quant_router)
router.include_router(paper_router)

# Reuse the Gemini adapter for high-performance low-cost inference
gemini_adapter = GeminiAdapter()


@router.post("/ocr")
async def plugin_ocr(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    language: str = Form("auto"),
    tenant_id: str = Depends(verify_api_key)
):
    """
    OCR / Image Text Recognition Plugin.
    Uploads an image, extracts all text and structures it.
    """
    start_time = time.time()
    try:
        # Read file bytes
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = image.content_type or "image/jpeg"

        # Construct prompt
        prompt = (
            f"You are an expert OCR engine. Extract all text from this image. "
            f"Preserve formatting, layout, tables, and paragraphs where possible. "
            f"Target language setting: {language}. "
            f"Return ONLY the extracted text content. Do not write any greetings or explanations."
        )

        # Construct OpenAI-compatible multimodal payload
        payload = {
            "model": "gemini-1.5-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}
                        }
                    ]
                }
            ],
            "tenant_id": tenant_id
        }

        # Run completion
        response = await gemini_adapter.generate(payload)
        choices = response.get("choices", [])
        extracted_text = choices[0].get("message", {}).get("content", "") if choices else ""

        if not extracted_text:
            if gemini_adapter.mock_mode:
                extracted_text = (
                    f"--- [Mock OCR Result for {image.filename}] ---\n"
                    f"Receipt / Document Details:\n"
                    f"Date: {time.strftime('%Y-%m-%d')}\n"
                    f"Total Amount: $128.50\n"
                    f"Merchant: Antigravity AI Coffee Hub\n"
                    f"Items Purchased:\n"
                    f"1. Double Espresso (x2) - $8.00\n"
                    f"2. Smart Routing Gateway Sub (Monthly) - $120.50\n"
                    f"Thank you for your business!"
                )
            else:
                raise HTTPException(status_code=502, detail="Failed to extract text from the image")

        latency_ms = (time.time() - start_time) * 1000.0
        usage = response.get("usage", {})
        cost_usd = usage.get("estimated_cost_usd", 0.0)

        from inferroute.main import db_log_request
        background_tasks.add_task(
            db_log_request,
            tenant_id=tenant_id,
            model="gemini-1.5-flash",
            logical_model="plugin/ocr",
            provider="gemini",
            prompt_tokens=usage.get("prompt_tokens", 250),
            completion_tokens=usage.get("completion_tokens", 100),
            cost_usd=cost_usd,
            cache_hit=False,
            cache_type=None,
            prefix_cache_hit=False,
            dedup_hit=False,
            primary_backend="gemini",
            selected_backend="gemini",
            fallback_count=0,
            routing_policy="plugin",
            circuit_state="CLOSED",
            slo_met=latency_ms <= settings.SLO_P95_MS,
            status_str="completed",
            error_message=None,
            queue_ms=0.0,
            ttft_ms=latency_ms,
            latency_ms=latency_ms
        )

        return {
            "success": True,
            "plugin": "ocr",
            "filename": image.filename,
            "extracted_text": extracted_text,
            "latency_ms": round(latency_ms, 1),
            "estimated_cost_usd": cost_usd
        }

    except Exception as e:
        logger.error(f"OCR plugin execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


@router.post("/summarize")
async def plugin_summarize(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    max_length: int = Form(300),
    tenant_id: str = Depends(verify_api_key)
):
    """
    Summarize Plugin.
    Summarizes long input text and lists key bullet points.
    """
    start_time = time.time()
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text body cannot be empty")

        prompt = (
            f"Please summarize the following text. Highlight the core message and key takeaways "
            f"using bullet points. Limit the summary to approximately {max_length} words. \n\n"
            f"Text content:\n{text}"
        )

        payload = {
            "model": "gemini-1.5-flash",
            "messages": [{"role": "user", "content": prompt}],
            "tenant_id": tenant_id
        }

        response = await gemini_adapter.generate(payload)
        choices = response.get("choices", [])
        summary_text = choices[0].get("message", {}).get("content", "") if choices else ""

        if not summary_text and gemini_adapter.mock_mode:
            summary_text = (
                f"### [Mock Summary] Key Takeaways:\n"
                f"- The input document was successfully analyzed.\n"
                f"- A 56% reduction in operational cost was observed when routing model calls.\n"
                f"- Concurrency limiters protected model nodes from overload."
            )

        latency_ms = (time.time() - start_time) * 1000.0
        usage = response.get("usage", {})
        cost_usd = usage.get("estimated_cost_usd", 0.0)

        from inferroute.main import db_log_request
        background_tasks.add_task(
            db_log_request,
            tenant_id=tenant_id,
            model="gemini-1.5-flash",
            logical_model="plugin/summarize",
            provider="gemini",
            prompt_tokens=usage.get("prompt_tokens", 50),
            completion_tokens=usage.get("completion_tokens", 80),
            cost_usd=cost_usd,
            cache_hit=False,
            cache_type=None,
            prefix_cache_hit=False,
            dedup_hit=False,
            primary_backend="gemini",
            selected_backend="gemini",
            fallback_count=0,
            routing_policy="plugin",
            circuit_state="CLOSED",
            slo_met=latency_ms <= settings.SLO_P95_MS,
            status_str="completed",
            error_message=None,
            queue_ms=0.0,
            ttft_ms=latency_ms,
            latency_ms=latency_ms
        )

        return {
            "success": True,
            "plugin": "summarize",
            "summary": summary_text,
            "latency_ms": round(latency_ms, 1),
            "estimated_cost_usd": cost_usd
        }

    except Exception as e:
        logger.error(f"Summarize plugin failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summarize failed: {str(e)}")


@router.post("/translate")
async def plugin_translate(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    target_lang: str = Form("Chinese"),
    tenant_id: str = Depends(verify_api_key)
):
    """
    Translate Plugin.
    Translates input text into the target language.
    """
    start_time = time.time()
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text body cannot be empty")

        prompt = (
            f"You are a professional translator. Translate the following text into {target_lang}. "
            f"Maintain the original tone, context, and formatting. "
            f"Return ONLY the translation. Do not include notes or explanations.\n\n"
            f"Text to translate:\n{text}"
        )

        payload = {
            "model": "gemini-1.5-flash",
            "messages": [{"role": "user", "content": prompt}],
            "tenant_id": tenant_id
        }

        response = await gemini_adapter.generate(payload)
        choices = response.get("choices", [])
        translated_text = choices[0].get("message", {}).get("content", "") if choices else ""

        if not translated_text and gemini_adapter.mock_mode:
            translated_text = f"这是模拟翻译结果 (Target language: {target_lang}):\n{text}"

        latency_ms = (time.time() - start_time) * 1000.0
        usage = response.get("usage", {})
        cost_usd = usage.get("estimated_cost_usd", 0.0)

        from inferroute.main import db_log_request
        background_tasks.add_task(
            db_log_request,
            tenant_id=tenant_id,
            model="gemini-1.5-flash",
            logical_model="plugin/translate",
            provider="gemini",
            prompt_tokens=usage.get("prompt_tokens", 40),
            completion_tokens=usage.get("completion_tokens", 40),
            cost_usd=cost_usd,
            cache_hit=False,
            cache_type=None,
            prefix_cache_hit=False,
            dedup_hit=False,
            primary_backend="gemini",
            selected_backend="gemini",
            fallback_count=0,
            routing_policy="plugin",
            circuit_state="CLOSED",
            slo_met=latency_ms <= settings.SLO_P95_MS,
            status_str="completed",
            error_message=None,
            queue_ms=0.0,
            ttft_ms=latency_ms,
            latency_ms=latency_ms
        )

        return {
            "success": True,
            "plugin": "translate",
            "translated_text": translated_text,
            "latency_ms": round(latency_ms, 1),
            "estimated_cost_usd": cost_usd
        }

    except Exception as e:
        logger.error(f"Translate plugin failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translate failed: {str(e)}")
