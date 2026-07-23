"""
InferRoute Academic Paper Reader Plugin.
Provides specialized endpoints for summarizing research papers, extracting core mathematical formulas,
and comparing baseline methods.
"""
import time
import logging
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status
from inferroute.auth import verify_api_key
from inferroute.adapters.gemini import GeminiAdapter

logger = logging.getLogger("inferroute.plugins.paper")

router = APIRouter(prefix="/paper", tags=["Academic Paper Reader"])

gemini_adapter = GeminiAdapter()


class PaperSummaryRequest(BaseModel):
    title: Optional[str] = Field("Academic Research Paper", example="ROUTERBENCH: A Benchmark for Multi-LLM Routing System")
    paper_text: str = Field(..., example="This paper formalizes multi-LLM routing as a multi-objective optimization problem...")
    focus_area: Optional[str] = Field("all", example="methodology", description="Focus: methodology, formulas, baselines, or all")


@router.post("/summarize")
async def summarize_paper(
    request: PaperSummaryRequest,
    tenant_id: str = Depends(verify_api_key)
):
    """
    Academic Paper Reader & Summarizer Endpoint.
    Extracts key takeaways, methodology, LaTeX math formulas, and baseline findings.
    """
    start_time = time.time()
    try:
        prompt = (
            f"You are an Academic AI Research Assistant.\n"
            f"Paper Title: {request.title}\n"
            f"Focus Area: {request.focus_area}\n\n"
            f"Paper Content Excerpt:\n{request.paper_text[:3000]}\n\n"
            f"Provide a structured academic summary:\n"
            f"1. Core Contribution & Problem Statement\n"
            f"2. Key Methodology & Mathematical Formulation (with LaTeX equations if applicable)\n"
            f"3. Experimental Baselines & Key Benchmark Results\n"
            f"4. Practical Applications & Limitations"
        )

        payload = {
            "model": "gemini-1.5-flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }

        response = await gemini_adapter.generate(payload)
        output_text = response["choices"][0]["message"]["content"]
        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "plugin": "paper_summarize",
            "title": request.title,
            "summary": output_text,
            "latency_ms": latency_ms,
            "tenant_id": tenant_id
        }
    except Exception as e:
        logger.error(f"Paper summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Paper summarization failed: {str(e)}")
