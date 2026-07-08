import os
import sys
import json
import time
import asyncio
from typing import Any, List
from httpx import AsyncClient

# Add project root to path to resolve imports correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inferroute.main import app
from benchmarks.evaluate_quality import evaluate_response_quality

DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "workload.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "eval_results.json")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# API Token for acme_corp tenant
HEADERS = {"Authorization": "Bearer sk-inferroute-demo"}

# Defined evaluation scenarios
SCENARIOS = [
    {
        "name": "always-openai",
        "payload_patch": {"model": "gpt-4o-mini", "routing": {"allow_local": False}}
    },
    {
        "name": "always-gemini",
        "payload_patch": {"model": "gemini-1.5-flash", "routing": {"allow_local": False}}
    },
    {
        "name": "always-vllm",
        "payload_patch": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "routing": {"allow_cloud": False}}
    },
    {
        "name": "cheapest-first",
        "payload_patch": {"model": "edge/auto", "routing": {"policy": "cost"}}
    },
    {
        "name": "fastest-first",
        "payload_patch": {"model": "edge/auto", "routing": {"policy": "latency"}}
    },
    {
        "name": "heuristic-reliability",
        "payload_patch": {"model": "edge/auto", "routing": {"policy": "reliability"}}
    },
    {
        "name": "learned-router",
        "payload_patch": {"model": "edge/auto", "routing": {"policy": "learned"}}
    },
    {
        "name": "cascade-router",
        "payload_patch": {"model": "meta-llama/Meta-Llama-3-8B-Instruct"} # Base cheap vLLM model, evaluator handles retry cascade
    }
]

async def execute_request(client: AsyncClient, payload: dict) -> dict[str, Any]:
    """Issues POST completion to the gateway and measures latencies."""
    start_time = time.time()
    
    # We query the chat completions route
    response = await client.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=10.0
    )
    
    latency = (time.time() - start_time) * 1000.0
    
    if response.status_code != 200:
        return {
            "success": False,
            "error": response.text,
            "latency_ms": latency,
            "ttft_ms": latency,
            "cost_usd": 0.0,
            "content": ""
        }
        
    data = response.json()
    choices = data.get("choices", [])
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    timing = data.get("timing", {})
    usage = data.get("usage", {})
    
    return {
        "success": True,
        "content": content,
        "latency_ms": timing.get("latency_ms", latency),
        "ttft_ms": timing.get("ttft_ms", latency / 2.0), # Estimate TTFT if missing
        "cost_usd": usage.get("estimated_cost_usd", 0.0),
        "model_used": data.get("model", "")
    }

async def run_evaluation():
    print("=============================================================")
    print("Starting InferRoute Learning-Based Router Evaluation Harness")
    print("=============================================================")
    
    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
        
    print(f"Loaded {len(prompts)} prompts across {len(set(p['category'] for p in prompts))} categories.")
    
    results = []
    
    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    # Run client inside the FastAPI lifespan context manager
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Give lifespan a millisecond to boot databases
        await asyncio.sleep(0.5)
        
        for scenario in SCENARIOS:
            name = scenario["name"]
            print(f"\n---> Running scenario: {name}")
            
            for item in prompts:
                prompt_id = item["id"]
                category = item["category"]
                prompt_text = item["prompt"]
                requires_json = item["requires_json"]
                expected_keys = item.get("expected_keys")
                ref_keywords = item.get("reference_keywords")
                
                # Base request body
                payload = {
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": 0.2,
                    "max_output_tokens": 128
                }
                payload.update(scenario["payload_patch"])
                
                # Execution with cascade fallback logic
                if name == "cascade-router":
                    # Step 1: Query cheap vLLM model
                    res = await execute_request(client, payload)
                    
                    if res["success"]:
                        # Evaluate quality
                        q_score = evaluate_response_quality(
                            category, res["content"], requires_json, expected_keys, ref_keywords
                        )
                    else:
                        q_score = 0.0
                        
                    # If quality is poor (< 0.60), trigger cascade fallback to strong OpenAI cloud model
                    if q_score < 0.60:
                        print(f"  [Cascade Triggered] ID={prompt_id} Category={category} (Cheap Quality={q_score:.2f} < 0.6)")
                        
                        # Prepare cascade payload (route directly to OpenAI strong model)
                        cascade_payload = payload.copy()
                        cascade_payload["model"] = "gpt-4o-mini"
                        cascade_payload["routing"] = {"allow_local": False}
                        
                        cascade_res = await execute_request(client, cascade_payload)
                        
                        if cascade_res["success"]:
                            final_content = cascade_res["content"]
                            final_quality = evaluate_response_quality(
                                category, final_content, requires_json, expected_keys, ref_keywords
                            )
                            # Total combined latency and cost for the chain
                            res = {
                                "success": True,
                                "content": final_content,
                                "latency_ms": res["latency_ms"] + cascade_res["latency_ms"],
                                "ttft_ms": res["ttft_ms"], # TTFT is of first attempt
                                "cost_usd": res["cost_usd"] + cascade_res["cost_usd"],
                                "model_used": cascade_res["model_used"]
                            }
                            q_score = final_quality
                        else:
                            # Keep original response if fallback failed
                            pass
                            
                        fallback_triggered = True
                    else:
                        fallback_triggered = False
                else:
                    # Regular single-route execution
                    res = await execute_request(client, payload)
                    fallback_triggered = False
                    
                    if res["success"]:
                        q_score = evaluate_response_quality(
                            category, res["content"], requires_json, expected_keys, ref_keywords
                        )
                    else:
                        q_score = 0.0
                
                # Check SLO compliance (target: p95 latency < 1000ms, success=True)
                slo_compliant = res["success"] and (res["latency_ms"] < 1000.0)
                
                results.append({
                    "scenario": name,
                    "prompt_id": prompt_id,
                    "category": category,
                    "success": res["success"],
                    "model_used": res.get("model_used", "unknown"),
                    "latency_ms": res["latency_ms"],
                    "ttft_ms": res["ttft_ms"],
                    "cost_usd": res["cost_usd"],
                    "quality_score": q_score,
                    "fallback_triggered": fallback_triggered,
                    "slo_compliant": slo_compliant
                })
                
                print(f"  ID={prompt_id:<14} Model={res.get('model_used', 'None'):<14} Latency={res['latency_ms']:>6.1f}ms Cost=${res['cost_usd']:.6f} Quality={q_score:.2f}")

    # Write evaluation results to file
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nEvaluation finished. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
