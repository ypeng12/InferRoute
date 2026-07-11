"""
Unit and integration tests for FrugalGPT cascading router and prompt adaptation.
"""
import json
import pytest
from fastapi.testclient import TestClient
from inferroute.main import app
from inferroute.prompt_adapter import adapt_prompt, compress_few_shot_examples
from inferroute.validator import ReliabilityScorer

client = TestClient(app)
HEADERS = {"Authorization": "Bearer sk-inferroute-demo"}


# ── Test 1: Prompt Adaptation Trimming ─────────────────────────────────────────

def test_prompt_adaptation_trimming():
    """
    Ensure prompt adapter trims few-shot examples for cheap local backends (ollama, vllm)
    but preserves them for premium cloud models (openai, gemini).
    """
    few_shot_prompt = (
        "Classify the sentiment of these headlines:\n\n"
        "Example 1:\nHeadline: Gold prices surge amid inflation fears.\nSentiment: positive\n\n"
        "Example 2:\nHeadline: Retail sales plummet in Q3.\nSentiment: negative\n\n"
        "Example 3:\nHeadline: Central bank holds interest rates steady.\nSentiment: neutral\n\n"
        "Headline: Stock index hits record high today.\nSentiment:"
    )
    
    messages = [{"role": "user", "content": few_shot_prompt}]
    
    # Cheap backend -> should compress few-shot examples to keep max 1 example
    compressed = adapt_prompt(messages, "ollama")
    assert len(compressed[0]["content"]) < len(few_shot_prompt)
    assert "Example 1:" in compressed[0]["content"]
    assert "Example 2:" not in compressed[0]["content"]
    assert "Stock index hits record high today." in compressed[0]["content"]
    
    # Premium backend -> should preserve full few-shots
    preserved = adapt_prompt(messages, "openai")
    assert len(preserved[0]["content"]) == len(few_shot_prompt)
    assert "Example 2:" in preserved[0]["content"]
    assert "Example 3:" in preserved[0]["content"]


# ── Test 2: Reliability Scorer Math Classification ───────────────────────────

def test_reliability_scorer_math():
    """
    Test that ReliabilityScorer correctly matches known prompt templates
    and evaluates math answer correctness.
    """
    scorer = ReliabilityScorer()
    req = {
        "messages": [{"role": "user", "content": "Solve for x: 5x - 15 = 20. Output ONLY the final numeric value of x as an integer."}]
    }
    
    # Correct response (7) should score 1.0
    assert scorer.evaluate_reliability(req, "7") == 1.0
    assert scorer.evaluate_reliability(req, "The answer is 7") == 1.0
    
    # Incorrect response (5) should score 0.0
    assert scorer.evaluate_reliability(req, "5") == 0.0
    assert scorer.evaluate_reliability(req, "x = 10") == 0.0


# ── Test 3: Server-side Cascade Blocking Integration ─────────────────────────

def test_cascade_blocking_integration():
    """
    Test cascade routing logic on a math prompt.
    Since Ollama returns incorrect math answers, the router should escalate
    and eventually accept OpenAI/Gemini response.
    """
    payload = {
        "model": "edge/auto",
        "messages": [{"role": "user", "content": "Solve for x: 5x - 15 = 20. Output ONLY the final numeric value of x as an integer."}],
        "stream": False,
        "routing": {
            "policy": "cascade",
            "acceptance_threshold": 0.8,
            "cascade_chain": ["ollama", "vllm", "openai"]
        }
    }
    
    response = client.post("/v1/chat/completions", headers=HEADERS, json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # The final accepted model should be openai (since ollama and vllm return wrong math results)
    assert data["model"] == "openai"
    assert data["choices"][0]["message"]["content"].strip() == "7"
    
    # Verify cascade routing trace metadata
    route = data.get("route", {})
    assert route["policy"] == "cascade"
    assert route["fallback_count"] > 0
    assert len(route["cascade_steps"]) > 0
    
    # Confirm the first step failed quality but last step accepted
    assert route["cascade_steps"][0]["backend"] == "ollama"
    assert route["cascade_steps"][0]["accepted"] is False
    assert route["cascade_steps"][-1]["backend"] == "openai"
    assert route["cascade_steps"][-1]["accepted"] is True


# ── Test 4: Server-side Cascade Streaming Integration ────────────────────────

def test_cascade_streaming_integration():
    """
    Test cascade streaming execution flow.
    Ensures that stream chunks are correctly buffered and pumped only when accepted,
    and returns a final end-of-stream stats chunk.
    """
    payload = {
        "model": "edge/auto",
        "messages": [{"role": "user", "content": "Solve for x: 5x - 15 = 20. Output ONLY the final numeric value of x as an integer."}],
        "stream": True,
        "routing": {
            "policy": "cascade",
            "acceptance_threshold": 0.8,
            "cascade_chain": ["ollama", "vllm", "openai"]
        }
    }
    
    response = client.post("/v1/chat/completions", headers=HEADERS, json=payload)
    assert response.status_code == 200
    
    # Parse SSE stream chunks
    lines = response.text.split("\n")
    events = []
    for line in lines:
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str != "[DONE]":
                events.append(json.loads(data_str))
                
    assert len(events) > 0
    
    # Find the final stats chunk
    stats_chunk = next((e for e in events if e.get("id") == "inferroute-stream-end"), None)
    assert stats_chunk is not None
    
    route = stats_chunk.get("route", {})
    assert route["policy"] == "cascade"
    assert route["selected_backend"] == "openai"
    assert route["fallback_count"] > 0
    
    cascade_steps = route.get("cascade_steps", [])
    assert cascade_steps[0]["backend"] == "ollama"
    assert cascade_steps[0]["accepted"] is False
    assert cascade_steps[-1]["backend"] == "openai"
    assert cascade_steps[-1]["accepted"] is True
