"""
Unit and integration tests for BYOK headers, budget limits, /v1/usage, and /v1/responses endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from inferroute.main import app
from inferroute.auth import extract_byok_keys, check_budget_limits
from inferroute.plugins.quant import validate_quant_strategy
from fastapi import HTTPException

client = TestClient(app)
DEMO_KEY = "sk-inferroute-demo"


def test_extract_byok_keys():
    class DummyRequest:
        headers = {
            "x-openai-api-key": "sk-user-openai-key-123",
            "x-gemini-api-key": "ai-user-gemini-key-456"
        }

    keys = extract_byok_keys(DummyRequest())
    assert keys.get("openai") == "sk-user-openai-key-123"
    assert keys.get("gemini") == "ai-user-gemini-key-456"


@pytest.mark.asyncio
async def test_check_budget_limits():
    # Admin is exempt
    await check_budget_limits("admin", estimated_cost_usd=10.0, max_cost_limit_usd=0.50)

    # Within cap
    await check_budget_limits("acme_corp", estimated_cost_usd=0.10, max_cost_limit_usd=0.50)

    # Exceeds cap
    with pytest.raises(HTTPException) as exc_info:
        await check_budget_limits("acme_corp", estimated_cost_usd=0.80, max_cost_limit_usd=0.50)
    assert exc_info.value.status_code == 400


def test_v1_usage_endpoint():
    resp = client.get("/v1/usage", headers={"Authorization": f"Bearer {DEMO_KEY}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "usage.summary"
    assert "total_spend_usd" in data
    assert "estimated_savings_percent" in data


def test_v1_responses_endpoint():
    payload = {
        "model": "edge/auto",
        "messages": [{"role": "user", "content": "Test responses API"}]
    }
    resp = client.post("/v1/responses", json=payload, headers={"Authorization": f"Bearer {DEMO_KEY}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"


def test_quant_strategy_validation():
    valid_strat = {"ticker": "AAPL", "stop_loss_pct": 0.05, "take_profit_pct": 0.15}
    valid, msg = validate_quant_strategy(valid_strat)
    assert valid is True

    invalid_strat = {"ticker": "", "stop_loss_pct": 0.99}
    valid, msg = validate_quant_strategy(invalid_strat)
    assert valid is False
