"""
Unit tests for InferRoute Quant.ai and Paper Reader plugins.
"""
import pytest
from fastapi.testclient import TestClient
from inferroute.main import app

client = TestClient(app)

def test_quant_analyze_endpoint():
    payload = {
        "ticker": "AAPL",
        "timeframe": "daily",
        "technical_indicators": {"RSI": 65.0, "MACD": "bullish"}
    }
    response = client.post(
        "/v1/plugins/quant/analyze",
        json=payload,
        headers={"Authorization": "Bearer sk-inferroute-demo"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["plugin"] == "quant_analyze"
    assert data["ticker"] == "AAPL"
    assert "bullish_score" in data


def test_quant_news_sentiment_endpoint():
    payload = {
        "text": "Apple announced record quarterly revenue exceeding analyst expectations.",
        "company_context": "Apple Inc."
    }
    response = client.post(
        "/v1/plugins/quant/news-sentiment",
        json=payload,
        headers={"Authorization": "Bearer sk-inferroute-demo"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["plugin"] == "quant_news_sentiment"
    assert "sentiment_score" in data


def test_paper_summarize_endpoint():
    payload = {
        "title": "ROUTERBENCH",
        "paper_text": "This paper formalizes multi-LLM routing as a multi-objective optimization problem."
    }
    response = client.post(
        "/v1/plugins/paper/summarize",
        json=payload,
        headers={"Authorization": "Bearer sk-inferroute-demo"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["plugin"] == "paper_summarize"
    assert "summary" in data


def test_analytics_summary_endpoint():
    response = client.get("/v1/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "total_cost_saved_usd" in data
    assert "circuit_breakers" in data
