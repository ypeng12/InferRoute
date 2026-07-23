import pytest
from fastapi.testclient import TestClient
from inferroute.main import app

client = TestClient(app)

def test_plugin_summarize_mock():
    # Test text summarization in mock mode
    response = client.post(
        "/v1/plugins/summarize",
        data={"text": "This is a long test text that we want to summarize using our smart router.", "max_length": 150},
        headers={"Authorization": "Bearer sk-inferroute-demo"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["plugin"] == "summarize"
    assert "summary" in data

def test_plugin_translate_mock():
    # Test translation in mock mode
    response = client.post(
        "/v1/plugins/translate",
        data={"text": "Hello world", "target_lang": "Chinese"},
        headers={"Authorization": "Bearer sk-inferroute-demo"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["plugin"] == "translate"
    assert "translated_text" in data

def test_plugin_ocr_mock():
    # Test OCR plugin with a mock file upload
    import io
    file_data = b"fake-image-bytes"
    response = client.post(
        "/v1/plugins/ocr",
        files={"image": ("test.png", io.BytesIO(file_data), "image/png")},
        data={"language": "auto"},
        headers={"Authorization": "Bearer sk-inferroute-demo"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["plugin"] == "ocr"
    assert "extracted_text" in data
