from fastapi.testclient import TestClient
import pytest
from app.main import app

client = TestClient(app)

def test_health_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "endpoints" in data

def test_get_documents_endpoint():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)

def test_submit_feedback_endpoint_success():
    payload = {
        "query": "How do I create background tasks?",
        "answer": "Use BackgroundTasks parameter and add_task method.",
        "rating": "up",
        "comment": "Accurate and clear!"
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "feedback_id" in data

def test_submit_feedback_invalid_rating():
    payload = {
        "query": "Invalid query",
        "answer": "Some answer",
        "rating": "invalid_rating_value"
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 400

def test_ingest_empty_bad_request():
    response = client.post("/ingest", json={"urls": []})
    assert response.status_code in (400, 422)

def test_query_endpoint_success():
    response = client.post("/query", json={"question": "How to use dependencies in FastAPI?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "execution_time_sec" in data
    assert data["question"] == "How to use dependencies in FastAPI?"

def test_query_endpoint_validation_error():
    response = client.post("/query", json={"question": "a"}) # Under min_length=2
    assert response.status_code == 422
