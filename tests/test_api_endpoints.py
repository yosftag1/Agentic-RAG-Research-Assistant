"""Tests for the new backend integration endpoints."""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from research_assistant.api.server import app

# Create a test client
client = TestClient(app)

def test_health_check_returns_stats():
    """Verify health endpoint returns stats and config."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "llm_provider" in data
    assert "stats" in data
    assert "doc_count" in data["stats"]

def test_settings_update():
    """Verify settings endpoint updates runtime configuration."""
    # Read original to restore later if needed, though this is a test environment
    original_health = client.get("/health").json()
    
    # Send update
    payload = {
        "llm_provider": "openai",
        "embedding_provider": "openai",
        "search_strategy": "mmr",
        "source_preference": "web",
    }
    response = client.post("/settings", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["settings"]["llm_provider"] == "openai"
    assert data["settings"]["search_strategy"] == "mmr"
    assert data["settings"]["source_preference"] == "web"
    
    # Reset
    reset_payload = {
        "llm_provider": original_health["llm_provider"],
        "embedding_provider": original_health["embedding_provider"],
        "search_strategy": original_health["search_strategy"],
        "source_preference": original_health["settings"].get("source_preference", "auto"),
    }
    client.post("/settings", json=reset_payload)


def test_settings_update_resets_collection_when_embedding_changes():
    """Changing the embedding model should trigger an automatic collection reset."""
    with patch("research_assistant.ingestion.embedder.reset_collection") as mock_reset:
        response = client.post("/settings", json={"embedding_model": "nomic-embed-text"})

    assert response.status_code == 200
    data = response.json()
    assert data["collection_reset"] is True
    mock_reset.assert_called_once()


def test_query_response_allows_structured_sources():
    """Query response model should accept structured source metadata items."""
    from research_assistant.api.server import QueryResponse

    response = QueryResponse(
        answer="Answer",
        sources=[{
            "label": "Example article",
            "url": "https://example.com",
            "origin": "Web page",
            "source": "https://example.com",
            "source_type": "web",
        }],
        web_search_results=[],
        steps=[],
    )

    assert response.sources[0]["label"] == "Example article"


def test_query_stream_endpoint_emits_final_payload():
    """The streaming query endpoint should emit progress events and a final payload."""
    streamed_result = {
        "final_answer": "Streamed answer",
        "sources": [{
            "label": "Example source",
            "url": "https://example.com",
            "origin": "Web page",
            "source": "https://example.com",
            "source_type": "web",
        }],
        "web_search_results": [],
        "steps": [{"agent": "Classifier", "detail": "done"}],
    }

    async def fake_stream_runner(query, history=None, source_preference="auto", progress_callback=None):
        if progress_callback is not None:
            await progress_callback({
                "type": "status",
                "agent": "Classifier",
                "state": "running",
                "message": "Classifying intent and resolving context",
            })
            await progress_callback({
                "type": "step",
                "agent": "Classifier",
                "state": "done",
                "message": "Completed",
                "step": streamed_result["steps"][0],
            })
        return streamed_result

    with patch("research_assistant.api.server.run_research_assistant_stream", side_effect=fake_stream_runner):
        response = client.post("/query/stream", json={"question": "test question"})

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.strip().splitlines()]
    assert any(event["type"] == "status" for event in events)
    final_event = next(event for event in events if event["type"] == "final")
    assert final_event["data"]["answer"] == "Streamed answer"
    assert final_event["data"]["sources"][0]["label"] == "Example source"

def test_documents_list_endpoint():
    """Verify we can fetch the document list."""
    response = client.get("/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
