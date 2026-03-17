"""Tests for the LLM Factory."""

from unittest.mock import MagicMock, patch

import pytest

from research_assistant.llm_factory import get_embeddings, get_llm


@patch("research_assistant.llm_factory.get_settings")
def test_get_llm_gemini(mock_get_settings):
    """Test getting a Gemini model."""
    mock_settings = MagicMock()
    mock_settings.llm_provider = "gemini"
    mock_settings.llm_model = "gemini-2.0-flash"
    mock_get_settings.return_value = mock_settings
    
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_gemini:
        get_llm(temperature=0.5)
        mock_gemini.assert_called_once()
        args, kwargs = mock_gemini.call_args
        assert kwargs["temperature"] == 0.5


@patch("research_assistant.llm_factory.get_settings")
def test_get_llm_openai(mock_get_settings):
    """Test getting an OpenAI model."""
    mock_settings = MagicMock()
    mock_settings.llm_provider = "openai"
    mock_settings.llm_model = "gpt-4o"
    mock_get_settings.return_value = mock_settings
    
    with patch("langchain_openai.ChatOpenAI") as mock_openai:
        get_llm(temperature=0.1)
        mock_openai.assert_called_once()
        args, kwargs = mock_openai.call_args
        assert kwargs["temperature"] == 0.1
        assert kwargs["model"] == "gpt-4o"


@patch("research_assistant.llm_factory.get_settings")
def test_get_llm_ollama(mock_get_settings):
    """Test getting an Ollama model."""
    mock_settings = MagicMock()
    mock_settings.llm_provider = "ollama"
    mock_get_settings.return_value = mock_settings

    with patch("langchain_ollama.ChatOllama") as mock_ollama:
        get_llm(temperature=0.7)
        mock_ollama.assert_called_once()
        args, kwargs = mock_ollama.call_args
        assert kwargs["temperature"] == 0.7


@patch("research_assistant.llm_factory.get_settings")
def test_get_embeddings_ollama_falls_back_to_embedding_model(mock_get_settings):
    """Unsupported Ollama chat models should fall back to an embedding model."""
    mock_settings = MagicMock()
    mock_settings.embedding_provider = "ollama"
    mock_settings.embedding_model = "llama3.2"
    mock_settings.ollama_embedding_model = "nomic-embed-text"
    mock_settings.ollama_base_url = "http://localhost:11434"
    mock_get_settings.return_value = mock_settings

    class BrokenEmbeddings:
        def embed_query(self, text):
            raise ValueError("model does not support embeddings")

        def embed_documents(self, texts):
            raise ValueError("model does not support embeddings")

    class WorkingEmbeddings:
        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    def factory(*, model, base_url):
        assert base_url == "http://localhost:11434"
        if model == "llama3.2":
            return BrokenEmbeddings()
        if model == "nomic-embed-text":
            return WorkingEmbeddings()
        raise AssertionError(f"Unexpected model: {model}")

    with patch("langchain_ollama.OllamaEmbeddings", side_effect=factory) as mock_embeddings:
        embeddings = get_embeddings()
        result = embeddings.embed_query("hello")

    assert result == [0.1, 0.2, 0.3]
    assert mock_embeddings.call_count == 2
