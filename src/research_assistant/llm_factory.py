"""Factory for instantiating LLMs and embedding models across different providers."""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from research_assistant.config import get_settings

logger = logging.getLogger(__name__)


class FallbackEmbeddings(Embeddings):
    """Try multiple embedding backends/models until one succeeds."""

    def __init__(self, embedding_factories: list[tuple[str, Callable[[], Embeddings]]]):
        self._embedding_factories = embedding_factories
        self._active_name: str | None = None
        self._active_embeddings: Embeddings | None = None

    def _get_or_create(self, name: str, factory: Callable[[], Embeddings]) -> Embeddings:
        if self._active_name == name and self._active_embeddings is not None:
            return self._active_embeddings
        embeddings = factory()
        self._active_name = name
        self._active_embeddings = embeddings
        return embeddings

    def _run_with_fallback(self, method_name: str, *args: Any) -> Any:
        last_error: Exception | None = None

        for name, factory in self._embedding_factories:
            try:
                embeddings = self._get_or_create(name, factory)
                method = getattr(embeddings, method_name)
                return method(*args)
            except Exception as exc:
                last_error = exc
                self._active_name = None
                self._active_embeddings = None
                logger.warning("Embedding backend %s failed: %s", name, exc)
                continue

        if last_error is not None:
            raise RuntimeError(
                "No embedding backend succeeded. If using Ollama, pull an embedding "
                "model like 'nomic-embed-text' or 'mxbai-embed-large'."
            ) from last_error
        raise RuntimeError("No embedding backend configured.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._run_with_fallback("embed_documents", texts)

    def embed_query(self, text: str) -> list[float]:
        return self._run_with_fallback("embed_query", text)


def get_llm(temperature: float = 0.3, **kwargs: Any) -> BaseChatModel:
    """Get a chat model instance based on the configured provider."""
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model_name = settings.llm_model
        if "gemini" in model_name:
            model_name = "gpt-4o"

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=settings.openai_api_key,
            **kwargs,
        )
    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "Ollama provider is selected but 'langchain-ollama' is not installed. "
                "Install dependencies with 'pip install .' (or 'pip install .[ollama]') "
                "or switch llm_provider to 'gemini'/'openai'."
            ) from exc
        return ChatOllama(
            model=settings.llm_model or settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            **kwargs,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
            **kwargs,
        )


import re as _re


def extract_reasoning(response) -> str | None:
    """Extract chain-of-thought reasoning from a thinking model response.

    Handles:
    - DeepSeek R1 / Qwen Thinking: <think>...</think> tags in content
    - OpenAI o1: reasoning_content in additional_kwargs
    - Anthropic extended thinking / Gemini: thinking field in additional_kwargs
    - Structured content blocks (list-type content)
    """
    content = getattr(response, 'content', '') or ''

    # Handle list-type content (some providers use structured content blocks)
    if isinstance(content, list):
        think_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'thinking':
                    think_parts.append(block.get('thinking', ''))
        if think_parts:
            return '\n\n'.join(think_parts).strip()
        content = ' '.join(
            b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text'
        )

    # DeepSeek R1, Qwen thinking, some Ollama models: <think>...</think> tags
    think_match = _re.search(r'<think>(.*?)</think>', str(content), _re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    # Check additional_kwargs for provider-specific reasoning fields
    ak = getattr(response, 'additional_kwargs', None)
    if isinstance(ak, dict):
        for key in ('reasoning_content', 'thinking', 'reasoning'):
            val = ak.get(key)
            if val:
                return str(val).strip()

    return None


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output (DeepSeek R1 style)."""
    if not isinstance(text, str):
        return str(text) if text else ''
    return _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()


def get_embeddings() -> Embeddings:
    """Get an embeddings instance based on the configured provider."""
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
        )
    elif provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "Ollama embedding provider is selected but 'langchain-ollama' is not installed. "
                "Install dependencies with 'pip install .' (or 'pip install .[ollama]') "
                "or switch embedding_provider to 'gemini'/'openai'."
            ) from exc

        requested_model = settings.embedding_model or settings.ollama_embedding_model
        fallback_models: list[str] = []
        for candidate in (
            requested_model,
            settings.ollama_embedding_model,
            "nomic-embed-text",
            "mxbai-embed-large",
        ):
            if candidate and candidate not in fallback_models:
                fallback_models.append(candidate)

        embedding_factories = [
            (
                f"ollama:{model_name}",
                lambda model_name=model_name: OllamaEmbeddings(
                    model=model_name,
                    base_url=settings.ollama_base_url,
                ),
            )
            for model_name in fallback_models
        ]
        return FallbackEmbeddings(embedding_factories)
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
