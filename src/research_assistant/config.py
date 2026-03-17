"""Configuration management using pydantic-settings.

All settings are loaded from environment variables or a .env file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings — loaded from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: str = "gemini"
    embedding_provider: str = "gemini"
    google_api_key: str = Field(default="", validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"))
    openai_api_key: str = ""
    
    llm_model: str = "gemini-2.0-flash"
    embedding_model: str = "models/gemini-embedding-001"
    
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    search_strategy: str = "hybrid"
    mmr_diversity: float = 0.3
    source_preference: str = "auto"

    chroma_persist_dir: str = "./chroma_db"
    document_dir: str = "./data"
    doc_registry_path: str = "./doc_registry.db"

    tavily_api_key: str = ""

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def data_path(self) -> Path:
        return Path(self.document_dir)

    def model_post_init(self, __context) -> None:
        # Render env values are sometimes pasted with quotes/spaces/newlines.
        self.google_api_key = self.google_api_key.strip().strip('"').strip("'")
        self.openai_api_key = self.openai_api_key.strip().strip('"').strip("'")
        self.llm_provider = self.llm_provider.strip().lower()
        self.embedding_provider = self.embedding_provider.strip().lower()
        
        # Startup validation log (without exposing actual key)
        key_status = "set" if self.google_api_key else "NOT SET"
        logger.info(
            f"Settings loaded: provider={self.llm_provider}, "
            f"google_api_key={key_status}, llm_model={self.llm_model}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
