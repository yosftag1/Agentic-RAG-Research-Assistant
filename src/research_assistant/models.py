"""Shared Pydantic models for the research assistant."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata for a research paper."""
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    source_type: str = "unknown"
