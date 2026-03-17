"""Text splitting strategies for chunking documents.

Uses LangChain's RecursiveCharacterTextSplitter with configurable
chunk size and overlap. Preserves source metadata across chunks.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_assistant.config import get_settings

logger = logging.getLogger(__name__)


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into smaller chunks for embedding.

    Args:
        documents: List of documents to split.
        chunk_size: Characters per chunk (default from settings).
        chunk_overlap: Overlap between chunks (default from settings).

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(
        "Split %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )

    return chunks
