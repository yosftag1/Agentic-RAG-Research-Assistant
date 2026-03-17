"""Tests for document ingestion pipeline."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from research_assistant.ingestion.loader import load_file, load_documents
from research_assistant.ingestion.splitter import split_documents


class TestLoader:
    """Test document loading."""

    def test_unsupported_extension(self, tmp_path: Path):
        """Unsupported file types raise ValueError."""
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(bad_file)

    def test_load_txt_file(self, tmp_path: Path):
        """Loading a .txt file returns documents with metadata."""
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("This is a test document with some content.")
        docs = load_file(txt_file)
        assert len(docs) >= 1
        assert "test document" in docs[0].page_content
        assert docs[0].metadata["source_type"] == "file"

    def test_load_documents_file_not_found(self):
        """Non-existent paths raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path/file.txt")

    def test_load_documents_url_detected(self):
        """URLs are detected and routed to web loader."""
        with patch("research_assistant.ingestion.loader.WebBaseLoader") as mock:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock.return_value = mock_instance
            load_documents("https://example.com")
            mock.assert_called_once_with("https://example.com")


class TestSplitter:
    """Test document splitting."""

    def test_split_creates_chunks(self, tmp_path: Path):
        """Splitting a document produces multiple chunks."""
        from langchain_core.documents import Document

        long_text = "This is a sentence. " * 200  # ~4000 chars
        docs = [Document(page_content=long_text, metadata={"source": "test"})]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_split_preserves_metadata(self):
        """Splitting preserves source metadata."""
        from langchain_core.documents import Document

        docs = [Document(page_content="Short text", metadata={"source": "paper.pdf"})]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        assert chunks[0].metadata["source"] == "paper.pdf"
