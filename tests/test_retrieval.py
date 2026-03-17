"""Tests for retrieval pipeline."""

from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from research_assistant.retrieval.retriever import format_context


class TestFormatContext:
    """Test context formatting with citations."""

    def test_empty_documents(self):
        """Empty list returns 'no documents' message."""
        result = format_context([])
        assert "No relevant documents" in result

    def test_single_document(self):
        """Single document is formatted with source citation."""
        doc = Document(
            page_content="Attention is all you need.",
            metadata={"source": "paper.pdf", "page": 1},
        )
        result = format_context([doc])
        assert "[Source: paper.pdf]" in result
        assert "Attention is all you need." in result

    def test_multiple_documents(self):
        """Multiple documents are separated and numbered."""
        docs = [
            Document(
                page_content="First finding",
                metadata={"source": "a.pdf"},
            ),
            Document(
                page_content="Second finding",
                metadata={"source": "b.pdf"},
            ),
        ]
        result = format_context(docs)
        assert "[Source 1:" in result
        assert "[Source 2:" in result
        assert "---" in result  # Separator


class TestRetriever:
    """Test retriever functions (mocked vector store)."""

    @patch("research_assistant.retrieval.retriever.get_settings")
    @patch("research_assistant.retrieval.retriever.get_vectorstore")
    def test_retrieve_calls_vectorstore(self, mock_get_vs, mock_get_settings):
        """retrieve_from_vectorstore calls vectorstore.similarity_search."""
        from research_assistant.retrieval.retriever import retrieve_from_vectorstore

        mock_settings = MagicMock()
        mock_settings.retrieval_top_k = 3
        mock_settings.search_strategy = "semantic"
        mock_get_settings.return_value = mock_settings

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            Document(page_content="result", metadata={"source": "test"})
        ]
        mock_get_vs.return_value = mock_vs

        results = retrieve_from_vectorstore("test query", top_k=3, filter_=None)
        mock_vs.similarity_search.assert_called_once_with("test query", k=3, filter=None)
        assert len(results) == 1
