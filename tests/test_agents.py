"""Tests for agent orchestration."""

from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

import pytest

from research_assistant.tools.citation_tool import format_citations
from langchain_core.documents import Document


class TestCitationTool:
    """Test citation formatting."""

    def test_empty_sources(self):
        """No sources returns fallback message."""
        result = format_citations([])
        assert "No sources" in result

    def test_deduplicates_sources(self):
        """Duplicate sources are deduplicated."""
        docs = [
            Document(page_content="a", metadata={"source": "paper.pdf", "source_type": "file"}),
            Document(page_content="b", metadata={"source": "paper.pdf", "source_type": "file"}),
        ]
        result = format_citations(docs)
        assert result.count("[1]") == 1
        assert "[2]" not in result

    def test_web_sources(self):
        """Web sources are formatted with URL."""
        docs = [
            Document(
                page_content="content",
                metadata={
                    "source": "https://example.com",
                    "source_type": "web",
                    "source_kind": "webpage",
                    "title": "Example article",
                    "url": "https://example.com",
                },
            ),
        ]
        result = format_citations(docs)
        assert "Web page:" in result
        assert "Example article" in result
        assert "https://example.com" in result

    def test_scholar_sources_include_metadata(self):
        """Scholar citations include paper metadata and URL."""
        docs = [
            Document(
                page_content="content",
                metadata={
                    "source": "https://papers.example/transformers",
                    "source_type": "web",
                    "source_kind": "scholar",
                    "title": "Attention Is All You Need",
                    "authors": "Vaswani et al.",
                    "year": "2017",
                    "venue": "NeurIPS",
                    "citations": 123456,
                    "url": "https://papers.example/transformers",
                },
            ),
        ]
        result = format_citations(docs)
        assert "Paper:" in result
        assert "Attention Is All You Need" in result
        assert "Vaswani et al." in result
        assert "NeurIPS" in result
        assert "Citations: 123456" in result
        assert "https://papers.example/transformers" in result

    def test_file_with_page(self):
        """File sources include page numbers."""
        docs = [
            Document(
                page_content="content",
                metadata={"source": "paper.pdf", "source_type": "file", "page": 5},
            ),
        ]
        result = format_citations(docs)
        assert "File:" in result
        assert "p. 5" in result


class TestOrchestrator:
    """Test orchestrator classification (mocked LLM)."""

    @pytest.mark.asyncio
    @patch("research_assistant.agents.orchestrator.get_llm")
    async def test_classify_intent_lookup(self, mock_get_llm):
        """Factual queries are classified as 'lookup'."""
        from research_assistant.agents.orchestrator import classify_intent, ResearchState

        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = "lookup"
        mock_llm.__or__ = lambda self, other: mock_llm
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        # Patch the prompt | llm chain
        with patch("research_assistant.agents.orchestrator._CLASSIFY_PROMPT") as mock_prompt:
            mock_chain = AsyncMock()
            mock_chain.ainvoke = AsyncMock(return_value=mock_response)
            mock_prompt.__or__.return_value = mock_chain

            state: ResearchState = {
                "query": "What is an attention mechanism?",
                "intent": "",
                "context": "",
                "sources": [],
                "summary": "",
                "analysis": "",
                "draft": "",
                "final_answer": "",
                "messages": [],
            }

            result = await classify_intent(state)
            assert result["intent"] == "lookup"


class TestRetrieverAgent:
    """Test retriever source strategy behavior."""

    @pytest.mark.asyncio
    @patch("research_assistant.agents.retriever_agent.search_web_raw")
    @patch("research_assistant.agents.retriever_agent.retrieve_from_vectorstore")
    @patch("research_assistant.agents.retriever_agent.get_llm")
    async def test_general_query_uses_web_first(
        self,
        mock_get_llm,
        mock_retrieve,
        mock_search_web_raw,
    ):
        """Broad questions should use web sources without depending on local docs."""
        from research_assistant.agents.retriever_agent import run_retriever_agent

        mock_get_llm.return_value = object()
        mock_retrieve.return_value = []
        mock_search_web_raw.return_value = [
            {
                "title": "What is retrieval-augmented generation?",
                "href": "https://example.com/rag",
                "body": "RAG combines retrieval with generation.",
            }
        ]

        strategy_chain = AsyncMock()
        strategy_chain.ainvoke = AsyncMock(return_value=SimpleNamespace(content="WEB"))
        academic_chain = AsyncMock()
        academic_chain.ainvoke = AsyncMock(return_value=SimpleNamespace(content="GENERAL"))

        with patch("research_assistant.agents.retriever_agent._SOURCE_STRATEGY_PROMPT") as mock_strategy_prompt, \
             patch("research_assistant.agents.retriever_agent._ACADEMIC_JUDGE_PROMPT") as mock_academic_prompt:
            mock_strategy_prompt.__or__.return_value = strategy_chain
            mock_academic_prompt.__or__.return_value = academic_chain

            result = await run_retriever_agent("What is retrieval-augmented generation?")

        mock_retrieve.assert_not_called()
        assert "--- Web Results ---" in result["context"]
        assert any(doc.metadata.get("source_type") == "web" for doc in result["sources"])
        assert any("example.com/rag" in doc.metadata.get("source", "") for doc in result["sources"])

    @pytest.mark.asyncio
    @patch("research_assistant.agents.retriever_agent.search_web_raw")
    @patch("research_assistant.agents.retriever_agent.format_context")
    @patch("research_assistant.agents.retriever_agent.retrieve_from_vectorstore")
    @patch("research_assistant.agents.retriever_agent.get_llm")
    async def test_hybrid_query_combines_local_and_web(
        self,
        mock_get_llm,
        mock_retrieve,
        mock_format_context,
        mock_search_web_raw,
    ):
        """Hybrid questions should stay local when local context is sufficient."""
        from research_assistant.agents.retriever_agent import run_retriever_agent

        local_doc = Document(
            page_content="Local notes about RAG evaluation.",
            metadata={"source": "notes.pdf", "source_type": "file"},
        )
        mock_get_llm.return_value = object()
        mock_retrieve.return_value = [local_doc]
        mock_format_context.return_value = "Local notes about RAG evaluation."
        mock_search_web_raw.return_value = [
            {
                "title": "Recent RAG evaluation trends",
                "href": "https://example.com/rag-eval",
                "body": "Benchmarks increasingly measure grounding and faithfulness.",
            }
        ]

        strategy_chain = AsyncMock()
        strategy_chain.ainvoke = AsyncMock(return_value=SimpleNamespace(content="HYBRID"))
        sufficiency_chain = AsyncMock()
        sufficiency_chain.ainvoke = AsyncMock(return_value=SimpleNamespace(content="SUFFICIENT"))
        academic_chain = AsyncMock()
        academic_chain.ainvoke = AsyncMock(return_value=SimpleNamespace(content="GENERAL"))

        with patch("research_assistant.agents.retriever_agent._SOURCE_STRATEGY_PROMPT") as mock_strategy_prompt, \
             patch("research_assistant.agents.retriever_agent._RETRIEVER_JUDGE_PROMPT") as mock_judge_prompt, \
             patch("research_assistant.agents.retriever_agent._ACADEMIC_JUDGE_PROMPT") as mock_academic_prompt:
            mock_strategy_prompt.__or__.return_value = strategy_chain
            mock_judge_prompt.__or__.return_value = sufficiency_chain
            mock_academic_prompt.__or__.return_value = academic_chain

            result = await run_retriever_agent("Compare practical RAG evaluation methods")

        assert "Local notes about RAG evaluation." in result["context"]
        assert "--- Web Results ---" not in result["context"]
        assert any(doc.metadata.get("source_type") == "file" for doc in result["sources"])
        assert not any(doc.metadata.get("source_type") == "web" for doc in result["sources"])

    @pytest.mark.asyncio
    @patch("research_assistant.agents.retriever_agent.search_web_raw")
    @patch("research_assistant.agents.retriever_agent.format_context")
    @patch("research_assistant.agents.retriever_agent.retrieve_from_vectorstore")
    @patch("research_assistant.agents.retriever_agent.get_llm")
    async def test_hybrid_query_uses_web_when_local_insufficient(
        self,
        mock_get_llm,
        mock_retrieve,
        mock_format_context,
        mock_search_web_raw,
    ):
        """Hybrid questions should expand to web only when local context is insufficient."""
        from research_assistant.agents.retriever_agent import run_retriever_agent

        local_doc = Document(
            page_content="Sparse local note.",
            metadata={"source": "notes.pdf", "source_type": "file"},
        )
        mock_get_llm.return_value = object()
        mock_retrieve.return_value = [local_doc]
        mock_format_context.return_value = "Sparse local note."
        mock_search_web_raw.return_value = [
            {
                "title": "External RAG source",
                "href": "https://example.com/rag-extra",
                "body": "Additional context from web.",
            }
        ]

        strategy_chain = AsyncMock()
        strategy_chain.ainvoke = AsyncMock(return_value=SimpleNamespace(content="HYBRID"))
        sufficiency_chain = AsyncMock()
        sufficiency_chain.ainvoke = AsyncMock(return_value=SimpleNamespace(content="INSUFFICIENT"))
        academic_chain = AsyncMock()
        academic_chain.ainvoke = AsyncMock(return_value=SimpleNamespace(content="GENERAL"))

        with patch("research_assistant.agents.retriever_agent._SOURCE_STRATEGY_PROMPT") as mock_strategy_prompt, \
             patch("research_assistant.agents.retriever_agent._RETRIEVER_JUDGE_PROMPT") as mock_judge_prompt, \
             patch("research_assistant.agents.retriever_agent._ACADEMIC_JUDGE_PROMPT") as mock_academic_prompt:
            mock_strategy_prompt.__or__.return_value = strategy_chain
            mock_judge_prompt.__or__.return_value = sufficiency_chain
            mock_academic_prompt.__or__.return_value = academic_chain

            result = await run_retriever_agent("Compare practical RAG evaluation methods")

        assert "Sparse local note." in result["context"]
        assert "--- Web Results ---" in result["context"]
        assert any(doc.metadata.get("source_type") == "file" for doc in result["sources"])
        assert any(doc.metadata.get("source_type") == "web" for doc in result["sources"])

    @pytest.mark.asyncio
    @patch("research_assistant.agents.orchestrator.get_llm")
    async def test_merge_appends_sources_section(self, mock_get_llm):
        """Final answers always end with a concrete sources section."""
        from research_assistant.agents.orchestrator import merge_results

        mock_response = SimpleNamespace(content="Answer body")
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = object()

        docs = [
            Document(
                page_content="content",
                metadata={
                    "source": "https://example.com/rag",
                    "source_type": "web",
                    "source_kind": "webpage",
                    "title": "RAG article",
                    "url": "https://example.com/rag",
                },
            )
        ]

        with patch("research_assistant.agents.orchestrator._MERGE_PROMPT") as mock_prompt:
            mock_prompt.__or__.return_value = mock_chain
            result = await merge_results({
                "query": "What is RAG?",
                "resolved_query": "What is RAG?",
                "intent": "lookup",
                "context": "context",
                "sources": docs,
                "summary": "",
                "analysis": "",
                "draft": "",
                "final_answer": "",
                "web_search_results": [],
                "conversation_history": [],
                "messages": [],
                "steps": [],
            })

        assert result["final_answer"].startswith("Answer body")
        assert "## Sources" in result["final_answer"]
        assert "RAG article" in result["final_answer"]
        assert "https://example.com/rag" in result["final_answer"]
