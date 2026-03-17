"""Retriever Agent — searches local knowledge base and web.

Responsible for finding the most relevant information for a query,
using the user's documents when appropriate and online sources for
broader research questions.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from research_assistant.llm_factory import get_llm

from research_assistant.config import get_settings
from research_assistant.retrieval.retriever import (
    retrieve_from_vectorstore,
    format_context,
)
from research_assistant.tools.search_tool import search_web_raw, format_web_results

logger = logging.getLogger(__name__)

_ACADEMIC_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a query classifier. Determine if the following query is an academic "
        "research query that would benefit from Google Scholar search (e.g., asking "
        "about papers, studies, authors, scientific topics). "
        "Respond with ONLY 'ACADEMIC' or 'GENERAL'."
    )),
    ("human", "{query}"),
])

_SOURCE_STRATEGY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You decide where a research assistant should gather information from.\n\n"
        "Classify the query into exactly one of:\n"
        "- LOCAL  : the user is asking about their own files, uploaded documents, or stored papers\n"
        "- HYBRID : the question is topical and could benefit from both the user's materials and online sources\n"
        "- WEB    : the question is broad or general and should be answered from online sources rather than assuming the user's files contain it\n\n"
        "Choose LOCAL for phrasing like 'my files', 'my documents', 'what do I have', 'in the knowledge base'.\n"
        "Choose WEB for general knowledge, background explanations, recent developments, or broad research questions not tied to the user's files.\n"
        "Respond with ONLY one word: LOCAL, HYBRID, or WEB."
    )),
    ("human", "{query}"),
])

_RETRIEVER_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a relevance judge. Determine if the retrieved context is "
        "sufficient to answer the user's query. "
        "Respond with ONLY 'SUFFICIENT' or 'INSUFFICIENT'."
    )),
    ("human", "Query: {query}\n\nRetrieved context:\n{context}"),
])


async def run_retriever_agent(query: str, source_preference: str = "auto") -> dict:
    """Execute the retriever agent pipeline.

    1. Search the local vector store
    2. Judge relevance of results
    3. Fall back to web search if local results are insufficient
    4. Return combined context with sources

    Args:
        query: The user's research query.

    Returns:
        Dict with 'context' string and 'sources' list of Documents.
    """
    llm = get_llm(temperature=0)

    strategy_chain = _SOURCE_STRATEGY_PROMPT | llm
    strategy_response = await strategy_chain.ainvoke({"query": query})
    strategy = strategy_response.content.strip().upper()
    if strategy not in {"LOCAL", "HYBRID", "WEB"}:
        strategy = "HYBRID"

    preference = (source_preference or "auto").strip().lower()
    if preference not in {"auto", "local", "web", "papers"}:
        preference = "auto"

    # Preference nudges the routing but doesn't disable automatic decisions.
    if preference == "local" and strategy == "WEB":
        strategy = "HYBRID"
    elif preference == "web" and strategy == "LOCAL":
        strategy = "HYBRID"
    elif preference == "papers" and strategy == "LOCAL":
        strategy = "HYBRID"

    local_docs: list[Document] = []
    local_context = ""
    all_docs: list[Document] = []
    web_context = ""
    web_used = False
    is_sufficient = False
    local_source_preview: list[str] = []
    web_source_preview: list[str] = []

    should_check_local = strategy in {"LOCAL", "HYBRID"}
    if should_check_local:
        logger.info("[Retriever] Searching local knowledge base...")
        local_docs = retrieve_from_vectorstore(query)
        local_context = format_context(local_docs)
        all_docs.extend(local_docs)
        for doc in local_docs[:3]:
            meta = doc.metadata or {}
            local_source_preview.append(str(meta.get("title") or meta.get("source") or "Unknown source")[:90])

        if local_docs:
            judge_chain = _RETRIEVER_JUDGE_PROMPT | llm
            judgement = await judge_chain.ainvoke({
                "query": query,
                "context": local_context[:3000],
            })
            verdict = judgement.content.strip().upper()
            is_sufficient = verdict == "SUFFICIENT"

    # Web expansion is conservative:
    # - WEB strategy: always search web
    # - HYBRID strategy: only search web if local retrieval is insufficient
    # - LOCAL strategy: never auto-expand to web
    should_search_web = strategy == "WEB" or (strategy == "HYBRID" and not is_sufficient)

    if should_search_web:
        web_used = True
        if preference == "papers":
            is_academic = True
        else:
            academic_judgement = await (_ACADEMIC_JUDGE_PROMPT | llm).ainvoke({"query": query})
            is_academic = "ACADEMIC" in academic_judgement.content.upper()

        if is_academic:
            logger.info("[Retriever] Searching Scholar for academic/general research query...")
            import asyncio as _asyncio
            from research_assistant.tools.scholar_tool import search_scholar_multi as _scholar_multi

            raw_results = await _asyncio.to_thread(_scholar_multi, query, 5)
            if isinstance(raw_results, list):
                formatted_parts = []
                for result in raw_results:
                    formatted_parts.append(
                        (
                            f"Title: {result['title']}\n"
                            f"Authors: {result['authors']}\n"
                            f"Venue: {result['venue']} ({result['pub_year']})\n"
                            f"Citations: {result['citations']}\n"
                            f"Abstract: {result['abstract']}\n"
                            f"URL: {result.get('url', '')}"
                        )
                    )
                    all_docs.append(Document(
                        page_content=(
                            f"Title: {result['title']}\n"
                            f"Authors: {result['authors']}\n"
                            f"Venue: {result['venue']} ({result['pub_year']})\n"
                            f"Citations: {result['citations']}\n"
                            f"Abstract: {result['abstract']}"
                        ),
                        metadata={
                            "source": result.get("url") or result["title"],
                            "source_type": "web",
                            "source_kind": "scholar",
                            "title": result["title"],
                            "authors": result.get("authors", ""),
                            "venue": result.get("venue", ""),
                            "year": result.get("pub_year", ""),
                            "citations": result.get("citations", 0),
                            "url": result.get("url", ""),
                        },
                    ))
                    web_source_preview.append(str(result.get("title", "Untitled result"))[:90])
                web_results = "\n\n---\n\n".join(formatted_parts)
            else:
                web_results = str(raw_results)
                all_docs.append(Document(
                    page_content=web_results,
                    metadata={
                        "source": "Google Scholar",
                        "source_type": "web",
                        "source_kind": "scholar",
                        "title": "Google Scholar search results",
                    },
                ))
                web_source_preview.append("Google Scholar aggregated result")
            web_context = f"\n\n--- Google Scholar Results ---\n\n{web_results}"
        else:
            logger.info("[Retriever] Searching the web for general query...")
            raw_results = search_web_raw(query, 5)
            web_results = format_web_results(raw_results)
            for result in raw_results:
                web_source_preview.append(str(result.get("title") or result.get("href") or "Web result")[:90])
                all_docs.append(Document(
                    page_content=(
                        f"Title: {result.get('title', 'No title')}\n"
                        f"URL: {result.get('href', 'N/A')}\n"
                        f"Snippet: {result.get('body', 'No snippet')}"
                    ),
                    metadata={
                        "source": result.get("href") or result.get("title", "Web result"),
                        "source_type": "web",
                        "source_kind": "webpage",
                        "title": result.get("title", "No title"),
                        "url": result.get("href", ""),
                    },
                ))
            web_context = f"\n\n--- Web Results ---\n\n{web_results}"

    # Step 4: Combine
    combined_context = local_context + web_context

    logger.info(
        "[Retriever] Done — strategy=%s, %d local docs, web_used=%s",
        strategy,
        len(local_docs),
        web_used,
    )

    return {
        "context": combined_context,
        "sources": all_docs,
        "retrieval_strategy": strategy,
        "source_preference": preference,
        "local_doc_count": len(local_docs),
        "web_used": web_used,
        "local_sources": " | ".join(local_source_preview) if local_source_preview else "",
        "web_sources": " | ".join(web_source_preview[:3]) if web_source_preview else "",
    }
