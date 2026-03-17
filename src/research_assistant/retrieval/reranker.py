"""Re-ranking module — post-retrieval relevance re-ranking.

Uses the LLM to re-rank retrieved chunks by relevance to the query.
Can be extended with cross-encoder models for faster re-ranking.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from research_assistant.config import get_settings
from research_assistant.llm_factory import get_llm

logger = logging.getLogger(__name__)

_RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a relevance judge. Given a query and a document chunk, "
        "rate how relevant the chunk is to answering the query on a scale of 0-10. "
        "Respond with ONLY a single integer number, nothing else."
    )),
    ("human", "Query: {query}\n\nDocument chunk:\n{chunk}"),
])


async def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 3,
) -> list[Document]:
    """Re-rank documents by relevance using the LLM.

    Args:
        query: The original search query.
        documents: List of candidate documents.
        top_k: Number of top documents to return after re-ranking.

    Returns:
        Top-k documents re-ranked by relevance.
    """
    if len(documents) <= top_k:
        return documents

    llm = get_llm(temperature=0)

    chain = _RERANK_PROMPT | llm

    scored_docs: list[tuple[Document, int]] = []
    for doc in documents:
        try:
            response = await chain.ainvoke({
                "query": query,
                "chunk": doc.page_content[:1500],  # Limit chunk size for re-ranking
            })
            score = int(response.content.strip())
            scored_docs.append((doc, score))
        except (ValueError, TypeError):
            # If parsing fails, give a neutral score
            scored_docs.append((doc, 5))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    reranked = [doc for doc, _ in scored_docs[:top_k]]
    logger.info(
        "Re-ranked %d → %d documents for query: '%s'",
        len(documents),
        len(reranked),
        query[:80],
    )
    return reranked
