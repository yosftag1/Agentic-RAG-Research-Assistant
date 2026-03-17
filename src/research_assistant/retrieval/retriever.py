"""Retriever — queries ChromaDB and optionally falls back to web search.

Provides the core retrieval logic used by the Retriever Agent.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from research_assistant.config import get_settings
from research_assistant.ingestion.embedder import get_vectorstore
from research_assistant.retrieval.bm25_index import BM25Index, rrf_merge

logger = logging.getLogger(__name__)


def retrieve_from_vectorstore(
    query: str,
    top_k: int | None = None,
    collection_name: str = "research_docs",
    filter_: dict | None = None,
) -> list[Document]:
    """Retrieve relevant document chunks from the vector store."""
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    strategy = settings.search_strategy

    vectorstore = get_vectorstore(collection_name=collection_name)
    
    if strategy == "hybrid":
        return hybrid_search(query, top_k, collection_name, filter_)
        
    if strategy == "mmr":
        # Used Chroma Native MMR Search
        results = vectorstore.max_marginal_relevance_search(
            query, k=top_k, fetch_k=top_k * 4, lambda_mult=1.0 - settings.mmr_diversity, filter=filter_
        )
        logger.info("Retrieved %d chunks via MMR for query: '%s'", len(results), query[:80])
        return results

    # Default: Semantic search
    results = vectorstore.similarity_search(query, k=top_k, filter=filter_)
    logger.info("Retrieved %d chunks via Semantic Search for query: '%s'", len(results), query[:80])
    return results

def hybrid_search(
    query: str, 
    top_k: int = 5, 
    collection_name: str = "research_docs",
    filter_: dict | None = None
) -> list[Document]:
    """Perform a hybrid search combining semantic (vector) and keyword (BM25) search."""
    vectorstore = get_vectorstore(collection_name=collection_name)
    
    # Semantic Search
    semantic_results = vectorstore.similarity_search_with_score(
        query, k=max(top_k * 3, 10), filter=filter_
    )
    
    # BM25 Keyword Search
    # Fetch all docs from Chroma for dynamic BM25 (Note: optimize for large indices later)
    collection = vectorstore._collection
    all_data = collection.get(include=["documents", "metadatas"], where=filter_)
    
    all_docs = []
    if all_data and all_data.get("documents"):
        for text, meta in zip(all_data["documents"], all_data["metadatas"]):
            all_docs.append(Document(page_content=text, metadata=meta))
            
    bm25 = BM25Index(all_docs)
    bm25_results = bm25.search(query, top_k=max(top_k * 3, 10))
    
    # Merge with Reciprocal Rank Fusion
    merged = rrf_merge(semantic_results, bm25_results, k=60)
    
    final_results = merged[:top_k]
    logger.info("Retrieved %d chunks via Hybrid Search for query: '%s'", len(final_results), query[:80])
    return final_results


def retrieve_with_scores(
    query: str,
    top_k: int | None = None,
    collection_name: str = "research_docs",
) -> list[tuple[Document, float]]:
    """Retrieve documents with similarity scores.

    Args:
        query: The search query string.
        top_k: Number of results to return.
        collection_name: ChromaDB collection to search.

    Returns:
        List of (Document, score) tuples, ranked by similarity.
    """
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k

    vectorstore = get_vectorstore(collection_name=collection_name)
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    logger.info(
        "Retrieved %d chunks with scores for query: '%s'",
        len(results),
        query[:80],
    )
    return results


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a context string for the LLM.

    Args:
        documents: Retrieved document chunks.

    Returns:
        A formatted string with source citations.
    """
    if not documents:
        return "No relevant documents found."

    context_parts: list[str] = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        title = meta.get("title", source)
        authors = meta.get("authors", "")
        year = meta.get("year", "")
        venue = meta.get("venue", "")
        citations = meta.get("citation_count", "")
        
        # Build a rich citation string
        auth_str = f" — {authors}" if authors and authors != "[]" else ""
        year_str = f" ({year})" if year else ""
        venue_str = f", {venue}" if venue else ""
        cite_str = f". Cited by: {citations}" if citations else ""
        
        source_info = f'{title}{auth_str}{year_str}{venue_str}{cite_str} [Source: {source}]'
        
        context_parts.append(
            f"[Source {i}: {source_info}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)
