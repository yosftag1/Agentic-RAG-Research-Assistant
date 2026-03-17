"""BM25 Keyword Search and Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from langchain_core.documents import Document

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

logger = logging.getLogger(__name__)


class BM25Index:
    """Keyword search index using BM25."""

    def __init__(self, documents: List[Document]):
        """Initialize and build the BM25 index from documents."""
        self.documents = documents
        if not BM25Okapi or not documents:
            self.index = None
            return

        # Simple tokenization: lowercasing and splitting by whitespace
        tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in documents
        ]
        self.index = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index with %d documents", len(documents))

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer."""
        return text.lower().replace("\n", " ").split()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search the BM25 index and return scored documents."""
        if not self.index or not self.documents:
            return []

        tokenized_query = self._tokenize(query)
        doc_scores = self.index.get_scores(tokenized_query)
        
        # Pair documents with their BM25 scores
        scored_docs = list(zip(self.documents, doc_scores))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]


def rrf_merge(
    semantic_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    k: int = 60
) -> List[Document]:
    """Fuse semantic and keyword results using Reciprocal Rank Fusion."""
    
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    
    # Process semantic results
    for rank, (doc, _) in enumerate(semantic_results):
        # We need a unique identifier for each chunk. If none, use page_content hash.
        doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Process BM25 results
    for rank, (doc, _) in enumerate(bm25_results):
        doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Sort docs by RRF score descending
    sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    merged = [doc_map[doc_id] for doc_id in sorted_docs]
    return merged
