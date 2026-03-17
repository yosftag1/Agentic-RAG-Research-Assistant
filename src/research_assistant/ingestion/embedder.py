"""Embedding and ChromaDB storage.

Handles embedding document chunks and upserting them into ChromaDB.
Now integrated with SQLite Document Registry to prevent duplicates.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from research_assistant.config import get_settings
from research_assistant.llm_factory import get_embeddings
from research_assistant.ingestion.registry import get_registry, DocumentRecord

logger = logging.getLogger(__name__)

# Module-level cache for the vector store instance
_vectorstore: Chroma | None = None


def get_stored_collection_dimension(
    persist_dir: str | Path | None = None,
    collection_name: str = "research_docs",
) -> int | None:
    """Read the embedding dimension stored in an existing ChromaDB collection.

    Returns None if the collection does not exist yet.
    """
    import sqlite3

    settings = get_settings()
    p_dir = Path(persist_dir or settings.chroma_persist_dir)
    sqlite_path = p_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(sqlite_path))
        cur = conn.cursor()
        cur.execute("SELECT dimension FROM collections WHERE name = ?", (collection_name,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def clear_vectorstore_cache() -> None:
    """Clear the module-level vectorstore cache to force re-initialisation on next use."""
    global _vectorstore
    _vectorstore = None
    logger.info("ChromaDB vectorstore cache cleared.")


def reset_collection(
    collection_name: str = "research_docs",
    persist_dir: str | Path | None = None,
) -> None:
    """Delete all vectors from the ChromaDB collection and clear the registry.

    Called when the user switches embedding models.  The registry entries are
    removed because the old index is no longer recoverable under the new embedding model.
    """
    global _vectorstore
    _vectorstore = None

    settings = get_settings()
    p_dir = str(persist_dir or settings.chroma_persist_dir)

    try:
        import chromadb

        client = chromadb.PersistentClient(path=p_dir)
        client.delete_collection(collection_name)
        logger.info("Deleted ChromaDB collection '%s'", collection_name)
    except Exception as exc:
        logger.warning("Could not delete ChromaDB collection: %s", exc)

    # Remove all registry entries because the old index is invalid under the new embedding model.
    from research_assistant.ingestion.registry import get_registry

    registry = get_registry()
    count = registry.clear_all()
    logger.info("Removed %d registry records after collection reset.", count)


def get_embedding_function() -> Any:
    """Create and return the embedding function based on settings."""
    return get_embeddings()


def get_vectorstore(
    persist_directory: str | Path | None = None,
    collection_name: str = "research_docs",
) -> Chroma:
    """Get or create the ChromaDB vector store."""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    settings = get_settings()
    persist_dir = str(persist_directory or settings.chroma_persist_dir)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_function()

    _vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    logger.info("Initialized ChromaDB at %s (collection: %s)", persist_dir, collection_name)
    return _vectorstore


def compute_hash(content: str) -> str:
    """Compute SHA-256 hash of a string to detect duplicates."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def embed_and_store(
    chunks: list[Document],
    collection_name: str = "research_docs",
) -> Chroma | None:
    """Embed document chunks and store them in ChromaDB.
    
    Checks the document registry first to skip already-ingested files.

    Args:
        chunks: List of document chunks to embed and store.
        collection_name: ChromaDB collection name.

    Returns:
        The Chroma vector store instance, or None if skipped.
    """
    if not chunks:
        return None

    # We assume all chunks in the list come from the same source document.
    # In a mixed batch, you'd want to group them by source first.
    source = chunks[0].metadata.get("source", "unknown")
    
    # Compute a hash of the full unchunked document content
    full_content = "\n".join(chunk.page_content for chunk in chunks)
    file_hash = compute_hash(full_content)

    registry = get_registry()
    if registry.is_hash_indexed(file_hash):
        logger.info("Skipping %s — already ingested (hash match).", source)
        return get_vectorstore(collection_name=collection_name)

    vectorstore = get_vectorstore(collection_name=collection_name)
    
    # Cleanup any existing chunks for this source to ensure no duplicates 
    # (e.g., if re-ingesting a previously disabled or partially-ingested doc)
    collection = vectorstore._collection
    results = collection.get(where={"source": source})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])

    # Add documents in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        logger.info(
            "  → embedded batch %d–%d of %d",
            i + 1,
            min(i + batch_size, len(chunks)),
            len(chunks),
        )

    # Extract some metadata for the registry record
    m = chunks[0].metadata
    title = m.get("title", Path(source).name if "://" not in source else source)
    
    # Register the successful ingestion
    record = DocumentRecord(
        source=source,
        file_hash=file_hash,
        chunk_count=len(chunks),
        ingested_at=datetime.utcnow().isoformat() + "Z",
        title=title,
        authors=json.dumps(m.get("authors", [])),
        year=m.get("year", None)
    )
    registry.add(record)

    logger.info("Stored %d chunks in ChromaDB and updated registry.", len(chunks))
    return vectorstore


def delete_document(
    source: str,
    collection_name: str = "research_docs",
) -> bool:
    """Delete a document from ChromaDB and the SQLite registry.
    
    Returns:
        True if the document was found and deleted, False otherwise.
    """
    registry = get_registry()
    if not registry.delete(source):
        return False

    settings = get_settings()
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        collection = client.get_collection(collection_name)
        results = collection.get(where={"source": source})
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info("Deleted %d chunks from ChromaDB for source: %s", len(results["ids"]), source)
    except Exception as exc:
        logger.warning("Could not delete source %s from ChromaDB collection %s: %s", source, collection_name, exc)
        
    return True


def disable_document(
    source: str,
    collection_name: str = "research_docs",
) -> bool:
    """Disable a document in the registry and remove its chunks from ChromaDB.
    
    Returns:
        True if the document was found and disabled, False otherwise.
    """
    registry = get_registry()
    if not registry.disable(source):
        return False

    settings = get_settings()
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        collection = client.get_collection(collection_name)
        results = collection.get(where={"source": source})
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info("Deleted %d chunks from ChromaDB for disabled source: %s", len(results["ids"]), source)
    except Exception as exc:
        logger.warning("Could not disable source %s in ChromaDB collection %s: %s", source, collection_name, exc)
        
    return True


def reset_vectorstore() -> None:
    """Reset the cached vectorstore instance."""
    global _vectorstore
    _vectorstore = None
