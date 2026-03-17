"""Document Registry — SQLite-based tracker for ingested files.

Records what has been embedded to prevent duplicates and allow deletion.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from research_assistant.config import get_settings

logger = logging.getLogger(__name__)


class DocumentRecord(BaseModel):
    """Record of an ingested document."""
    model_config = ConfigDict(from_attributes=True)

    source: str          # File path or URL
    file_hash: str       # SHA-256 hash of contents (for deduplication)
    chunk_count: int     # Number of chunks in ChromaDB
    ingested_at: str     # ISO format timestamp
    title: str = ""
    authors: str = ""    # JSON string of list
    year: int | None = None
    status: str = "active" # active or disabled


class DocumentRegistry:
    """SQLite-based registry for ingested documents."""

    def __init__(self, db_path: str | Path | None = None):
        """Initialize and create schema if needed."""
        settings = get_settings()
        self.db_path = Path(db_path or settings.doc_registry_path)
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    source TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL,
                    title TEXT,
                    authors TEXT,
                    year INTEGER,
                    status TEXT DEFAULT 'active'
                )
            ''')
            # Migration back-compat: add status column if missing for existing database files
            try:
                conn.execute("ALTER TABLE documents ADD COLUMN status TEXT DEFAULT 'active'")
            except sqlite3.OperationalError:
                pass
                
            # Index for quick hash lookups
            conn.execute('CREATE INDEX IF NOT EXISTS idx_hash ON documents(file_hash)')

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, record: DocumentRecord) -> None:
        """Register a new document."""
        with self._get_conn() as conn:
            conn.execute(
                '''
                INSERT OR REPLACE INTO documents 
                (source, file_hash, chunk_count, ingested_at, title, authors, year, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    record.source,
                    record.file_hash,
                    record.chunk_count,
                    record.ingested_at,
                    record.title,
                    record.authors,
                    record.year,
                    record.status,
                )
            )
        logger.info("Registered document: %s (%d chunks)", record.source, record.chunk_count)

    def is_hash_indexed(self, file_hash: str) -> bool:
        """Check if a file with this hash has already been ingested and is active."""
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT 1 FROM documents WHERE file_hash = ? AND status = 'active'", (file_hash,))
            return cursor.fetchone() is not None

    def list_all(self) -> list[DocumentRecord]:
        """List all ingested documents, ordered by newest first."""
        with self._get_conn() as conn:
            cursor = conn.execute('SELECT * FROM documents ORDER BY ingested_at DESC')
            return [DocumentRecord.model_validate(dict(row)) for row in cursor]

    def delete(self, source: str) -> bool:
        """Remove a document from the registry.
        
        Note: The actual deletion from ChromaDB must be handled separately.
        
        Returns:
            True if a record was actually deleted.
        """
        with self._get_conn() as conn:
            cursor = conn.execute('DELETE FROM documents WHERE source = ?', (source,))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("Removed document from registry: %s", source)
            return deleted

    def disable(self, source: str) -> bool:
        """Mark a document as disabled (keeps record but marks inactive)."""
        with self._get_conn() as conn:
            cursor = conn.execute("UPDATE documents SET status = 'disabled' WHERE source = ?", (source,))
            disabled = cursor.rowcount > 0
            if disabled:
                logger.info("Disabled document in registry: %s", source)
            return disabled

    def enable(self, source: str) -> bool:
        """Mark a document as active."""
        with self._get_conn() as conn:
            cursor = conn.execute("UPDATE documents SET status = 'active' WHERE source = ?", (source,))
            enabled = cursor.rowcount > 0
            if enabled:
                logger.info("Enabled document in registry: %s", source)
            return enabled

    def reset_all(self) -> int:
        """Mark all active documents as 'disabled' so they can be re-ingested.

        Used after a collection reset (e.g. embedding model change) to unblock
        re-ingestion without losing the historical record.

        Returns:
            Number of documents marked as disabled.
        """
        with self._get_conn() as conn:
            cursor = conn.execute("UPDATE documents SET status = 'disabled' WHERE status = 'active'")
            count = cursor.rowcount
            if count:
                logger.info("Reset %d document records to 'disabled'", count)
            return count

    def clear_all(self) -> int:
        """Remove all registry records.

        Used when an embedding model change invalidates the entire vector index and
        the stored document entries are no longer recoverable as active documents.
        """
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM documents")
            count = cursor.rowcount
            if count:
                logger.info("Cleared %d document records from registry", count)
            return count

    def get_stats(self) -> dict:
        """Get global stats about ingested documents."""
        with self._get_conn() as conn:
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as doc_count, 
                    COALESCE(SUM(chunk_count), 0) as total_chunks 
                FROM documents
            ''')
            row = cursor.fetchone()
            return dict(row)

# Module-level singleton
_registry: DocumentRegistry | None = None

def get_registry() -> DocumentRegistry:
    """Get the cached registry instance."""
    global _registry
    if _registry is None:
        _registry = DocumentRegistry()
    return _registry
