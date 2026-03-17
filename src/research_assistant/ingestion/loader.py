"""Document loaders — PDF, Markdown, Web, plain text.

Provides a unified `load_documents` function that auto-detects format
and returns a list of LangChain `Document` objects.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)

logger = logging.getLogger(__name__)

# Map file extensions to loader classes
_EXTENSION_LOADERS: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_file(path: Path) -> list[Document]:
    """Load a single file and return its documents.

    Args:
        path: Path to the file to load.

    Returns:
        List of Document objects extracted from the file.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = path.suffix.lower()
    loader_cls = _EXTENSION_LOADERS.get(ext)
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {list(_EXTENSION_LOADERS.keys())}"
        )

    logger.info("Loading %s with %s", path, loader_cls.__name__)
    loader = loader_cls(str(path))
    docs = loader.load()

    # Tag each document with source metadata
    for doc in docs:
        doc.metadata["source"] = str(path)
        doc.metadata["source_type"] = "file"

    return docs


def load_directory(directory: Path, glob: str = "**/*") -> list[Document]:
    """Recursively load all supported files from a directory.

    Args:
        directory: Root directory to scan.
        glob: Glob pattern for file matching.

    Returns:
        List of Document objects from all supported files.
    """
    all_docs: list[Document] = []
    supported_exts = set(_EXTENSION_LOADERS.keys())

    for file_path in sorted(directory.glob(glob)):
        if file_path.is_file() and file_path.suffix.lower() in supported_exts:
            try:
                docs = load_file(file_path)
                all_docs.extend(docs)
                logger.info("  → loaded %d chunks from %s", len(docs), file_path.name)
            except Exception as e:
                logger.warning("  ✗ failed to load %s: %s", file_path.name, e)

    logger.info("Loaded %d documents from %s", len(all_docs), directory)
    return all_docs


def load_url(url: str) -> list[Document]:
    """Load content from a web URL.

    Args:
        url: The URL to fetch and parse.

    Returns:
        List of Document objects extracted from the web page.
    """
    logger.info("Loading web page: %s", url)
    loader = WebBaseLoader(url)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = url
        doc.metadata["source_type"] = "web"

    return docs


def load_documents(source: str) -> list[Document]:
    """Unified entry point — auto-detects if source is a file, directory, or URL.

    Args:
        source: Path to a file/directory, or a URL string.

    Returns:
        List of Document objects from the source.
    """
    if source.startswith(("http://", "https://")):
        return load_url(source)

    path = Path(source)
    if path.is_dir():
        return load_directory(path)
    elif path.is_file():
        return load_file(path)
    else:
        raise FileNotFoundError(f"Source not found: {source}")
