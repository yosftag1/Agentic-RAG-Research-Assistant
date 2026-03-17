"""Citation tracking tool — formats and manages source citations."""

from __future__ import annotations

from langchain_core.documents import Document


def format_citations(documents: list[Document]) -> str:
    """Generate a citation list from documents.

    Args:
        documents: List of source documents.

    Returns:
        Formatted citation list.
    """
    seen_sources: set[str] = set()
    citations: list[str] = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown source")
        if source in seen_sources:
            continue
        seen_sources.add(source)

        meta = doc.metadata
        source_type = meta.get("source_type", "unknown")
        source_kind = meta.get("source_kind", "")
        title = meta.get("title", "")
        authors = meta.get("authors", "")
        year = meta.get("year", "")
        venue = meta.get("venue", "")
        citations_count = meta.get("citation_count", meta.get("citations", ""))
        url = meta.get("url", "")
        page = doc.metadata.get("page", None)

        citation = f"[{len(citations) + 1}] "
        if source_type == "web":
            if source_kind == "scholar":
                details: list[str] = []
                if authors:
                    details.append(str(authors))
                if year:
                    details.append(str(year))
                if venue:
                    details.append(str(venue))
                if citations_count not in ("", None):
                    details.append(f"Citations: {citations_count}")
                detail_text = f" ({'; '.join(details)})" if details else ""
                citation += f"Paper: {title or source}{detail_text}"
                if url:
                    citation += f". URL: {url}"
            else:
                label = title or source
                citation += f"Web page: {label}"
                if url and url != label:
                    citation += f". URL: {url}"
        elif source_type == "file":
            citation += f"File: {source}"
            if page is not None:
                citation += f", p. {page}"
        else:
            citation += source

        citations.append(citation)

    return "\n".join(citations) if citations else "No sources available."
