"""Google Scholar search tool."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    from scholarly import scholarly
except ImportError:
    scholarly = None

logger = logging.getLogger(__name__)


def _derive_pdf_url(eprint_url: str) -> str:
    """Derive a direct PDF URL from an eprint/arXiv URL."""
    if not eprint_url:
        return ""
    if "arxiv.org/abs/" in eprint_url:
        arxiv_id = eprint_url.split("arxiv.org/abs/")[-1].rstrip("/")
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if "arxiv.org/pdf/" in eprint_url:
        return eprint_url if eprint_url.endswith(".pdf") else eprint_url + ".pdf"
    if eprint_url.endswith(".pdf"):
        return eprint_url
    return ""


class ScholarSearchInput(BaseModel):
    query: str = Field(description="The search query, keywords, or paper title.")
    max_results: int = Field(default=5, description="Maximum number of results to return.")


@tool("scholar_search", args_schema=ScholarSearchInput)
def search_scholar(query: str, max_results: int = 5) -> str:
    """Search Google Scholar for academic papers and return structured metadata.
    
    Use this tool when the user queries for scientific literature, research papers,
    authors, or academic topics that require scholarly sources.
    """
    results = search_scholar_raw(query, max_results)
    
    if isinstance(results, str):
        return results
        
    if not results:
        return "No results found on Google Scholar for this query."
        
    formatted = []
    for r in results:
        result_str = (
            f"Title: {r['title']}\n"
            f"Authors: {r['authors']}\n"
            f"Venue: {r['venue']} ({r['pub_year']})\n"
            f"Citations: {r['citations']}\n"
            f"Abstract: {r['abstract']}\n"
            f"URL: {r['url']}"
        )
        formatted.append(result_str)
        
    return "\n\n---\n\n".join(formatted)


def search_scholar_raw(query: str, max_results: int = 5) -> list[dict] | str:
    """Search Google Scholar and return raw dictionary results. Useful for APIs."""
    if scholarly is None:
        return "Error: 'scholarly' package is not installed."
        
    logger.info("Searching Google Scholar for: '%s'", query)
    try:
        search_query = scholarly.search_pubs(query)
        results = []
        
        for _ in range(max_results):
            try:
                pub = next(search_query)
                bib = pub.get("bib", {})
                
                title = bib.get("title", "Unknown Title")
                authors = ", ".join(bib.get("author", ["Unknown Author"]))
                venue = bib.get("venue", "Unknown Venue")
                pub_year = bib.get("pub_year", "Unknown Year")
                citations = pub.get("num_citations", 0)
                abstract = bib.get("abstract", "No abstract available.")
                url = pub.get("pub_url", "")
                eprint_url = pub.get("eprint_url", "")
                pdf_url = _derive_pdf_url(eprint_url)

                results.append({
                    "title": title,
                    "authors": authors,
                    "venue": venue,
                    "pub_year": pub_year,
                    "citations": citations,
                    "abstract": abstract,
                    "url": url,
                    "eprint_url": eprint_url,
                    "pdf_url": pdf_url,
                })
            except StopIteration:
                break
                
        return results
    except Exception as e:
        logger.error("Scholar search failed: %s", str(e))
        return f"Error occurred while searching Google Scholar: {str(e)}"


def search_scholar_multi(query: str, max_results: int = 5) -> list[dict] | str:
    """Search Google Scholar with multiple query variants for broader coverage.

    Tries the original query plus two variants, deduplicates by title, and
    returns up to 2× max_results unique papers.
    """
    if scholarly is None:
        return "Error: 'scholarly' package is not installed."

    words = query.split()
    variants = [query, query + " survey review"]
    if len(words) >= 2:
        variants.append(query + " recent advances")

    seen_titles: set[str] = set()
    all_results: list[dict] = []

    for variant in variants:
        batch = search_scholar_raw(variant, max_results)
        if isinstance(batch, str):
            continue  # skip error responses
        for r in batch:
            key = r["title"].lower().strip()
            if key not in seen_titles:
                seen_titles.add(key)
                all_results.append(r)
        if len(all_results) >= max_results * 2:
            break

    if not all_results:
        return "No results found on Google Scholar for this query."

    return all_results


def fetch_pdf_semantic_scholar(title: str) -> str | None:
    """Query Semantic Scholar to find an open-access PDF URL for a paper.

    Returns the PDF URL string, or None if not found.
    """
    try:
        encoded = urllib.parse.quote(title[:150])
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={encoded}&fields=openAccessPdf,externalIds&limit=1"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchAssistant/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            papers = data.get("data", [])
            if papers:
                paper = papers[0]
                open_access = paper.get("openAccessPdf")
                if open_access and open_access.get("url"):
                    return open_access["url"]
                external_ids = paper.get("externalIds", {})
                arxiv_id = external_ids.get("ArXiv")
                if arxiv_id:
                    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    except Exception as e:
        logger.warning("Semantic Scholar PDF lookup failed for '%s': %s", title, e)

    return None
