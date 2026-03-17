from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from research_assistant.config import get_settings
from research_assistant.agents.orchestrator import run_research_assistant, run_research_assistant_stream
from research_assistant.ingestion.loader import load_documents
from research_assistant.ingestion.splitter import split_documents
from research_assistant.ingestion.embedder import embed_and_store, disable_document
from research_assistant.retrieval.retriever import retrieve_from_vectorstore, format_context
from research_assistant.ingestion.registry import get_registry

app = FastAPI(
    title="Research Assistant API",
    description="🔬 Personal Research Assistant — Multi-Agent RAG System",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent.parent / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class QueryRequest(BaseModel):
    question: str
    history: list[dict] | None = None
    source_preference: str = "auto"


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    web_search_results: list[dict] | None = None
    steps: list[dict] | None = None


def _dimension_mismatch_detail(err: str) -> dict | None:
    """Return a structured error payload for embedding dimension mismatches."""
    if "dimension" in err.lower() and ("expecting" in err.lower() or "got" in err.lower()):
        return {
            "type": "embedding_dimension_mismatch",
            "message": (
                "The embedding model has changed since your documents were indexed. "
                "Please reset the collection in Settings → Reset Collection, "
                "then re-ingest your documents."
            ),
            "raw": err,
        }
    return None


def _provider_auth_detail(err: str) -> dict | None:
    """Return a structured error payload for invalid provider credentials."""
    lower = err.lower()
    if "api_key_invalid" in lower or "api key not valid" in lower:
        return {
            "type": "provider_auth_error",
            "message": (
                "Google API key is invalid for the selected Gemini provider. "
                "Set a valid GOOGLE_API_KEY (or GEMINI_API_KEY) in Render environment variables, "
                "ensure no extra quotes/spaces, then redeploy."
            ),
            "raw": err,
        }
    return None


class IngestResponse(BaseModel):
    chunks_ingested: int
    message: str
    doc_hash: str | None = None
    status: str | None = None


class DocumentResponse(BaseModel):
    source: str
    file_hash: str
    chunk_count: int
    ingested_at: str
    status: str


class SettingsUpdateRequest(BaseModel):
    llm_provider: str | None = None
    embedding_provider: str | None = None
    search_strategy: str | None = None
    source_preference: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    ollama_base_url: str | None = None
    llm_model: str | None = None
    embedding_model: str | None = None


class SettingsResponse(BaseModel):
    settings: dict
    collection_reset: bool = False


class ModelsRequest(BaseModel):
    provider: str
    base_url: str | None = None


class ModelsResponse(BaseModel):
    models: list[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    content: str
    source: str
    score: float | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


class TextIngestRequest(BaseModel):
    text: str
    title: str
    source_url: str


class IngestFromURLRequest(BaseModel):
    url: str
    title: str


class FindPDFRequest(BaseModel):
    title: str
    eprint_url: str = ""


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5


class WebSearchResponse(BaseModel):
    results: list[dict]
    error: str | None = None

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main web application."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            content="Research Assistant API is running. Frontend assets are not bundled in this deploy.",
            status_code=200,
        )
    return HTMLResponse(content=index_path.read_text(), status_code=200)

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Ask a research question — routed through the multi-agent pipeline."""
    try:
        effective_preference = request.source_preference or get_settings().source_preference
        result = await run_research_assistant(
            request.question,
            history=request.history or [],
            source_preference=effective_preference,
        )
        if isinstance(result, dict):
            answer = result.get("final_answer", "")
            sources = result.get("sources", [])
            web_results = result.get("web_search_results", [])
            steps = result.get("steps", [])
        else:
            answer = str(result)
            sources = []
            web_results = []
            steps = []

        return QueryResponse(
            answer=answer,
            sources=sources,
            web_search_results=web_results,
            steps=steps,
        )
    except Exception as e:
        err = str(e)
        # Surface a specific, actionable error for embedding dimension mismatches
        mismatch = _dimension_mismatch_detail(err)
        if mismatch is not None:
            raise HTTPException(
                status_code=409,
                detail=mismatch,
            )
        auth_error = _provider_auth_detail(err)
        if auth_error is not None:
            raise HTTPException(status_code=401, detail=auth_error)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """Stream live agent progress and the final answer as NDJSON lines."""

    async def event_stream():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        async def progress_callback(event: dict) -> None:
            await queue.put(event)

        async def runner() -> None:
            try:
                effective_preference = request.source_preference or get_settings().source_preference
                result = await run_research_assistant_stream(
                    request.question,
                    history=request.history or [],
                    source_preference=effective_preference,
                    progress_callback=progress_callback,
                )
                await queue.put({
                    "type": "final",
                    "data": {
                        "answer": result.get("final_answer", ""),
                        "sources": result.get("sources", []),
                        "web_search_results": result.get("web_search_results", []),
                        "steps": result.get("steps", []),
                    },
                })
            except Exception as exc:
                err = str(exc)
                auth_error = _provider_auth_detail(err)
                await queue.put({
                    "type": "error",
                    "detail": _dimension_mismatch_detail(err) or auth_error or {"message": err},
                })
            finally:
                await queue.put({"type": "done"})

        task = asyncio.create_task(runner())
        try:
            while True:
                event = await queue.get()
                yield json.dumps(event) + "\n"
                if event.get("type") == "done":
                    break
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    import contextlib

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(file: UploadFile = File(...)):
    """Ingest a document file (PDF/TXT/MD) uploaded from the frontend."""
    try:
        import shutil
        import tempfile
        from pathlib import Path

        # Create a temporary file to hold the upload
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        docs = load_documents(tmp_path)
        
        # We want to preserve the original filename in the registry, not the temp path
        for doc in docs:
            doc.metadata["source"] = file.filename
            
        chunks = split_documents(docs)
        embed_and_store(chunks)

        registry = get_registry()
        record_hash = None
        status = "active"
        if docs and hasattr(docs[0], 'metadata'):
           import hashlib
           try:
               content = Path(tmp_path).read_bytes()
               record_hash = hashlib.sha256(content).hexdigest()
           except Exception:
               pass
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
               
        return IngestResponse(
            chunks_ingested=len(chunks),
            message=f"Successfully ingested {len(chunks)} chunks from {file.filename}",
            doc_hash=record_hash,
            status=status
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """Quick vector search without agent processing."""
    try:
        from research_assistant.retrieval.retriever import retrieve_with_scores

        results = retrieve_with_scores(request.query, top_k=request.top_k)
        return SearchResponse(
            results=[
                SearchResult(
                    content=doc.page_content[:1000],
                    source=doc.metadata.get("source", "Unknown"),
                    score=score,
                )
                for doc, score in results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/web-search", response_model=WebSearchResponse)
async def web_search_endpoint(request: WebSearchRequest):
    """Search Google Scholar with multiple query variants. Enriches results with PDF URLs."""
    import asyncio
    try:
        from research_assistant.tools.scholar_tool import (
            search_scholar_multi,
            fetch_pdf_semantic_scholar,
        )

        results = await asyncio.to_thread(search_scholar_multi, request.query, request.max_results)

        if isinstance(results, str):
            return WebSearchResponse(results=[], error=results)

        # Enrich results that don't already have a pdf_url with Semantic Scholar (parallel)
        async def _enrich_one(paper: dict) -> dict:
            if not paper.get("pdf_url"):
                pdf_url = await asyncio.to_thread(fetch_pdf_semantic_scholar, paper["title"])
                if pdf_url:
                    paper["pdf_url"] = pdf_url
            return paper

        enriched = await asyncio.gather(*[_enrich_one(r) for r in results])
        return WebSearchResponse(results=list(enriched))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/paper/find-pdf")
async def find_paper_pdf(request: FindPDFRequest):
    """Try to find a direct open-access PDF URL for a paper."""
    import asyncio
    from research_assistant.tools.scholar_tool import _derive_pdf_url, fetch_pdf_semantic_scholar

    # First try derivation from eprint_url (instant, no network call)
    if request.eprint_url:
        pdf_url = _derive_pdf_url(request.eprint_url)
        if pdf_url:
            return {"pdf_url": pdf_url}

    # Fall back to Semantic Scholar API
    pdf_url = await asyncio.to_thread(fetch_pdf_semantic_scholar, request.title)
    return {"pdf_url": pdf_url}


@app.post("/ingest/from-url", response_model=IngestResponse)
async def ingest_from_url_endpoint(request: IngestFromURLRequest):
    """Download a PDF from a URL and ingest it into the knowledge base."""
    import asyncio
    import tempfile
    import urllib.request
    from pathlib import Path

    def _download_and_ingest() -> int:
        req = urllib.request.Request(
            request.url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAssistant/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
            content_type = response.headers.get("Content-Type", "")

        suffix = ".html" if "html" in content_type else ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            docs = load_documents(tmp_path)
            for doc in docs:
                doc.metadata["source"] = request.url
                doc.metadata["title"] = request.title
            chunks = split_documents(docs)
            embed_and_store(chunks)
            return len(chunks)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    try:
        chunk_count = await asyncio.to_thread(_download_and_ingest)
        return IngestResponse(
            chunks_ingested=chunk_count,
            message=f"Successfully ingested {chunk_count} chunks from '{request.title}'",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest from URL: {str(e)}")


@app.post("/ingest/text")
async def ingest_text_endpoint(request: TextIngestRequest):
    """Ingest raw text (e.g. from a web search result abstract) into the database."""
    try:
        import tempfile
        from pathlib import Path

        # Create a temporary file to hold the text content so we can reuse the normal loaders
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", dir="/tmp", mode="w", encoding="utf-8") as tmp:
            tmp.write(request.text)
            tmp_path = tmp.name

        docs = load_documents(tmp_path)
        
        # Override the metadata to point to our virtual source
        virtual_source = request.source_url if request.source_url else f"Web: {request.title}"
        for doc in docs:
            doc.metadata["source"] = virtual_source
            doc.metadata["title"] = request.title
            
        chunks = split_documents(docs)
        embed_and_store(chunks)

        registry = get_registry()
        record_hash = None
        status = "active"
        if docs and hasattr(docs[0], 'metadata'):
           import hashlib
           try:
               content = Path(tmp_path).read_bytes()
               record_hash = hashlib.sha256(content).hexdigest()
           except Exception:
               pass
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
               
        return {
            "chunks_ingested": len(chunks),
            "source": virtual_source,
            "message": f"Successfully ingested {len(chunks)} chunks from {request.title}",
            "status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    settings = get_settings()
    registry = get_registry()
    stats = registry.get_stats()
    
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "search_strategy": settings.search_strategy,
        "settings": settings.model_dump(),
        "stats": stats
    }


@app.get("/documents", response_model=list[DocumentResponse])
async def list_documents():
    """Fetch all documents from the registry."""
    try:
        registry = get_registry()
        docs = registry.list_all()
        return [
            DocumentResponse(
                source=doc.source,
                file_hash=doc.file_hash,
                chunk_count=doc.chunk_count,
                ingested_at=doc.ingested_at,
                status=doc.status
            )
            for doc in docs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/toggle", response_model=dict)
async def toggle_document(request: dict):
    """Toggle a document between active and disabled using its source path."""
    source = request.get("source")
    if not source:
        raise HTTPException(status_code=400, detail="Missing source")
        
    try:
        registry = get_registry()
        # Find the current status
        docs = registry.list_all()
        target_doc = next((d for d in docs if d.source == source), None)
        
        if not target_doc:
            raise HTTPException(status_code=404, detail="Document not found")
            
        if target_doc.status == "active":
            # Strip from chromadb
            disable_document(source)
            new_status = "disabled"
        else:
            # Re-embed needs the actual file
            # Quick approach: call load/split/embed manually
            # But wait: if it's a URL or web ingested text we don't have the file anymore!
            # For this MVP, we try to load it from the filesystem if it's a path. If it fails, we catch it.
            try:
                loaded = load_documents(source)
                chunks = split_documents(loaded)
                embed_and_store(chunks)
                registry.enable(source)
                new_status = "active"
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Cannot re-enable this source automatically. "
                        f"Please re-upload or re-ingest it with the current embedding model. Error: {e}"
                    ),
                )
            
        return {"message": f"Document {new_status}", "status": new_status, "source": source}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/delete")
async def delete_document_endpoint(source: str):
    """Permanently delete a document from the system."""
    if not source:
        raise HTTPException(status_code=400, detail="Missing source parameter")
        
    try:
        from research_assistant.ingestion.embedder import delete_document
        success = delete_document(source)
        if success:
            return {"message": f"Document {source} permanently deleted."}
        else:
            raise HTTPException(status_code=404, detail="Document not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/settings", response_model=SettingsResponse)
async def update_settings(request: SettingsUpdateRequest):
    """Update runtime configuration parameters."""
    try:
        settings = get_settings()
        previous_embedding_provider = settings.embedding_provider
        previous_embedding_model = settings.embedding_model
        collection_reset = False
        
        if request.llm_provider:
            settings.llm_provider = request.llm_provider
        if request.embedding_provider:
            settings.embedding_provider = request.embedding_provider
        if request.search_strategy:
            settings.search_strategy = request.search_strategy
        if request.source_preference:
            settings.source_preference = request.source_preference
        if request.openai_api_key:
            settings.openai_api_key = request.openai_api_key
        if request.google_api_key:
            settings.google_api_key = request.google_api_key
        if request.ollama_base_url:
            settings.ollama_base_url = request.ollama_base_url
        if request.llm_model:
            settings.llm_model = request.llm_model
        if request.embedding_model:
            settings.embedding_model = request.embedding_model

        embedding_changed = (
            (request.embedding_provider and request.embedding_provider != previous_embedding_provider) or
            (request.embedding_model and request.embedding_model != previous_embedding_model)
        )

        # If the embedding backend changed, reset the collection so stale vectors
        # do not remain bound to the old embedding dimension.
        if request.embedding_provider or request.embedding_model:
            from research_assistant.ingestion.embedder import clear_vectorstore_cache, reset_collection
            clear_vectorstore_cache()
            if embedding_changed:
                reset_collection()
                collection_reset = True
            
        # Optional: Save to .env programmatically for persistence, but memory change is fine for runtime testing
        return SettingsResponse(
            settings=settings.model_dump(exclude={"openai_api_key", "google_api_key", "tavily_api_key"}),
            collection_reset=collection_reset,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset-collection")
async def reset_collection_endpoint():
    """Delete the ChromaDB collection and remove stale document records.

    Use this after changing the embedding model to avoid dimension-mismatch errors.
    Existing document entries are removed because they must be re-uploaded / re-ingested
    to rebuild the vector index with the new model.
    """
    try:
        from research_assistant.ingestion.embedder import reset_collection
        reset_collection()
        return {
            "message": (
                "Collection reset successfully. "
                "Previous indexed documents were removed. Please re-ingest your documents to rebuild the index with the current embedding model."
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models", response_model=ModelsResponse)
async def fetch_models(request: ModelsRequest):
    """Fetch available models dynamically from the provider."""
    try:
        import urllib.request
        import json
        
        if request.provider == "ollama":
            url = request.base_url or get_settings().ollama_base_url
            # Clean up the URL
            url = url.rstrip("/")
            req = urllib.request.Request(f"{url}/api/tags")
            try:
                with urllib.request.urlopen(req, timeout=3) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode('utf-8'))
                        models = [m.get("name") for m in data.get("models", [])]
                        return ModelsResponse(models=models)
                    else:
                        raise Exception(f"Failed to fetch. Status code: {response.status}")
            except Exception as outer_e:
                 # It failed, raise the error so we can see what went wrong
                 raise Exception(f"Failed to connect to Ollama at {url}: {str(outer_e)}")
                 
        elif request.provider == "gemini":
            # Return standard Gemini models statically to avoid needing API key just to list them
            gemini_models = [
                "gemini-2.0-flash",
                "gemini-2.5-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "models/text-embedding-004",
                "models/gemini-embedding-001"
            ]
            return ModelsResponse(models=gemini_models)
            
        return ModelsResponse(models=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
