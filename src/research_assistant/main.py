"""CLI entry point — Typer app with ingest, query, and summarize commands."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.logging import RichHandler

app = typer.Typer(
    name="research-assistant",
    help="🔬 Personal Research Assistant — Multi-Agent RAG System",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )



@app.command()
def ingest(
    source: str = typer.Argument(
        ...,
        help="Path to a file, directory, or URL to ingest.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """📥 Ingest documents into the knowledge base.

    Supports PDF, TXT, MD files, directories, and web URLs.

    Examples:
        research-assistant ingest ./papers/attention.pdf
        research-assistant ingest ./data/
        research-assistant ingest https://arxiv.org/abs/2301.00001
    """
    _setup_logging(verbose)

    from research_assistant.ingestion.loader import load_documents
    from research_assistant.ingestion.splitter import split_documents
    from research_assistant.ingestion.embedder import embed_and_store

    with console.status("[bold green]Ingesting documents..."):
        # Load
        console.print(f"\n📄 Loading from: [cyan]{source}[/]")
        docs = load_documents(source)
        console.print(f"   → Loaded [bold]{len(docs)}[/] documents")

        # Split
        chunks = split_documents(docs)
        console.print(f"   → Split into [bold]{len(chunks)}[/] chunks")

        # Embed & store
        embed_and_store(chunks)
        console.print(f"   → Embedded and stored in ChromaDB ✅")

    console.print(
        Panel(
            f"[green]✅ Successfully ingested {len(chunks)} chunks from {source}[/]",
            title="Ingestion Complete",
        )
    )



@app.command()
def query(
    question: str = typer.Argument(
        ...,
        help="Your research question.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """🔍 Ask a question — the multi-agent system will research and answer.

    The orchestrator will classify your intent and route to the appropriate
    agents (retriever, summarizer, analyst, writer).

    Examples:
        research-assistant query "What are the key findings on attention mechanisms?"
        research-assistant query "Compare transformer and RNN architectures"
        research-assistant query "Write research notes on self-supervised learning"
    """
    _setup_logging(verbose)

    from research_assistant.agents.orchestrator import run_research_assistant

    console.print(f"\n🔬 Researching: [cyan]{question}[/]\n")

    with console.status("[bold blue]Agents working..."):
        answer = asyncio.run(run_research_assistant(question))

    console.print(Panel(Markdown(answer), title="Research Assistant", border_style="blue"))



@app.command()
def search(
    query_text: str = typer.Argument(
        ...,
        help="Search query for the local knowledge base.",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """🗂️ Search the local knowledge base (vector store only, no agents).

    Quick retrieval without agent processing.

    Examples:
        research-assistant search "attention mechanism" --top-k 3
    """
    _setup_logging(verbose)

    from research_assistant.retrieval.retriever import retrieve_with_scores

    with console.status("[bold green]Searching..."):
        results = retrieve_with_scores(query_text, top_k=top_k)

    if not results:
        console.print("[yellow]No results found.[/]")
        return

    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        console.print(
            Panel(
                f"[dim]Score: {score:.4f} | Source: {source}[/]\n\n"
                f"{doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}",
                title=f"Result {i}",
                border_style="green",
            )
        )



@app.command()
def info() -> None:
    """ℹ️  Show configuration and knowledge base stats."""
    from research_assistant.config import get_settings
    from research_assistant.ingestion.embedder import get_vectorstore

    settings = get_settings()

    console.print(Panel(
        f"[bold]LLM Model:[/]       {settings.llm_model}\n"
        f"[bold]Embedding:[/]       {settings.embedding_model}\n"
        f"[bold]Chunk Size:[/]      {settings.chunk_size}\n"
        f"[bold]Chunk Overlap:[/]   {settings.chunk_overlap}\n"
        f"[bold]Top-K:[/]           {settings.retrieval_top_k}\n"
        f"[bold]ChromaDB Path:[/]   {settings.chroma_persist_dir}\n"
        f"[bold]Data Dir:[/]        {settings.document_dir}",
        title="⚙️  Configuration",
    ))

    try:
        vs = get_vectorstore()
        collection = vs._collection
        count = collection.count()
        console.print(f"\n📊 Knowledge base: [bold]{count}[/] chunks indexed")
    except Exception:
        console.print("\n📊 Knowledge base: [yellow]not initialized[/]")


if __name__ == "__main__":
    app()
