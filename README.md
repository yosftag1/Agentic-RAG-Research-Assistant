# 🔬 Personal Research Assistant

> Multi-agent RAG system for personal research — ingest papers, ask questions, get cited answers.

## Architecture

```
User → Orchestrator → [Retriever] → ChromaDB / Web
                    → [Summarizer] → LLM
                    → [Analyst]    → LLM
                    → [Writer]     → LLM
                    → Merge → Final Answer
```

**5 Agents** working together via LangGraph:
- **Orchestrator** — classifies intent, routes to agents, merges results
- **Retriever** — searches local vector store + web fallback
- **Summarizer** — single/multi-document summarization
- **Analyst** — comparative analysis with structured output
- **Writer** — drafts research notes, reviews, reports

## Quick Start

### 1. Install

```bash
cd research-assistant
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Ingest documents

```bash
# Single file
research-assistant ingest ./data/paper.pdf

# Entire directory
research-assistant ingest ./data/

# Web page
research-assistant ingest https://example.com/article
```

### 4. Ask questions

```bash
# Factual lookup
research-assistant query "What are the main findings on attention mechanisms?"

# Summarize
research-assistant query "Summarize the key contributions of this paper"

# Analyze
research-assistant query "Compare transformer and RNN architectures"

# Draft
research-assistant query "Write research notes on self-supervised learning"
```

### 5. Quick search (no agents)

```bash
research-assistant search "attention mechanism" --top-k 3
```

### 6. API server (optional)

```bash
uvicorn research_assistant.api.server:app --reload
# Docs at http://localhost:8000/docs
```

## Project Structure

```
src/research_assistant/
├── config.py              # Settings (pydantic-settings)
├── main.py                # CLI (Typer)
├── ingestion/             # Document loading → chunking → embedding
├── retrieval/             # Vector search + re-ranking
├── agents/                # LangGraph orchestrator + 4 specialized agents
├── tools/                 # Web search, citation tracking
└── api/                   # FastAPI server
```

## Supported Formats

| Format | Loader |
|--------|--------|
| PDF | PyPDF |
| Markdown | TextLoader |
| Plain text | TextLoader |
| Web pages | WebBaseLoader |

## Configuration

All settings can be configured via `.env` or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required |
| `LLM_MODEL` | `gpt-4o` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Results per query |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `DOC_REGISTRY_PATH` | `./doc_registry.db` | Document registry SQLite path |
