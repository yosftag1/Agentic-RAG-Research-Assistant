"""Orchestrator — LangGraph state graph that routes queries to specialized agents.

This is the central brain of the research assistant. It:
1. Classifies the user's intent and resolves pronoun/reference queries via history
2. Routes to the appropriate agent(s)
3. Merges results and formats the final response
"""

from __future__ import annotations

import json as _json
import logging
import operator
import re as _re
from typing import Annotated, Awaitable, Callable, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from research_assistant.llm_factory import get_llm, extract_reasoning, strip_thinking_tags
from research_assistant.config import get_settings
from research_assistant.agents.retriever_agent import run_retriever_agent
from research_assistant.agents.summarizer_agent import run_summarizer_agent
from research_assistant.agents.analyst_agent import run_analyst_agent
from research_assistant.agents.writer_agent import run_writer_agent
from research_assistant.tools.citation_tool import format_citations

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict], Awaitable[None]]



def _is_thinking_model() -> bool:
    """Return True if the configured LLM is a reasoning/thinking model."""
    model = (get_settings().llm_model or "").lower()
    return any(x in model for x in ("2.5", "o1-", "o3", "thinking", "r1"))


def _parse_classify_response(content: str, fallback_query: str) -> tuple[str, str]:
    """Robustly extract (intent, resolved_query) from the LLM classify response."""
    text = content.strip()
    fence = _re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, _re.DOTALL)
    if fence:
        text = fence.group(1)
    obj = _re.search(r'\{[^{}]*\}', text, _re.DOTALL)
    if obj:
        try:
            data = _json.loads(obj.group())
            intent = data.get("intent", "lookup").strip().lower()
            resolved = (data.get("resolved_query") or fallback_query).strip()
            return intent, resolved
        except ValueError:
            pass
    lower = text.lower()
    for candidate in ("websearch", "analyze", "summarize", "draft", "lookup"):
        if candidate in lower:
            return candidate, fallback_query
    return "lookup", fallback_query



class ResearchState(TypedDict):
    """State that flows through the LangGraph."""
    query: str
    resolved_query: str                                     # query after reference resolution
    intent: str
    context: str
    sources: list
    summary: str
    analysis: str
    draft: str
    final_answer: str
    web_search_results: list[dict]
    source_preference: str
    conversation_history: list[dict]                        # [{role, content}, ...]
    messages: Annotated[list, operator.add]
    steps: Annotated[list[dict], operator.add]              # pipeline steps (append-only)


def _serialize_sources(sources: list) -> list[dict]:
    """Convert source Documents into frontend-friendly, deduplicated source items."""
    seen: set[str] = set()
    items: list[dict] = []

    for doc in sources or []:
        meta = getattr(doc, "metadata", {}) or {}
        source = str(meta.get("source", "Unknown source"))
        if source in seen:
            continue
        seen.add(source)

        source_type = str(meta.get("source_type", "unknown"))
        source_kind = str(meta.get("source_kind", ""))
        title = str(meta.get("title", "") or source)
        url = str(meta.get("url", "") or "")
        if not url and source.startswith("http"):
            url = source

        if source_type == "file":
            origin = "Local file"
        elif source_kind == "scholar":
            origin = "Google Scholar paper"
        elif source_type == "web":
            origin = "Web page"
        else:
            origin = "Other"

        items.append({
            "label": title,
            "url": url,
            "origin": origin,
            "source": source,
            "source_type": source_type,
        })

    return items


def _preview_sources(sources: list, limit: int = 3) -> str:
    """Build a short source preview string for step diagnostics."""
    if not sources:
        return "None"
    names: list[str] = []
    for doc in sources[:limit]:
        meta = getattr(doc, "metadata", {}) or {}
        title = str(meta.get("title") or meta.get("source") or "Unknown source")
        names.append(title[:90])
    if len(sources) > limit:
        names.append(f"+{len(sources) - limit} more")
    return " | ".join(names)



_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a query classifier for a research assistant.\n\n"
        "Do TWO things and respond with JSON only — no markdown, no code fences:\n\n"
        "1. Classify the user's current query intent into exactly one of:\n"
        "   - lookup   : needs factual retrieval from the knowledge base\n"
        "   - summarize: wants a summary of documents or topics\n"
        "   - analyze  : wants comparison, critique, or deep analysis\n"
        "   - draft    : wants to write or generate research content\n"
        "   - websearch: wants to search online for new academic papers\n\n"
        "2. If the query contains pronouns or references to prior messages "
        "(e.g. 'it', 'those papers', 'that topic', 'the same subject', 'what I asked before', "
        "'show me the most recent ones'), expand it into a fully self-contained query "
        "using the conversation history. If there are no such references, keep it as-is.\n\n"
        'Respond with exactly: {{"intent": "...", "resolved_query": "..."}}'
    )),
    ("human", "Conversation history (oldest first):\n{history}\n\nCurrent query: {query}"),
])


async def classify_intent(state: ResearchState) -> dict:
    """Classify intent and resolve any contextual references using history."""
    llm = get_llm(temperature=0)

    history_msgs = state.get("conversation_history", [])
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content'][:400]}" for m in history_msgs[-8:]
    ) or "(none)"

    chain = _CLASSIFY_PROMPT | llm
    response = await chain.ainvoke({"query": state["query"], "history": history_text})

    intent, resolved_query = _parse_classify_response(response.content, state["query"])
    valid_intents = {"lookup", "summarize", "analyze", "draft", "websearch"}
    if intent not in valid_intents:
        intent = "lookup"

    resolution_note = ""
    if resolved_query.strip().lower() != state["query"].strip().lower():
        resolution_note = "\n\n*Query expanded from conversation context.*"

    reasoning = extract_reasoning(response)
    step = {
        "agent": "Classifier",
        "icon": "🎯",
        "status": "done",
        "detail": (
            f"**Intent detected:** `{intent}`\n\n"
            f"**Resolved query:** {resolved_query}\n\n"
            f"**Conversation turns inspected:** {len(history_msgs[-8:])}"
            + resolution_note
        ),
        "thinking": _is_thinking_model(),
        "reasoning": reasoning,
    }

    logger.info("[Orchestrator] Intent: %s | Resolved: %r", intent, resolved_query)
    return {"intent": intent, "resolved_query": resolved_query, "steps": [step]}



async def retrieve_node(state: ResearchState) -> dict:
    """Run the retriever agent to find relevant context."""
    effective_query = state.get("resolved_query") or state["query"]
    source_preference = state.get("source_preference", "auto")
    result = await run_retriever_agent(effective_query, source_preference=source_preference)

    sources = result["sources"]
    local_sources = result.get("local_sources", [])
    web_sources = result.get("web_sources", [])
    local_count = result.get("local_doc_count")
    if local_count is None:
        local_count = len([d for d in sources if d.metadata.get("source_type") != "web"])
    web_used = result.get("web_used")
    if web_used is None:
        web_used = any(d.metadata.get("source_type") == "web" for d in sources)
    retrieval_strategy = (result.get("retrieval_strategy") or "HYBRID").title()

    detail_parts = [
        f"**Effective query:** {effective_query}",
        f"**Retrieval strategy:** {retrieval_strategy}",
        f"**Retriever mode:** {get_settings().search_strategy.upper()}",
        f"**Source preference:** {result.get('source_preference', 'auto').title()}",
        f"**Local chunks found:** {local_count}",
        f"**Local source preview:** {local_sources or 'None'}",
    ]
    if web_used:
        detail_parts.append(f"**Online source preview:** {web_sources or 'Used but no titles extracted'}")
    detail_parts.append(f"**Combined context size:** {len(result.get('context', ''))} chars")

    step = {
        "agent": "Retriever",
        "icon": "🔍",
        "status": "done",
        "detail": "\n\n".join(detail_parts),
        "thinking": False,
    }
    return {"context": result["context"], "sources": sources, "steps": [step]}


async def summarize_node(state: ResearchState) -> dict:
    """Run the summarizer agent on the retrieved context."""
    result = await run_summarizer_agent(
        content=state["context"],
        detail_level="standard",
        multi_doc=len(state.get("sources", [])) > 1,
    )
    step = {
        "agent": "Summarizer",
        "icon": "📝",
        "status": "done",
        "detail": (
            f"**Input context size:** {len(state.get('context', ''))} chars\n\n"
            f"**Sources considered:** {len(state.get('sources', []))}\n\n"
            f"**Output length:** {len(result['summary'])} chars"
        ),
        "thinking": _is_thinking_model(),
        "reasoning": result.get("reasoning"),
    }
    return {"summary": result["summary"], "steps": [step]}


async def analyze_node(state: ResearchState) -> dict:
    """Run the analyst agent on the retrieved context."""
    effective_query = state.get("resolved_query") or state["query"]
    result = await run_analyst_agent(
        query=effective_query,
        context=state["context"],
        mode="analyze",
    )
    step = {
        "agent": "Analyst",
        "icon": "📊",
        "status": "done",
        "detail": (
            f"**Input context size:** {len(state.get('context', ''))} chars\n\n"
            f"**Sources compared:** {len(state.get('sources', []))}\n\n"
            f"**Output length:** {len(result['analysis'])} chars"
        ),
        "thinking": _is_thinking_model(),
        "reasoning": result.get("reasoning"),
    }
    return {"analysis": result["analysis"], "steps": [step]}


async def write_node(state: ResearchState) -> dict:
    """Run the writer agent to draft content."""
    effective_query = state.get("resolved_query") or state["query"]
    result = await run_writer_agent(
        query=effective_query,
        context=state["context"],
        task="notes",
    )
    step = {
        "agent": "Writer",
        "icon": "✍️",
        "status": "done",
        "detail": (
            f"**Task:** notes\n\n"
            f"**Input context size:** {len(state.get('context', ''))} chars\n\n"
            f"**Draft length:** {len(result['content'])} chars"
        ),
        "thinking": _is_thinking_model(),
        "reasoning": result.get("reasoning"),
    }
    return {"draft": result["content"], "steps": [step]}


async def websearch_node(state: ResearchState) -> dict:
    """Run the web search explicitly triggered by user, using resolved query."""
    import asyncio
    from research_assistant.tools.scholar_tool import search_scholar_multi

    effective_query = state.get("resolved_query") or state["query"]
    results = await asyncio.to_thread(search_scholar_multi, effective_query, 5)
    if isinstance(results, str):
        results = []

    step = {
        "agent": "Web Search",
        "icon": "🌐",
        "status": "done",
        "detail": (
            f"**Query used:** {effective_query}\n\n"
            f"**Papers found:** {len(results)}"
        ),
        "thinking": False,
    }
    return {
        "web_search_results": results,
        "final_answer": (
            "I searched Google Scholar for your query. "
            "Here are the papers I found — select any to ingest into your Knowledge Base."
        ),
        "steps": [step],
    }



_MERGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a research assistant. Compose a final, well-structured "
        "response to the user's query using the provided agent outputs. "
        "Be thorough, clear, and include source citations using [Source: ...] notation.\n\n"
        "Guidelines:\n"
        "- Prefer more recent papers; note chronological differences when sources conflict.\n"
        "- Weight highly-cited papers more heavily for established findings.\n"
        "- Note the venue or quality of each source when mentioning it.\n"
        "- Always cite factual claims to the supplied sources when possible.\n"
        "- For web pages, use the page title and include the URL in the answer.\n"
        "- For papers, include Authors, Year, Venue, and Citation count when available.\n"
        "- If there is prior conversation context, make the answer coherent with it.\n\n"
        "Format the response in Markdown."
    )),
    ("human", (
        "Prior conversation context:\n{history}\n\n"
        "User query: {query}\n"
        "Resolved query: {resolved_query}\n\n"
        "Intent: {intent}\n\n"
        "Retrieved Context:\n{context}\n\n"
        "Summary:\n{summary}\n\n"
        "Analysis:\n{analysis}\n\n"
        "Draft:\n{draft}\n\n"
        "Sources:\n{citations}"
    )),
])


async def merge_results(state: ResearchState) -> dict:
    """Merge all agent outputs into a final answer."""
    llm = get_llm(temperature=0.3)
    citations = format_citations(state.get("sources", []))

    history_msgs = state.get("conversation_history", [])
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content'][:300]}" for m in history_msgs[-6:]
    ) or "(none)"

    chain = _MERGE_PROMPT | llm
    response = await chain.ainvoke({
        "query": state["query"],
        "resolved_query": state.get("resolved_query") or state["query"],
        "intent": state["intent"],
        "history": history_text,
        "context": state.get("context", "None")[:3000],
        "summary": state.get("summary", "None"),
        "analysis": state.get("analysis", "None"),
        "draft": state.get("draft", "None"),
        "citations": citations,
    })

    reasoning = extract_reasoning(response)
    raw_content = response.content if isinstance(response.content, str) else str(response.content)
    final_answer = strip_thinking_tags(raw_content).strip()
    if citations and citations != "No sources available.":
        final_answer = f"{final_answer}\n\n## Sources\n{citations}"

    step = {
        "agent": "Merge & Format",
        "icon": "🔀",
        "status": "done",
        "detail": (
            f"**Context window passed to merge:** {len(state.get('context', 'None')[:3000])} chars\n\n"
            f"**Sources serialized:** {len(state.get('sources', []))}\n\n"
            f"**Final answer length:** {len(final_answer)} chars"
        ),
        "thinking": _is_thinking_model(),
        "reasoning": reasoning,
    }
    logger.info("[Orchestrator] Final answer generated (%d chars)", len(final_answer))
    return {"final_answer": final_answer, "steps": [step]}



def route_by_intent(state: ResearchState) -> str:
    return {
        "lookup": "retrieve", "summarize": "retrieve",
        "analyze": "retrieve", "draft": "retrieve",
        "websearch": "websearch",
    }.get(state.get("intent", "lookup"), "retrieve")


def route_after_retrieval(state: ResearchState) -> str:
    return {
        "lookup": "merge", "summarize": "summarize",
        "analyze": "analyze", "draft": "write",
    }.get(state.get("intent", "lookup"), "merge")


async def _emit_progress(progress_callback: ProgressCallback | None, event: dict) -> None:
    """Emit a progress event when a streaming callback is provided."""
    if progress_callback is not None:
        await progress_callback(event)


def _merge_state_update(state: ResearchState, update: dict) -> None:
    """Apply a node update to the mutable state used by the streaming runner."""
    for key, value in update.items():
        if key == "steps":
            state["steps"].extend(value)
        else:
            state[key] = value


def _running_message(agent: str, state: ResearchState) -> str:
    """Map an agent name to a user-facing live status line."""
    query = (state.get("resolved_query") or state.get("query") or "").strip()
    short_query = query[:120] + ("..." if len(query) > 120 else "")
    return {
        "Classifier": f"Classifying intent from query: {short_query}",
        "Retriever": (
            f"Running {get_settings().search_strategy.upper()} retrieval for top-{get_settings().retrieval_top_k} chunks "
            f"(preference: {state.get('source_preference', 'auto')})"
        ),
        "Summarizer": f"Summarizing {len(state.get('sources', []))} retrieved sources",
        "Analyst": f"Analyzing agreement/contradictions across {len(state.get('sources', []))} sources",
        "Writer": "Drafting notes from retrieved context",
        "Web Search": "Querying Google Scholar variants and ranking papers",
        "Merge & Format": (
            f"Merging summary/analysis/draft with {len(state.get('sources', []))} citations"
        ),
    }.get(agent, "Running agent")


async def run_research_assistant_stream(
    query: str,
    history: list[dict] | None = None,
    source_preference: str = "auto",
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Run a query step-by-step while emitting live progress updates."""
    state: ResearchState = {
        "query": query,
        "resolved_query": query,
        "intent": "",
        "context": "",
        "sources": [],
        "summary": "",
        "analysis": "",
        "draft": "",
        "final_answer": "",
        "web_search_results": [],
        "source_preference": source_preference,
        "conversation_history": history or [],
        "messages": [],
        "steps": [],
    }

    await _emit_progress(progress_callback, {
        "type": "status",
        "agent": "System",
        "state": "running",
        "message": "Preparing the agent workflow",
    })

    async def run_stage(agent_label: str, node_fn) -> dict:
        await _emit_progress(progress_callback, {
            "type": "status",
            "agent": agent_label,
            "state": "running",
            "message": _running_message(agent_label, state),
        })
        update = await node_fn(state)
        _merge_state_update(state, update)
        step = (update.get("steps") or [None])[-1]
        if step is not None:
            await _emit_progress(progress_callback, {
                "type": "step",
                "agent": agent_label,
                "state": "done",
                "message": step.get("detail", "Completed"),
                "step": step,
            })
        return update

    await run_stage("Classifier", classify_intent)
    next_stage = route_by_intent(state)

    if next_stage == "websearch":
        await run_stage("Web Search", websearch_node)
    else:
        await run_stage("Retriever", retrieve_node)
        after_retrieval = route_after_retrieval(state)
        if after_retrieval == "summarize":
            await run_stage("Summarizer", summarize_node)
        elif after_retrieval == "analyze":
            await run_stage("Analyst", analyze_node)
        elif after_retrieval == "write":
            await run_stage("Writer", write_node)
        await run_stage("Merge & Format", merge_results)

    return {
        "final_answer": state.get("final_answer", ""),
        "sources": _serialize_sources(state.get("sources", [])),
        "web_search_results": state.get("web_search_results", []),
        "steps": state.get("steps", []),
    }



def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)
    graph.add_node("classify", classify_intent)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("write", write_node)
    graph.add_node("websearch", websearch_node)
    graph.add_node("merge", merge_results)
    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_by_intent, {
        "retrieve": "retrieve", "websearch": "websearch",
    })
    graph.add_conditional_edges("retrieve", route_after_retrieval, {
        "summarize": "summarize", "analyze": "analyze",
        "write": "write", "merge": "merge",
    })
    graph.add_edge("summarize", "merge")
    graph.add_edge("analyze", "merge")
    graph.add_edge("write", "merge")
    graph.add_edge("merge", END)
    graph.add_edge("websearch", END)
    return graph.compile()



async def run_research_assistant(
    query: str,
    history: list[dict] | None = None,
    source_preference: str = "auto",
) -> dict:
    """Run a query through the full agent pipeline.

    Args:
        query:   The user's research question.
        history: Optional prior turns [{role, content}, ...].

    Returns:
        Dict with final_answer, web_search_results, and steps.
    """
    graph = build_research_graph()
    initial_state: ResearchState = {
        "query": query,
        "resolved_query": query,
        "intent": "",
        "context": "",
        "sources": [],
        "summary": "",
        "analysis": "",
        "draft": "",
        "final_answer": "",
        "web_search_results": [],
        "source_preference": source_preference,
        "conversation_history": history or [],
        "messages": [],
        "steps": [],
    }
    result = await graph.ainvoke(initial_state)
    return {
        "final_answer": result.get("final_answer", ""),
        "sources": _serialize_sources(result.get("sources", [])),
        "web_search_results": result.get("web_search_results", []),
        "steps": result.get("steps", []),
    }
