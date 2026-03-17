"""Summarizer Agent — produces concise summaries of documents or contexts.

Supports single-document and multi-document summarization at
configurable detail levels.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

from research_assistant.config import get_settings
from research_assistant.llm_factory import get_llm, extract_reasoning, strip_thinking_tags

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a research summarizer. Produce a clear, accurate summary "
        "of the provided content. Preserve key findings, methodologies, "
        "and conclusions. Use bullet points for clarity when appropriate.\n\n"
        "Detail level: {detail_level}\n"
        "- 'brief': 2-3 sentence abstract\n"
        "- 'standard': paragraph-level summary with key points\n"
        "- 'detailed': comprehensive summary preserving nuance and data"
    )),
    ("human", "Please summarize the following content:\n\n{content}"),
])

_MULTI_DOC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a research synthesizer. Summarize and cross-reference "
        "the following sources into a unified summary. Highlight:\n"
        "- Common themes and findings\n"
        "- Contradictions or disagreements\n"
        "- Unique contributions from each source\n"
        "Cite sources using [Source N] notation."
    )),
    ("human", "Sources to synthesize:\n\n{content}"),
])


async def run_summarizer_agent(
    content: str,
    detail_level: str = "standard",
    multi_doc: bool = False,
) -> dict:
    """Execute the summarizer agent.

    Args:
        content: Text content to summarize.
        detail_level: One of 'brief', 'standard', 'detailed'.
        multi_doc: If True, use multi-document synthesis prompt.

    Returns:
        Dict with 'summary' string.
    """
    llm = get_llm(temperature=0.3)

    if multi_doc:
        chain = _MULTI_DOC_PROMPT | llm
        response = await chain.ainvoke({"content": content})
    else:
        chain = _SUMMARIZE_PROMPT | llm
        response = await chain.ainvoke({
            "content": content,
            "detail_level": detail_level,
        })

    reasoning = extract_reasoning(response)
    summary = strip_thinking_tags(response.content) if isinstance(response.content, str) else response.content
    logger.info("[Summarizer] Generated %s summary (%d chars, reasoning=%s)",
                detail_level, len(summary), bool(reasoning))

    return {"summary": summary, "reasoning": reasoning}
