"""Writer Agent — drafts structured research outputs.

Produces research notes, literature reviews, and reports in
Markdown or LaTeX format.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

from research_assistant.config import get_settings
from research_assistant.llm_factory import get_llm, extract_reasoning, strip_thinking_tags

logger = logging.getLogger(__name__)

_WRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an academic research writer. Draft well-structured research "
        "content based on the provided context and instructions.\n\n"
        "Guidelines:\n"
        "- Use clear, academic language\n"
        "- Organize with proper headings and sections\n"
        "- Include citations using [Source N] notation\n"
        "- Be thorough but concise\n"
        "- Output format: {output_format}"
    )),
    ("human", (
        "Task: {task}\n\n"
        "Query/Topic: {query}\n\n"
        "Source Context:\n{context}\n\n"
        "Additional instructions: {instructions}"
    )),
])

_WRITING_TASKS = {
    "notes": "Write structured research notes summarizing the sources.",
    "review": "Write a literature review synthesizing the sources.",
    "report": "Write a research report with introduction, findings, and conclusions.",
    "draft": "Draft content based on the provided instructions.",
}


async def run_writer_agent(
    query: str,
    context: str,
    task: str = "notes",
    instructions: str = "",
    output_format: str = "markdown",
) -> dict:
    """Execute the writer agent.

    Args:
        query: The research topic or question.
        context: Source context with citations.
        task: Writing task type — 'notes', 'review', 'report', 'draft'.
        instructions: Additional user instructions for the writer.
        output_format: 'markdown' or 'latex'.

    Returns:
        Dict with 'content' string.
    """
    llm = get_llm(temperature=0.4)

    task_description = _WRITING_TASKS.get(task, _WRITING_TASKS["draft"])

    chain = _WRITE_PROMPT | llm
    response = await chain.ainvoke({
        "task": task_description,
        "query": query,
        "context": context,
        "instructions": instructions or "None",
        "output_format": output_format,
    })

    reasoning = extract_reasoning(response)
    content = strip_thinking_tags(response.content) if isinstance(response.content, str) else response.content
    logger.info("[Writer] Generated %s (%d chars, format=%s, reasoning=%s)",
                task, len(content), output_format, bool(reasoning))

    return {"content": content, "reasoning": reasoning}
