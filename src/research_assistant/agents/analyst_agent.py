"""Analyst Agent — compares findings, identifies patterns and contradictions.

Performs structured analysis over retrieved sources.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

from research_assistant.config import get_settings
from research_assistant.llm_factory import get_llm, extract_reasoning, strip_thinking_tags

logger = logging.getLogger(__name__)

_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a research analyst. Analyze the provided sources and produce "
        "a structured analysis that includes:\n\n"
        "1. **Key Findings**: Main results and conclusions from the sources\n"
        "2. **Agreements**: Where sources agree or corroborate each other\n"
        "3. **Contradictions**: Where sources disagree or present conflicting data\n"
        "4. **Gaps**: What questions remain unanswered\n"
        "5. **Methodology Assessment**: Strengths and weaknesses of approaches used\n\n"
        "Important Guidelines:\n"
        "- Cite sources using their provided [Source: ...] notation.\n"
        "- Pay attention to source metadata (e.g., Publication Year, Citation Count) if available.\n"
        "- Give more weight to highly-cited or more recent research when evaluating contradictions.\n"
        "- Note the venue or quality of each source if relevant."
    )),
    ("human", "Query: {query}\n\nSources to analyze:\n\n{context}"),
])

_COMPARISON_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a research analyst. Compare the following sources and create "
        "a structured comparison table and narrative. Focus on:\n"
        "- Methodological differences\n"
        "- Results and outcome differences\n"
        "- Relative strengths and limitations\n\n"
        "Important Guidelines:\n"
        "- Cite sources using their provided [Source: ...] notation.\n"
        "- Explicitly compare the recency (Year) and authority (Citation Count) of the sources when available in the context metadata.\n"
        "- Use markdown tables where appropriate to summarize the comparison."
    )),
    ("human", "Query: {query}\n\nSources to compare:\n\n{context}"),
])


async def run_analyst_agent(
    query: str,
    context: str,
    mode: str = "analyze",
) -> dict:
    """Execute the analyst agent.

    Args:
        query: The research question driving the analysis.
        context: Formatted source context.
        mode: 'analyze' for general analysis, 'compare' for comparison.

    Returns:
        Dict with 'analysis' string.
    """
    llm = get_llm(temperature=0.2)

    if mode == "compare":
        chain = _COMPARISON_PROMPT | llm
    else:
        chain = _ANALYSIS_PROMPT | llm

    response = await chain.ainvoke({
        "query": query,
        "context": context,
    })

    reasoning = extract_reasoning(response)
    analysis = strip_thinking_tags(response.content) if isinstance(response.content, str) else response.content
    logger.info("[Analyst] Generated %s (%d chars, reasoning=%s)", mode, len(analysis), bool(reasoning))

    return {"analysis": analysis, "reasoning": reasoning}
