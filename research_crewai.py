"""
Module 2 Bridge Project - CrewAI

Hierarchical CrewAI implementation.

The manager decides how the researcher and writer should work.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Disable tracing prompt on first run.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import Agent, Crew, LLM, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402
from dotenv import find_dotenv, load_dotenv  # noqa: E402

try:
    from crewai.events.listeners.tracing.utils import mark_first_execution_done

    mark_first_execution_done()
except Exception:
    pass


# ---------------------------------------------------------------------------
# SEARCH TOOL
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search_tools import web_search  # noqa: E402


# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------

load_dotenv()
load_dotenv(find_dotenv(usecwd=True))

api_key = os.environ.get("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY not found.\n"
        "Create a file named .env in the project folder containing:\n"
        "    OPENROUTER_API_KEY=sk-or-...\n"
        "then run this script again."
    )


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    temperature=0,
    api_key=api_key,
)


# ---------------------------------------------------------------------------
# TOOL
# ---------------------------------------------------------------------------

@tool("Web Search")
def search_web(query: str) -> str:
    """Search the web for information about a query.

    Returns search findings with source URLs.

    The search can return the exact text NO_RESULTS when the search
    found nothing, or SEARCH_UNAVAILABLE when the search could not run.

    Use this tool when you need external evidence for a question.
    """
    return web_search(query)


# ---------------------------------------------------------------------------
# BUILD CREW
# ---------------------------------------------------------------------------

def build_crew() -> Crew:

    # -----------------------------------------------------------------------
    # RESEARCHER
    # -----------------------------------------------------------------------

    researcher = Agent(
        role="Web Researcher",
        goal=(
            "Find reliable evidence that can answer the user's question "
            "and preserve the source URLs."
        ),
        backstory=(
            "You are a careful and evidence-focused web researcher. "
            "Use the Web Search tool to find information relevant to the "
            "question. Report exactly what the search actually returns and "
            "preserve source URLs. If the search returns NO_RESULTS, report "
            "that plainly. If the search returns SEARCH_UNAVAILABLE, report "
            "that plainly. Never fill missing information from memory and "
            "never invent facts."
        ),
        tools=[search_web],
        llm=llm,
        verbose=True,
        max_iter=4,
    )

    # -----------------------------------------------------------------------
    # WRITER
    # -----------------------------------------------------------------------

    writer = Agent(
        role="Evidence-Based Writer",
        goal=(
            "Produce an accurate final response only when the research "
            "provides enough evidence."
        ),
        backstory=(
            "You are a cautious evidence-based writer. You trust the "
            "research evidence rather than your own memory. You would "
            "rather clearly say that something could not be verified than "
            "produce a polished answer that might be wrong. Use only the "
            "research provided to you. Never invent facts."
        ),
        llm=llm,
        verbose=True,
        max_iter=4,
    )

    # -----------------------------------------------------------------------
    # RESEARCH TASK
    # -----------------------------------------------------------------------

    research_task = Task(
        description="""
Research the following question:

{question}

Use the Web Search tool to find relevant evidence.

Requirements:
- Search the web for evidence related to the question.
- Report exactly what the search returned.
- Preserve the source URLs.
- Do not substitute your own knowledge if the search finds nothing.
- If the search returns NO_RESULTS, clearly report NO_RESULTS.
- If the search returns SEARCH_UNAVAILABLE, clearly report
  SEARCH_UNAVAILABLE.
- Do not invent facts.
""",
        expected_output="""
A research report containing:
- the relevant evidence found
- the source URLs
- a clear statement about whether evidence was found
- NO_RESULTS or SEARCH_UNAVAILABLE when applicable
""",
    )

    # -----------------------------------------------------------------------
    # WRITING TASK
    # -----------------------------------------------------------------------

    write_task = Task(
        description="""
Produce the final response for:

{question}

Use ONLY the evidence produced by the researcher.

There are two possible outcomes.

OUTCOME 1 — Evidence is sufficient:

- Begin with exactly:
  VERDICT: ANSWERED
- Answer the question in under 180 words.
- Include at least one source URL from the research.
- Every factual claim must be supported by the research.
- Do not invent facts.

OUTCOME 2 — Evidence is insufficient:

- Begin with exactly:
  VERDICT: COULD NOT VERIFY
- Do NOT answer the question.
- Write one sentence explaining that it could not be verified.
- Include one or two short points explaining what is missing.
- End with exactly:

NEXT SEARCH: <one useful search query>

Do not use outside knowledge to fill gaps.
Do not guess.
""",
        expected_output="""
A final response beginning with either:

VERDICT: ANSWERED

or:

VERDICT: COULD NOT VERIFY

If answered:
- under 180 words
- includes evidence
- includes a source URL

If not verified:
- does not answer the question
- explains the evidence gap
- ends with a NEXT SEARCH line
""",
    )

    # -----------------------------------------------------------------------
    # HIERARCHICAL CREW
    # -----------------------------------------------------------------------

    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True,
        tracing=False,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:

    question = (
        " ".join(sys.argv[1:]).strip()
        or "Anthropic was founded by former OpenAI employees"
    )

    result = str(
        build_crew().kickoff(
            inputs={"question": question}
        )
    )

    print("\n" + "=" * 70)
    print(f"QUESTION : {question}")
    print("=" * 70)

    print(result)

    # -----------------------------------------------------------------------
    # ROUTE DETECTION
    #
    # Unlike LangGraph, there is no explicit route_taken state.
    # We infer the route from the final text.
    # -----------------------------------------------------------------------

    result_upper = result.upper()

    if "VERDICT: ANSWERED" in result_upper:
        route = "ANSWERED"

    elif "VERDICT: COULD NOT VERIFY" in result_upper:
        route = "GAP_REPORTED"

    else:
        route = "UNKNOWN"

    print("\n" + "=" * 70)
    print(f"ROUTE : {route}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())