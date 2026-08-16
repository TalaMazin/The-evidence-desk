"""
Module 2 Bridge Project - LangGraph

Build an agent that searches the web, decides whether what it found is good
enough, and then does ONE OF TWO different things.

    question
       |
    plan_query    (LLM)
       |
    run_search    (TOOL)
       |
    assess        (LLM)
       |
       +--- ENOUGH -------> write_answer
       |
       +--- NOT_ENOUGH ---> report_gap
"""

from __future__ import annotations

import os
import sys 
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search_tools import NO_RESULTS, SEARCH_UNAVAILABLE, web_search  # noqa: E402


# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------

load_dotenv()
load_dotenv(find_dotenv(usecwd=True))

api_key = os.environ.get("OPENROUTER_API_KEY")
base_url = os.environ.get(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)

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

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    base_url=base_url,
    api_key=api_key,
)


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

class DeskState(TypedDict):
    question: str
    query: str
    evidence: str
    verdict: str
    reasoning: str
    output: str
    route_taken: str


# ---------------------------------------------------------------------------
# NODE 1 - PLAN QUERY
# ---------------------------------------------------------------------------

def plan_query(state: DeskState) -> DeskState:
    """Turn the user's question into one useful search query."""

    prompt = f"""
Turn the following question into ONE short web search query.

Question:
{state["question"]}

Rules:
- Return ONLY the search query.
- Do not explain your answer.
- Do not add labels such as "Search query:".
"""

    response = llm.invoke(prompt)

    state["query"] = response.content.strip()

    return state


# ---------------------------------------------------------------------------
# NODE 2 - RUN SEARCH
# ---------------------------------------------------------------------------

def run_search(state: DeskState) -> DeskState:
    """Run the actual web search. No LLM is used here."""

    state["evidence"] = web_search(state["query"])

    return state


# ---------------------------------------------------------------------------
# NODE 3 - ASSESS
# ---------------------------------------------------------------------------

def assess(state: DeskState) -> DeskState:
    """Decide whether the evidence is sufficient to answer honestly."""

    # Search could not run.
    if state["evidence"] == SEARCH_UNAVAILABLE:
        state["verdict"] = "NOT_ENOUGH"
        state["reasoning"] = (
            "The search was unavailable, so there is no evidence to assess."
        )
        return state

    # Search ran but found nothing.
    if state["evidence"] == NO_RESULTS:
        state["verdict"] = "NOT_ENOUGH"
        state["reasoning"] = (
            "The search returned no results, so there is not enough evidence "
            "to answer the question."
        )
        return state

    prompt = f"""
You are an evidence evaluator.

Question:
{state["question"]}

Search evidence:
{state["evidence"]}

Determine whether the search evidence is good enough to answer the question
honestly.

Rules:
- The evidence must actually support the answer.
- Do not assume facts that are not present in the evidence.
- If the evidence is incomplete or ambiguous, choose NOT_ENOUGH.
- Give 1-3 short bullets explaining your decision.
- On the final line write exactly one of:

ENOUGH

or

NOT_ENOUGH
"""

    response = llm.invoke(prompt)

    raw = response.content.strip()

    state["reasoning"] = raw
    state["verdict"] = read_verdict(raw)

    return state


# ---------------------------------------------------------------------------
# READ VERDICT
# ---------------------------------------------------------------------------

def read_verdict(raw: str) -> str:
    """
    Safely extract ENOUGH or NOT_ENOUGH from model output.

    IMPORTANT:
    Check NOT_ENOUGH before ENOUGH because NOT_ENOUGH contains ENOUGH.
    """

    lines = [
        line.strip().upper()
        for line in raw.splitlines()
        if line.strip()
    ]

    if not lines:
        return "NOT_ENOUGH"

    # Look at the last non-empty line first.
    last_line = lines[-1]

    if "NOT_ENOUGH" in last_line:
        return "NOT_ENOUGH"

    if "ENOUGH" in last_line:
        return "ENOUGH"

    # Safe fallback.
    return "NOT_ENOUGH"


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------

def choose_next(
    state: DeskState,
) -> Literal["write_answer", "report_gap"]:
    """Choose the next node based on the verdict."""

    if state["verdict"] == "ENOUGH":
        return "write_answer"

    return "report_gap"


# ---------------------------------------------------------------------------
# NODE 4A - WRITE ANSWER
# ---------------------------------------------------------------------------

def write_answer(state: DeskState) -> DeskState:
    """Answer the question using only the available evidence."""

    prompt = f"""
Answer the following question using ONLY the supplied search evidence.

Question:
{state["question"]}

Evidence:
{state["evidence"]}

Rules:
- Every factual claim must be supported by the evidence.
- Do not invent facts.
- Do not use outside knowledge.
- Include at least one source URL from the evidence.
- If some part of the question is not covered by the evidence, say so.
- Keep the response under 180 words.
"""

    response = llm.invoke(prompt)

    state["output"] = response.content.strip()
    state["route_taken"] = "answered"

    return state


# ---------------------------------------------------------------------------
# NODE 4B - REPORT GAP
# ---------------------------------------------------------------------------

def report_gap(state: DeskState) -> DeskState:
    """Report insufficient evidence without answering the question."""
    
    prompt = f"""
The question below could not be verified from the available evidence.

Question:
{state["question"]}

Evidence:
{state["evidence"]}

Assessment:
{state["reasoning"]}

DO NOT answer the question.

Produce exactly these components:

1. One sentence plainly saying that the question could not be verified.
2. One or two short bullets explaining what evidence is missing.
3. A final line in exactly this format:

NEXT SEARCH: <one useful search query>

Do not invent facts.
Do not give a likely answer.
Do not provide a date, name, number, or other fact that is not supported.
"""

    response = llm.invoke(prompt)

    state["output"] = response.content.strip()
    state["route_taken"] = "gap_reported"

    return state


# ---------------------------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(DeskState)

    # Register nodes.
    graph.add_node("plan_query", plan_query)
    graph.add_node("run_search", run_search)
    graph.add_node("assess", assess)
    graph.add_node("write_answer", write_answer)
    graph.add_node("report_gap", report_gap)

    # Entry point.
    graph.set_entry_point("plan_query")

    # Fixed part of the route.
    graph.add_edge("plan_query", "run_search")
    graph.add_edge("run_search", "assess")

    # Conditional branch.
    graph.add_conditional_edges(
        "assess",
        choose_next,
        {
            "write_answer": "write_answer",
            "report_gap": "report_gap",
        },
    )

    # Both possible endings.
    graph.add_edge("write_answer", END)
    graph.add_edge("report_gap", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

def run(question: str) -> DeskState:
    return build_graph().invoke(
        {
            "question": question,
            "query": "",
            "evidence": "",
            "verdict": "",
            "reasoning": "",
            "output": "",
            "route_taken": "",
        }
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    question = (
        " ".join(sys.argv[1:]).strip()
        or "Microsoft was founded by Bill Gates and Paul Allen."
    )

    result = run(question)

    print("=" * 70)
    print(f"QUESTION : {result['question']}")
    print("=" * 70)

    print(f"\n[1] SEARCH QUERY")
    print(result["query"])

    print(f"\n[2] EVIDENCE ({len(result['evidence'])} chars)")
    print(result["evidence"][:700])

    print("\n[3] ASSESSMENT")
    print(result["reasoning"])

    print("\n" + "=" * 70)
    print(f"VERDICT : {result['verdict']}")
    print(f"ROUTE   : {result['route_taken'].upper()}")
    print("=" * 70)

    print(result["output"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())