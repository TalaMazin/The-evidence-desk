# Module 2 Bridge Project — The Evidence Desk

## Overview

This project builds an evidence-based web research agent using **two frameworks**:

* **LangGraph**
* **CrewAI**

Both versions follow the same basic idea:

```text
Question
   ↓
Search the web
   ↓
Evaluate the evidence
   ↓
 ┌───────────────┐
 │ Enough?       │
 └───────┬───────┘
     YES │ NO
         │
         ├──────────────→ Answer
         │
         └──────────────→ Report Gap
```

The important part of the project is that the agent should **not invent an answer when there is not enough evidence**.

The assignment specifically uses a real claim and the intentionally invalid query:

```text
the flurbotron 9000 was released in 2019
```

The second query tests whether the systems can correctly refuse to make up information.

---

# Setup

Both implementations use OpenRouter and the `gpt-4o-mini` model.

Create a `.env` file in the project folder:

```text
OPENROUTER_API_KEY=sk-or-...
```

The LangGraph implementation also supports:

```text
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

If the base URL is not provided, this value is used automatically.

The search functionality comes from the provided `search_tools.py` file.

---

# LangGraph Implementation

The LangGraph version represents the workflow as an explicit graph.

```text
                  question
                     ↓
                plan_query
                     ↓
                run_search
                     ↓
                  assess
                 /      \
            ENOUGH      NOT_ENOUGH
              ↓             ↓
        write_answer    report_gap
              ↓             ↓
             END           END
```

## Nodes

### `plan_query`

Uses the LLM to turn the user's question into one short web search query.

### `run_search`

Calls the actual `web_search()` function.

This node does **not** use an LLM.

### `assess`

Evaluates whether the search evidence is sufficient.

It also handles two special search results directly:

* `NO_RESULTS`
* `SEARCH_UNAVAILABLE`

Both result in a `NOT_ENOUGH` verdict.

### `write_answer`

Runs when the evidence is sufficient.

The response must:

* Use only the supplied evidence.
* Include a source URL.
* Avoid unsupported facts.
* Stay under 180 words.

### `report_gap`

Runs when the evidence is insufficient.

It must not answer the question. Instead, it explains the evidence gap and produces:

```text
NEXT SEARCH: <query>
```

---

# LangGraph Routing

The most important part of the LangGraph implementation is the conditional edge:

```python
graph.add_conditional_edges(
    "assess",
    choose_next,
    {
        "write_answer": "write_answer",
        "report_gap": "report_gap",
    },
)
```

The `choose_next()` function decides which node runs next.

This means the decision is an **explicit part of the program's control flow**.

The final state also records:

```text
route_taken = answered
```

or:

```text
route_taken = gap_reported
```

This makes the route directly observable.

---

# CrewAI Implementation

The CrewAI version implements the same overall idea using a **hierarchical crew**.

Instead of explicitly defining a conditional graph, it uses a manager to coordinate two agents:

```text
                 MANAGER
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
     RESEARCHER             WRITER
          ↓
      Web Search
```

## Researcher

The Researcher is responsible for finding evidence.

It has access to the:

```text
Web Search
```

tool.

Its instructions tell it to:

* Search for relevant evidence.
* Preserve source URLs.
* Report exactly what the search returns.
* Handle `NO_RESULTS`.
* Handle `SEARCH_UNAVAILABLE`.
* Never invent facts.

## Writer

The Writer produces the final response.

It does not have the search tool.

Instead, it receives the Researcher's evidence and is instructed to use only that evidence.

It has two possible outputs:

```text
VERDICT: ANSWERED
```

or:

```text
VERDICT: COULD NOT VERIFY
```

If the evidence is insufficient, it must end with:

```text
NEXT SEARCH: <query>
```

---

# CrewAI Hierarchical Process

The crew is created with:

```python
process=Process.hierarchical
manager_llm=llm
```

The manager decides how the Researcher and Writer should work.

Unlike LangGraph, there is no explicit `route_taken` state.

Instead, the program looks at the final text and detects:

```text
VERDICT: ANSWERED
```

or:

```text
VERDICT: COULD NOT VERIFY
```

It then prints:

```text
ROUTE : ANSWERED
```

or:

```text
ROUTE : GAP_REPORTED
```

---

# Testing

Both implementations should be tested with two queries.

## Test 1 — Real Claim

```text
Anthropic was founded by former OpenAI employees
```

Expected behaviour:

```text
ANSWERED
```

The response should contain supporting evidence and a source URL.

## Test 2 — Nonsense Claim

```text
the flurbotron 9000 was released in 2019
```

Expected behaviour:

```text
NOT_ENOUGH / COULD NOT VERIFY
```

The system should not invent a release date.

The assignment specifically requires saving the outputs from both frameworks for both queries.

---

# LangGraph vs CrewAI

|                     | LangGraph                 | CrewAI                 |
| ------------------- | ------------------------- | ---------------------- |
| Architecture        | Explicit graph            | Hierarchical crew      |
| Research            | `run_search` node         | Researcher agent       |
| Evidence assessment | `assess` node             | Manager/agents         |
| Decision            | Explicit conditional edge | Manager decision       |
| Answer              | `write_answer`            | Writer                 |
| Refusal             | `report_gap` code path    | Agent instruction      |
| Route tracking      | Explicit state            | Inferred from output   |
| Tools               | Web search function       | Web Search CrewAI tool |

The main architectural difference is **where the decision lives**.

In LangGraph, the decision is represented directly in the graph. In CrewAI, the manager is responsible for deciding how the agents should proceed. The assignment highlights this difference as **“refusing is a CODE PATH”** in LangGraph versus **“refusing is an INSTRUCTION”** in CrewAI.

---

# Assignment Questions

## 1. Which framework would you ship for your use case, and why?

I would ship **LangGraph** for this use case.

The main reason is that the decision to answer or refuse is explicitly represented in the graph. This makes the behaviour easier to inspect, test, and explain.

CrewAI is useful for coordinating multiple agents, but in this implementation the refusal behaviour depends more heavily on the manager and the instructions given to the agents.

For an evidence-based system where avoiding unsupported answers is important, I prefer the more explicit control provided by LangGraph.

---

## 2. On the nonsense query — did each version refuse?

The test query was:

```text
the flurbotron 9000 was released in 2019
```

### LangGraph

The LangGraph implementation is designed to refuse when the search produces no useful evidence.

It detects `NO_RESULTS`, sets the verdict to:

```text
NOT_ENOUGH
```

and routes the graph to `report_gap`.

### CrewAI

The CrewAI implementation is instructed to refuse when the research is insufficient.

The actual result should be recorded from the experiment here:

```text
[PASTE ACTUAL CREWAI OUTPUT HERE]
```

If CrewAI produces an invented answer, that is an important result because its refusal behaviour is instruction-based rather than an explicit conditional code path.

---

## 3. Where does the decision live in each one?

### LangGraph

The decision lives in:

```text
assess → choose_next → conditional edge
```

The possible routes are:

```text
ENOUGH
   ↓
write_answer
```

and:

```text
NOT_ENOUGH
   ↓
report_gap
```

This is an explicit program-level decision.

### CrewAI

The decision is handled by the hierarchical manager.

The manager coordinates the Researcher and Writer according to their tasks, goals, and instructions.

Therefore, if a customer asked how the system prevents unsupported answers, **LangGraph is easier to prove**, because the refusal route is visible directly in the graph.

---

## 4. Why couldn't the agent run the `NEXT SEARCH`?

The current implementations can generate a `NEXT SEARCH` query, but they do not have a loop that sends that query back to the search stage.

For example:

```text
report_gap
    ↓
NEXT SEARCH: some query
    ↓
       END
```

There is no connection like:

```text
report_gap
    ↓
plan_query / run_search
    ↓
assess
```

Therefore, the agent knows what it could search for next, but it has no graph path that actually performs that search.

The missing piece is a **loop back to the research stage**.

---

# Conclusion

This project compares two ways of building an evidence-based agent.

**LangGraph** makes the workflow and decision points explicit through nodes, state, and conditional edges.

**CrewAI** uses a hierarchical manager to coordinate specialized agents.

Both approaches can perform web research and produce evidence-based responses, but LangGraph provides more explicit control over the answer-versus-refusal decision in this particular implementation.

The main lesson from the project is that **refusing to answer when there is insufficient evidence is a valid and important behaviour**, rather than a failure.
