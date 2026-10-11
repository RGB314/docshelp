"""Deep research, step 4: a supervisor that plans, delegates to parallel researchers, and decides when to stop.

Academy: Project: Deep Research M5 "Research Supervisor".

Concepts (`src/docshelp/deep_research.py::build_supervisor`):
  * the supervisor's tools are decisions, not actions: `ConductResearch(topic)` delegates one sub-topic,
    `ResearchComplete` ends research, `think` plans. They're Pydantic models bound as tools: the model's
    tool calls *are* the plan.
  * parallel researchers: every `ConductResearch` in one turn starts a researcher subgraph (lesson 6.6),
    and `asyncio.gather` runs them together. Each sees only its own topic, so contexts stay small.
  * limits keep cost bounded: at most `max_concurrent` researchers per turn (extras are told to wait) and
    at most `max_iterations` supervisor turns
  * the supervisor reads only the *compressed* findings; the raw notes are kept for the report writer
  * a cheaper model can run the researchers while a stronger one supervises (`researcher_llm`)

Run:  uv run poe lesson 6.8           (offline sources)
      uv run poe lesson 6.8 --web     (plus web search)
"""

import argparse
import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep_research import build_researcher, build_supervisor, make_research_tools, supervisor_input
from docshelp.ingest import get_retriever

BRIEF = (
    "For the Nimbus Notes sales team: summarise how Nimbus Notes supports teams that work offline and on "
    "several devices, covering sync behaviour, offline limits per platform, and recent fixes from the changelog."
)


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    *,
    researcher_llm: BaseChatModel | None = None,
    web: bool = False,
    max_concurrent: int = 3,
) -> dict:
    researcher = build_researcher(researcher_llm or llm, make_research_tools(retriever, web=web))
    supervisor = build_supervisor(llm, researcher, max_concurrent=max_concurrent)
    header("Brief")
    print(BRIEF)
    result = asyncio.run(supervisor.ainvoke(supervisor_input(BRIEF, max_concurrent=max_concurrent)))

    header("What the supervisor decided, turn by turn")
    turns = []
    for message in result["supervisor_messages"]:
        if isinstance(message, AIMessage):
            decisions = [
                c["args"].get("research_topic") or c["args"].get("reflection") or c["name"] for c in message.tool_calls
            ]
            turns.append([c["name"] for c in message.tool_calls])
            summary = " | ".join(f"{c['name']}: {d}"[:90] for c, d in zip(message.tool_calls, decisions))
            print(f"  turn {len(turns)}: {summary}")
    header(f"{len(result.get('notes', []))} sets of findings")
    for note in result.get("notes", []):
        print(f"  - {note[:160]}")
    return {"turns": turns, "notes": result.get("notes", []), "messages": result["supervisor_messages"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", action="store_true", help="also search the web (needs `uv sync --extra web`)")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever(), researcher_llm=get_llm(fast=True), web=args.web)


if __name__ == "__main__":
    main()
