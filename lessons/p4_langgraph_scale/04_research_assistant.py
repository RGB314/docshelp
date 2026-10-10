"""Research assistant: write a help-center article from several reader perspectives.

Academy: Intro to LangGraph M4 L4 "Research Assistant", applied to writing Nimbus Notes help articles.

Concepts (the graph is `src/docshelp/research.py::build_research_graph`):
  * human-in-the-loop planning: the model proposes reader perspectives ("analysts"); an editor approves
    them or gives feedback, which loops back to re-plan (`interrupt()`, lesson 3.13)
  * map: one `Send` per analyst starts an interview (lesson 4.3)
  * the interview is a subgraph (lesson 4.2): the analyst asks, the docs and changelog are searched in
    parallel (lesson 4.1), an "expert" answers from what was found, and the analyst writes a section
  * reduce: the introduction, body and conclusion are written in parallel, then joined by `finalize`
  * why several perspectives: each interview digs into what *its* reader needs, so the article covers
    end users and admins without one long, unfocused prompt

Run:  uv run poe lesson 4.4                (scripted editor: one round of feedback, then approve)
      uv run poe lesson 4.4 --interactive  (you are the editor)
"""

import argparse
import uuid
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.research import build_research_graph

TOPIC = "Working offline and syncing safely in Nimbus Notes"


def scripted_editor() -> Callable[[dict], str]:
    answers = iter(["Make one of them a workspace admin.", ""])  # feedback once, then approve
    return lambda payload: next(answers, "")


def terminal_editor(payload: dict) -> str:
    return input(f"  {payload['question']} > ")


def run(llm: BaseChatModel, retriever: BaseRetriever, editor: Callable[[dict], str] | None = None) -> dict:
    editor = editor or scripted_editor()
    graph = build_research_graph(llm, retriever, InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    header(f"Topic: {TOPIC}")
    result = graph.invoke({"topic": TOPIC, "max_analysts": 2}, config)
    rounds = 0
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        rounds += 1
        print("Proposed perspectives:")
        for persona in payload["analysts"]:
            print(f"  - {persona}")
        answer = editor(payload)
        print(f"Editor: {answer or '(approved)'}\n")
        result = graph.invoke(Command(resume=answer), config)
    header("The article")
    print(result["article"])
    return {**result, "feedback_rounds": rounds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="review the perspectives yourself")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever(), terminal_editor if args.interactive else None)


if __name__ == "__main__":
    main()
