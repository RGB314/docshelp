"""Deep research, step 1: scoping. Ask a clarifying question if needed, then write a research brief.

Academy: Project: Deep Research M2 "Scoping".

Concepts (`src/docshelp/deep_research.py::build_scoping_graph`):
  * research requests are often vague ("how do we compare?"). Researching the wrong question is the
    costliest mistake a deep-research agent makes, so it checks scope first.
  * `ClarifyWithUser` (structured output): either one clarifying question or a confirmation. A question
    *ends the run*; the user's answer arrives as the next message on the same thread (checkpointer), and
    scoping runs again with the whole conversation.
  * `ResearchBrief`: the conversation condensed into one self-contained brief. From here on the
    researchers see only the brief, never the chat, which keeps their context small and focused.

Run:  uv run poe lesson 6.5
"""

import uuid
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep_research import build_scoping_graph

REQUEST = "Research how Nimbus Notes compares on offline use."


def scripted_user(question: str) -> str:
    return "For our sales team: compare offline support across desktop, mobile and web, as a one-page summary."


def run(llm: BaseChatModel, answer: Callable[[str], str] = scripted_user) -> dict:
    graph = build_scoping_graph(llm, InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    header(f"User: {REQUEST}")
    state = graph.invoke({"messages": [{"role": "user", "content": REQUEST}]}, config)
    questions = []
    while not state.get("research_brief"):  # the scoper asked something: answer it on the same thread
        questions.append(state["messages"][-1].text)
        reply = answer(questions[-1])
        print(f"Scoper: {questions[-1]}\nUser:   {reply}")
        state = graph.invoke({"messages": [{"role": "user", "content": reply}]}, config)
        if len(questions) >= 3:
            break
    print(f"Scoper: {state['messages'][-1].text}")
    header("Research brief")
    print(state.get("research_brief", "(not written)"))
    return {"questions": questions, "brief": state.get("research_brief", "")}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
