"""State, nodes and edges: a support-message pipeline as a LangGraph graph.

Academy: LangGraph Essentials L1 "Nodes", L2 "Edges"; Intro to LangGraph M1 L1 "Motivation",
M1 L2 "Simple Graph", M1 L4 "Chain".

Concepts:
  * why a graph: `create_agent` (Parts 1–2) is one fixed loop; real workflows need steps, branches and
    loops you control. LangGraph is the layer `create_agent` itself is built on.
  * state: a `TypedDict` every node reads; each node returns only the keys it changes (a partial update)
  * nodes: plain Python functions (or runnables), here two deterministic steps and one LLM step
  * edges: `START → clean → tag → reply → END`; `compile()` checks the wiring and returns a runnable
    with `invoke` / `stream` / `batch`, like any LangChain runnable
  * `get_graph().draw_mermaid()` draws the graph (paste into https://mermaid.live)

Run:  uv run poe lesson 3.1
"""

import re
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header

MESSAGE = "  My notes   won't SYNC since the update.  Email me at ada@acme.example  "
TAGS = {"sync": ("sync",), "billing": ("invoice", "refund", "price"), "login": ("password", "sso", "log in")}


class TicketState(TypedDict, total=False):
    message: str  # input
    cleaned: str  # set by clean
    tags: list[str]  # set by tag
    reply: str  # set by reply


def build(llm: BaseChatModel):
    def clean(state: TicketState) -> dict:
        text = re.sub(r"\s+", " ", state["message"]).strip()
        text = re.sub(r"\S+@\S+", "[email]", text)  # don't send personal data to the model
        text = text.replace("SYNC", "sync")
        return {"cleaned": text}

    def tag(state: TicketState) -> dict:
        lowered = state["cleaned"].lower()
        return {"tags": [name for name, words in TAGS.items() if any(w in lowered for w in words)]}

    def reply(state: TicketState) -> dict:
        prompt = (
            "Write a one-sentence acknowledgement to this Nimbus Notes support message, mentioning the topic "
            f"({', '.join(state['tags']) or 'general'}).\n\nMessage: {state['cleaned']}"
        )
        return {"reply": llm.invoke(prompt).text}

    builder = StateGraph(TicketState)
    builder.add_node(clean)  # the node's name is the function's name
    builder.add_node(tag)
    builder.add_node(reply)
    builder.add_edge(START, "clean")
    builder.add_edge("clean", "tag")
    builder.add_edge("tag", "reply")
    builder.add_edge("reply", END)
    return builder.compile()


def run(llm: BaseChatModel) -> dict:
    graph = build(llm)
    header("The graph")
    print(graph.get_graph().draw_mermaid())
    header("Each node's partial update, in order")
    state: dict = {"message": MESSAGE}
    for update in graph.stream(state, stream_mode="updates"):
        print(f"  {update}")
        for delta in update.values():
            state.update(delta)  # no reducers here, so each key is simply overwritten (lesson 3.6)
    header("Final state (what graph.invoke would return)")
    for key, value in state.items():
        print(f"  {key}: {value!r}")
    return state


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
