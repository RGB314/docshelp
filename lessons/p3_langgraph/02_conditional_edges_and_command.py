"""Conditional edges and `Command`: route each support message to the right team.

Academy: LangGraph Essentials L3 "Conditional Edges"; Intro to LangGraph M1 L5 "Router".

Concepts:
  * an LLM router: a node classifies the message with structured output (lesson 1.3)
  * `add_conditional_edges(source, route_fn)`: after `source`, call `route_fn(state)` and go to the node
    it returns. A `Literal[...]` return type tells LangGraph (and the diagram) the possible targets.
  * `Command(goto=..., update=...)`: the node itself updates state *and* picks the next node, so no
    edges are needed after it. Type the return as `Command[Literal[...]]` for the diagram.
  * which to use: conditional edges keep routing logic separate and visible; `Command` suits nodes
    whose decision and state update belong together (handoffs between agents, Part 4 and Part 6)
  * the same router, built both ways, sends the same messages to the same teams

Run:  uv run poe lesson 3.2
"""

from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from docshelp.config import describe, ensure_llm_ready, get_llm, structured_llm
from docshelp.console import header

Team = Literal["billing", "howto", "escalate"]
MESSAGES = [
    "Why was I charged twice this month?",
    "How do I share a notebook with a colleague?",
    "This is the third time I've lost notes. I want to talk to a manager NOW.",
]


class Route(BaseModel):
    intent: Team = Field(
        description="billing: charges, plans, invoices. howto: how to use a feature. "
        "escalate: angry customers, data loss, or asking for a human."
    )


class State(TypedDict, total=False):
    message: str
    intent: Team
    reply: str


def team_nodes() -> dict:
    return {
        "billing": lambda s: {"reply": "Routed to Billing: they'll check your invoices."},
        "howto": lambda s: {"reply": "Routed to the docs assistant."},
        "escalate": lambda s: {"reply": "Routed to a human support lead, with high priority."},
    }


def with_conditional_edges(llm: BaseChatModel):
    router = structured_llm(llm, Route)

    def classify(state: State) -> dict:
        return {"intent": router.invoke(state["message"]).intent}

    def route(state: State) -> Team:  # the routing function: pure, easy to test
        return state["intent"]

    builder = StateGraph(State)
    builder.add_node(classify)
    for name, node in team_nodes().items():
        builder.add_node(name, node)
        builder.add_edge(name, END)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route)
    return builder.compile()


def with_command(llm: BaseChatModel):
    router = structured_llm(llm, Route)

    def classify(state: State) -> Command[Team]:
        intent = router.invoke(state["message"]).intent
        return Command(update={"intent": intent}, goto=intent)  # update state and route in one step

    builder = StateGraph(State)
    builder.add_node(classify)
    for name, node in team_nodes().items():
        builder.add_node(name, node)
        builder.add_edge(name, END)
    builder.add_edge(START, "classify")  # no edges out of classify: Command decides
    return builder.compile()


def route_all(graph, label: str) -> list[str]:
    header(label)
    print(graph.get_graph().draw_mermaid())
    teams = []
    for message in MESSAGES:
        state = graph.invoke({"message": message})
        teams.append(state["intent"])
        print(f"  {message!r}\n    -> {state['intent']}: {state['reply']}")
    return teams


def run(llm: BaseChatModel) -> dict:
    return {
        "edges": route_all(with_conditional_edges(llm), "1. Router with add_conditional_edges"),
        "command": route_all(with_command(llm), "2. The same router with Command(goto=...)"),
    }


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
