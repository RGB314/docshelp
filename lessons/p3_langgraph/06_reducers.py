"""Reducers: how a state key combines updates from several nodes.

Academy: Intro to LangGraph M2 L2 "State Reducers".

Concepts:
  * the default reducer *overwrites* a key. Two nodes writing the same key in the same step (parallel
    branches) is ambiguous, so LangGraph raises `InvalidUpdateError`.
  * `Annotated[list[str], reducer]` attaches a reducer: `operator.add` appends lists; a custom function
    can do anything: here `merge_tags` merges, deduplicates and sorts
  * `add_messages` (used by `MessagesState`) appends messages, *replaces* one whose `id` already exists
    (edit a message), and deletes on `RemoveMessage(id=...)`. Parts 2.6 and 3.8–3.9 delete messages with it.
  * parallel branches like the two taggers here are covered properly in Part 4

Run:  uv run poe lesson 3.6
"""

from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header

MESSAGE = "I was charged for 12 seats but sync only works for 10 of us."
TOPICS = ("billing", "sync", "login", "sharing", "export")


def merge_tags(current: list[str] | None, new: list[str] | None) -> list[str]:
    """A custom reducer: (current value, update) -> new value."""
    return sorted(set(current or []) | set(new or []))


class Overwrite(TypedDict, total=False):
    message: str
    tags: list[str]  # no reducer: last write wins, and two writes in one step are an error


class Merged(TypedDict, total=False):
    message: str
    tags: Annotated[list[str], merge_tags]


def build(llm: BaseChatModel, schema: type):
    def keyword_tagger(state) -> dict:
        return {"tags": [t for t in TOPICS if t in state["message"].lower()]}

    def llm_tagger(state) -> dict:
        prompt = f"Reply with one word from {', '.join(TOPICS)}: the main topic of: {state['message']}"
        topic = llm.invoke(prompt).text.strip().strip(".").lower()
        return {"tags": [topic] if topic in TOPICS else []}

    builder = StateGraph(schema)
    builder.add_node(keyword_tagger)
    builder.add_node(llm_tagger)
    for node in ("keyword_tagger", "llm_tagger"):  # both run in the same step, in parallel
        builder.add_edge(START, node)
        builder.add_edge(node, END)
    return builder.compile()


def run(llm: BaseChatModel) -> dict:
    header("1. No reducer: two parallel writes to `tags`")
    try:
        build(llm, Overwrite).invoke({"message": MESSAGE})
        without = "no error"
    except InvalidUpdateError as err:
        without = type(err).__name__
        print(f"  {without}: {str(err).splitlines()[0]}")

    header("2. A custom reducer merges both writes")
    tags = build(llm, Merged).invoke({"message": MESSAGE})["tags"]
    print(f"  tags = {tags}")

    header("3. add_messages: append, replace by id, remove by id")
    history = add_messages([], [HumanMessage("Hi", id="1"), AIMessage("Hello!", id="2")])
    history = add_messages(history, [AIMessage("Hello! (edited)", id="2")])  # same id: replaced
    history = add_messages(history, [HumanMessage("spam", id="3")])
    history = add_messages(history, [RemoveMessage(id="3")])  # removed
    for message in history:
        print(f"  {message.type:5} id={message.id}: {message.content}")
    return {"without_reducer": without, "with_reducer": tags, "messages": history}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
