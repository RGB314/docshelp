"""Map-reduce with `Send`: take notes on each retrieved doc in parallel, then answer from the notes.

Academy: Intro to LangGraph M4 L3 "Map-reduce".

Concepts:
  * fan-out in lesson 4.1 had a fixed number of branches, set when the graph was built. Here the number
    depends on the data: one branch per retrieved document.
  * `Send("take_note", payload)`: returned from a conditional edge, each `Send` runs the node once with its
    own payload (which needn't be the graph's state). A list of them runs in parallel: the *map*.
  * the mapped node writes to a reducer key (`notes`, `operator.add`), and the next node reads them all:
    the *reduce*
  * why notes first: each doc is read in a small, focused call, and the final call sees short notes
    instead of full documents. This is how you scale to many sources (and Part 6's deep research).

Run:  uv run poe lesson 4.3
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever

QUESTION = "Can I use Nimbus Notes without internet, and does it sync afterwards?"


class State(TypedDict, total=False):
    question: str
    docs: list[dict]
    notes: Annotated[list[str], operator.add]  # the map's results
    answer: str


class NoteTask(TypedDict):  # what each Send carries: not the graph's state
    question: str
    doc: dict


def build(llm: BaseChatModel, retriever: BaseRetriever):
    def retrieve(state: State) -> dict:
        found = retriever.invoke(state["question"])
        return {"docs": [{"source": d.metadata.get("source", "?"), "text": d.page_content} for d in found]}

    def map_to_notes(state: State) -> list[Send]:
        return [Send("take_note", {"question": state["question"], "doc": doc}) for doc in state["docs"]]

    def take_note(task: NoteTask) -> dict:
        prompt = (
            f"In one sentence, what does this excerpt say that helps answer: {task['question']}\n"
            f"If nothing, reply 'nothing relevant'.\n\nExcerpt:\n{task['doc']['text']}"
        )
        return {"notes": [f"[{task['doc']['source']}] {llm.invoke(prompt).text.strip()}"]}

    def write_answer(state: State) -> dict:
        notes = "\n".join(sorted(state["notes"]))
        prompt = (
            f"Answer the question from these notes and cite the sources.\n\n{notes}\n\nQuestion: {state['question']}"
        )
        return {"answer": llm.invoke(prompt).text}

    builder = StateGraph(State)
    builder.add_node(retrieve)
    builder.add_node(take_note)
    builder.add_node(write_answer)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges("retrieve", map_to_notes, ["take_note"])
    builder.add_edge("take_note", "write_answer")  # runs once, after every take_note has finished
    builder.add_edge("write_answer", END)
    return builder.compile()


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    graph = build(llm, retriever)
    header(f"User: {QUESTION}")
    state = graph.invoke({"question": QUESTION})
    print(f"Mapped over {len(state['docs'])} documents. Notes:")
    for note in sorted(state["notes"]):
        print(f"  {note}")
    print(f"\nDocsHelp: {state['answer']}")
    return state


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
