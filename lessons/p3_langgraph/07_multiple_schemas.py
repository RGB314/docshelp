"""Multiple schemas: a narrow public interface, with private working state inside.

Academy: Intro to LangGraph M2 L3 "Multiple Schemas".

Concepts:
  * `StateGraph(OverallState, input_schema=AskInput, output_schema=AskOutput)`: callers send only
    `question` and get back only `answer` and `sources`. Retrieved documents and drafts stay internal.
  * private state: a node can write keys that aren't in the overall state (`draft`) for a later node
    that declares them in its own input type (`polish(state: DraftState)`). They never reach the caller.
  * why: a stable API for other services (and LangGraph Server, Part 8) while internals change freely,
    and no internal data leaking into responses

Run:  uv run poe lesson 3.7
"""

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import format_docs, get_retriever

QUESTION = "How do I export a notebook as a PDF?"


class AskInput(TypedDict):
    question: str


class AskOutput(TypedDict):
    answer: str
    sources: list[str]


class OverallState(AskInput, AskOutput, total=False):
    documents: list  # internal: retrieved chunks


class DraftState(TypedDict):  # private: passed from `draft` to `polish` only
    draft: str
    sources: list[str]


def build(llm: BaseChatModel, retriever: BaseRetriever):
    def retrieve(state: AskInput) -> dict:
        return {"documents": retriever.invoke(state["question"])}

    def draft(state: OverallState) -> DraftState:
        prompt = f"Answer briefly using only these docs.\n\n{format_docs(state['documents'])}\n\nQ: {state['question']}"
        sources = sorted({d.metadata.get("source", "?") for d in state["documents"]})
        return {"draft": llm.invoke(prompt).text, "sources": sources}

    def polish(state: DraftState) -> AskOutput:
        answer = f"{state['draft'].strip()}\n\nSources: {', '.join(state['sources'])}"
        return {"answer": answer, "sources": state["sources"]}

    builder = StateGraph(OverallState, input_schema=AskInput, output_schema=AskOutput)
    builder.add_node(retrieve)
    builder.add_node(draft)
    builder.add_node(polish)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "draft")
    builder.add_edge("draft", "polish")
    builder.add_edge("polish", END)
    return builder.compile()


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    graph = build(llm, retriever)
    header("Input and output schemas")
    print(f"  input:  {list(graph.get_input_jsonschema()['properties'])}")
    print(f"  output: {list(graph.get_output_jsonschema()['properties'])}")
    header(f"invoke({{'question': {QUESTION!r}}})")
    result = graph.invoke({"question": QUESTION})
    print(f"  returned keys: {sorted(result)}   (no documents, no draft)")
    print(f"\n{result['answer']}")
    return result


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
