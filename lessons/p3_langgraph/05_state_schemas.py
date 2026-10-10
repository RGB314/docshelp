"""State schemas: TypedDict, dataclass or Pydantic, and what each one checks.

Academy: Intro to LangGraph M2 L1 "State Schema".

Concepts:
  * the same ticket-priority graph with three state types
      TypedDict   type hints only, the lightest and most common choice
      dataclass   adds defaults and attribute access (`state.message`)
      Pydantic    validates at *runtime*: an invalid graph input raises `ValidationError`
  * the catch: Pydantic validates the graph's input, not what nodes return. If a node writes a model's
    raw text into state, nothing checks it. Validate model output where it enters the graph: structured
    output (lesson 1.3), or a check in the node itself, as `assess` does below.

Run:  uv run poe lesson 3.5
"""

from dataclasses import dataclass
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header

Priority = Literal["low", "normal", "high", "urgent"]
PRIORITIES = ("low", "normal", "high", "urgent")
MESSAGE = "Since this morning nobody on our team can open shared notebooks."


class TypedDictState(TypedDict, total=False):
    message: str
    priority: Priority


@dataclass
class DataclassState:
    message: str
    priority: Priority = "normal"


class PydanticState(BaseModel):
    message: str
    priority: Priority = "normal"


def build(llm: BaseChatModel, schema: type):
    def assess(state) -> dict:
        # TypedDict states are dicts; dataclass and Pydantic states use attributes.
        message = state["message"] if isinstance(state, dict) else state.message
        prompt = f"Reply with one word, the priority of this support request (low, normal, high or urgent): {message}"
        answer = llm.invoke(prompt).text.strip().strip(".").lower()
        # Validate the model's output yourself: no state schema checks what a node returns.
        return {"priority": answer if answer in PRIORITIES else "normal"}

    builder = StateGraph(schema)
    builder.add_node(assess)
    builder.add_edge(START, "assess")
    builder.add_edge("assess", END)
    return builder.compile()


def pydantic_graph(llm: BaseChatModel):
    return build(llm, PydanticState)


def run(llm: BaseChatModel) -> dict:
    results = {}
    for label, schema in (("typeddict", TypedDictState), ("dataclass", DataclassState), ("pydantic", PydanticState)):
        state = build(llm, schema).invoke({"message": MESSAGE})
        results[label] = state["priority"]  # invoke always returns a dict, whatever the schema
        print(f"{label:10} -> priority={results[label]!r}")

    header("Runtime validation of the input")
    try:
        pydantic_graph(llm).invoke({"message": MESSAGE, "priority": "critical"})
    except ValidationError as err:  # rejected before any node runs
        print(f"pydantic   rejected priority='critical': {err.errors()[0]['msg']}")
    print("typeddict and dataclass would accept it silently: their type hints are not checked at runtime.")
    return results


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
