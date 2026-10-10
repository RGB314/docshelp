"""Streaming a graph: full state, per-node updates, LLM tokens per node, and events.

Academy: Intro to LangGraph M3 L1 "Streaming".

Lesson 1.7 streamed a `create_agent` agent. A graph streams the same way, with more control:
  * `stream_mode="values"`: the full state after every step (good for UIs that redraw everything)
  * `stream_mode="updates"`: only what each node changed (good for progress and logs)
  * several modes at once: `stream_mode=["values", "updates"]` yields `(mode, chunk)` pairs
  * `stream_mode="messages"`: LLM tokens, with metadata saying which node produced them, so a UI can
    show only the answer node's tokens (`metadata["langgraph_node"] == "assistant"`)
  * `astream_events(version="v2")`: every event from every runnable inside (model start, each token, tool
    start/end, ...), for custom tracing or very detailed UIs
  * other modes: "custom" (lesson 1.7), "debug" and "checkpoints" (every saved checkpoint)

Run:  uv run poe lesson 3.11
"""

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.react import build_react_agent
from docshelp.tools import make_search_docs, quote_price

QUESTION = {"messages": [{"role": "user", "content": "Does Nimbus Notes work offline?"}]}


async def collect_events(agent) -> list[str]:
    kinds = []
    async for event in agent.astream_events(QUESTION, version="v2"):
        if event["event"] not in kinds:
            kinds.append(event["event"])
            print(f"  {event['event']:24} from {event['name']}")
    return kinds


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    agent = build_react_agent(llm, [make_search_docs(retriever), quote_price])

    header('1. stream_mode=["values", "updates"]: full state and per-node deltas, in one run')
    values, updates = [], []
    for mode, chunk in agent.stream(QUESTION, stream_mode=["values", "updates"]):
        if mode == "values":
            values.append(len(chunk["messages"]))
            print(f"  values:  state now has {values[-1]} messages")
        else:
            updates.extend(chunk)
            print(f"  updates: {', '.join(chunk)} finished")

    header('2. stream_mode="messages": tokens from the assistant node only')
    tokens = []
    for token, metadata in agent.stream(QUESTION, stream_mode="messages"):
        if metadata["langgraph_node"] == "assistant" and token.text:
            tokens.append(token.text)
            print(token.text, end="|", flush=True)
    print()

    header('3. astream_events(version="v2"): the first event of each kind')
    events = asyncio.run(collect_events(agent))
    return {"values": values, "updates": updates, "tokens": tokens, "events": events}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
