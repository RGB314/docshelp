"""A ReAct agent built by hand: the graph behind `create_agent`.

Academy: Intro to LangGraph M1 L4 "Chain", M1 L5 "Router", M1 L6 "Agent", M1 L7 "Agent with Memory";
LangGraph Essentials L4 "Memory".

Concepts (the graph is `src/docshelp/react.py::build_react_agent`):
  * `MessagesState`: a `messages` key with the `add_messages` reducer, so nodes *append* messages
  * the chain: a node calling a model with tools bound (`llm.bind_tools`)
  * the router: `tools_condition` sends tool calls to the prebuilt `ToolNode`, answers to END
  * the agent: the edge `tools → assistant` closes the loop (reason, act, observe, repeat: "ReAct")
  * memory: the same checkpointer + `thread_id` you used with `create_agent` in lesson 1.8, here passed to
    `compile()`, because it is a LangGraph feature that `create_agent` simply passes through

Run:  uv run poe lesson 3.3
"""

import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.react import build_react_agent
from docshelp.tools import make_search_docs, quote_price

CONVERSATION = [
    "What does the Team plan cost for 12 members, billed monthly?",
    "And for 20 members?",  # needs turn 1 to know which plan and billing cycle
]


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    agent = build_react_agent(llm, [make_search_docs(retriever), quote_price], checkpointer=InMemorySaver())
    header("The graph")
    print(agent.get_graph().draw_mermaid())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    paths = []
    for text in CONVERSATION:
        header(f"User: {text}")
        path = []
        for update in agent.stream({"messages": [{"role": "user", "content": text}]}, config, stream_mode="updates"):
            for node, delta in update.items():
                path.append(node)
                for message in delta["messages"]:
                    calls = getattr(message, "tool_calls", None)
                    print(f"  {node:9} {calls or message.text[:120]}")
        paths.append(path)
    answer = agent.get_state(config).values["messages"][-1].text
    print(f"\nDocsHelp: {answer}")
    return {"paths": paths, "answer": answer}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
