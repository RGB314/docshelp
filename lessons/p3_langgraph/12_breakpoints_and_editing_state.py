"""Breakpoints and editing state: pause before tools run, then change what happens.

Academy: Intro to LangGraph M3 L2 "Breakpoints", M3 L3 "Editing State and Human Feedback".

Concepts:
  * a static breakpoint: `compile(interrupt_before=["tools"])` pauses every run before the tools node
    (`interrupt_after` pauses after a node). Needs a checkpointer, like every pause.
  * inspect the pause: `get_state(config).next` is the node about to run; the last AI message holds the
    proposed tool calls
  * continue: `invoke(None, config)` resumes from the saved state (None means "no new input")
  * edit state: `update_state(config, {"messages": [msg]})`. With `add_messages`, a message with the same
    `id` *replaces* the old one (lesson 3.6), so a human can correct the tool arguments before they run.
  * act as a node: `update_state(..., as_node="tools")` writes the update as if `tools` had run. Here a
    human answers the tool call themselves, so the pricing tool never runs. The same pattern adds
    human feedback as if a `human_feedback` node had produced it.
  * breakpoints are fixed at compile time; lesson 3.13 pauses from *inside* a node, only when needed

Run:  uv run poe lesson 3.12
"""

import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.react import build_react_agent
from docshelp.tools import make_search_docs, quote_price

QUESTION = "How much is the Team plan for 12 members, billed monthly?"


def ask(agent, text: str) -> tuple[dict, AIMessage | None]:
    """Start a run; return its config and the AI message whose tool calls are waiting at the breakpoint."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
    state = agent.get_state(config)
    if not state.next:
        print("  (the model answered without calling a tool, so there was nothing to pause for)")
        return config, None
    pending = state.values["messages"][-1]
    print(f"  paused before {state.next}: {[(c['name'], c['args']) for c in pending.tool_calls]}")
    return config, pending


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    agent = build_react_agent(
        llm, [make_search_docs(retriever), quote_price], checkpointer=InMemorySaver(), interrupt_before=["tools"]
    )
    result: dict = {"paused_at": None, "edited_tool_result": None, "answered_for_tool": None}

    header("1. Edit the pending tool call, then continue")
    print(f"User: {QUESTION}")
    config, pending = ask(agent, QUESTION)
    if pending:
        result["paused_at"] = agent.get_state(config).next
        print("  The customer adds: 'Sorry, we're actually 15 people.' A human fixes the call:")
        calls = [{**c, "args": {**c["args"], "members": 15}} for c in pending.tool_calls]
        agent.update_state(config, {"messages": [AIMessage(pending.content, tool_calls=calls, id=pending.id)]})
        agent.invoke(None, config)  # resume: the edited call runs
        messages = agent.get_state(config).values["messages"]
        result["edited_tool_result"] = next(m.text for m in reversed(messages) if isinstance(m, ToolMessage))
        print(f"  tool ran with the edit: {result['edited_tool_result']}")
        print(f"DocsHelp: {messages[-1].text}")

    header("2. Answer for the tool (as_node='tools'), so it never runs")
    config, pending = ask(agent, QUESTION)
    if pending:
        answers = [ToolMessage("Pricing is down for maintenance.", tool_call_id=c["id"]) for c in pending.tool_calls]
        agent.update_state(config, {"messages": answers}, as_node="tools")
        print(f"  next node is now: {agent.get_state(config).next}")
        agent.invoke(None, config)
        messages = agent.get_state(config).values["messages"]
        result["answered_for_tool"] = next(m.text for m in reversed(messages) if isinstance(m, ToolMessage))
        print(f"DocsHelp: {messages[-1].text}")
    return result


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
