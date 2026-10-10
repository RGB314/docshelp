"""Short-term memory: a checkpointer keeps each conversation's history for you.

Academy: LangChain Essentials L6 "Memory", Intro to LangChain M1 L3 "Short-Term Memory".

Concepts:
  * lesson 1.1 resent the history by hand; an agent with a checkpointer saves its state after every
    step, keyed by `thread_id`, and reloads it on the next call
  * one thread_id = one conversation: other threads don't see it
  * `agent.get_state(config)` shows exactly what was saved
  * InMemorySaver is lost on exit; Part 3 swaps in a database-backed saver (same code otherwise)

Run:  uv run poe lesson 1.8
"""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header


def run(llm: BaseChatModel) -> dict:
    agent = create_agent(
        llm, tools=[], system_prompt="You are DocsHelp for Nimbus Notes. Be brief.", checkpointer=InMemorySaver()
    )
    acme = {"configurable": {"thread_id": "acme-support-chat"}}
    other = {"configurable": {"thread_id": "someone-else"}}

    def say(text: str, config: dict) -> str:
        reply = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)["messages"][-1].text
        print(f"[{config['configurable']['thread_id']}] User: {text}\n  DocsHelp: {reply}")
        return reply

    header("1. Two turns in the same thread: the agent remembers")
    say("Hi! Our company Acme has 12 seats on the Team plan.", acme)
    say("How many seats do we have?", acme)

    header("2. A different thread starts empty")
    say("How many seats do we have?", other)

    header("3. What the checkpointer saved for the first thread")
    saved = agent.get_state(acme).values["messages"]
    for m in saved:
        print(f"  {m.type:6} {m.text[:70]}")

    return {"saved_messages": len(saved)}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
