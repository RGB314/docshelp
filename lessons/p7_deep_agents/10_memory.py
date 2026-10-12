"""Memory: an `AGENTS.md` the agent always reads, and edits itself when it learns something lasting.

Academy: Introduction to Deep Agents M3 "Memory".

Concepts:
  * `memory=["/memories/AGENTS.md"]` loads that file into the system prompt at the start of every thread
    (`MemoryMiddleware`). Same idea as this repo's own `AGENTS.md` for coding agents: standing instructions
    and facts, always in context.
  * the agent updates it with its ordinary file tools (`edit_file`), so learning needs no special API. Its
    prompt tells it to save lasting preferences, not one-off details.
  * where it lives decides how long it lasts: on the default `StateBackend` it would vanish with the thread.
    Routed to a `StoreBackend` (lesson 7.5) it's shared by every thread, and with a persistent store
    (Postgres, or the server's store in Part 8) it survives restarts.
  * compared with Part 4's memory: there, a separate extractor wrote structured profiles into the Store.
    Here the agent curates one plain-text file. Simpler, readable by people, and good for instructions.

Use case: the support lead corrects the report format once; every later conversation follows it.

Run:  uv run poe lesson 7.10
"""

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain_core.language_models import BaseChatModel
from langgraph.store.memory import InMemoryStore

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import MEMORY_PATH, deep_agent, read_memory, seed_memory, team_memory_namespace

PROMPT = (
    "You write short incident updates for the Nimbus Notes support lead. When the lead states a lasting "
    f"preference, add it as a bullet to {MEMORY_PATH} with edit_file before answering."
)
THREAD_1 = "From now on, start every update with the ticket id in bold. Please remember that."
THREAD_2 = "Write a one-line update for ticket NIM-3A4B5C: sync stuck at 99% for a 4 GB notebook."


def run(llm: BaseChatModel) -> dict:
    store = InMemoryStore()
    seed_memory(store)  # the team's starting notes
    backend = CompositeBackend(
        default=StateBackend(), routes={"/memories/": StoreBackend(namespace=team_memory_namespace)}
    )
    agent = deep_agent(llm, backend=backend, memory=[MEMORY_PATH], store=store, system_prompt=PROMPT)

    header("Memory before")
    print(read_memory(store))
    for thread, text in (("monday", THREAD_1), ("tuesday", THREAD_2)):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": text}]}, {"configurable": {"thread_id": thread}}
        )
        print(f"\n[{thread}] Lead: {text}\n[{thread}] Agent: {result['messages'][-1].text}")
    header("Memory after (shared by every thread)")
    print(read_memory(store))
    return {"memory": read_memory(store), "answer": result["messages"][-1].text}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
