"""Messages, threads and checkpointers; human in the loop: a deep agent that pauses before it publishes.

Academy: Introduction to Deep Agents M1 "Messages, Threads, and Checkpointers", "HITL", "Test Your Skills".

Concepts:
  * a deep agent is a LangGraph graph, so persistence works as in Part 3: give it a checkpointer and a
    `thread_id`, and each thread keeps its own messages *and its own files* (the default `StateBackend`
    stores files in the thread's state). A new thread starts empty.
  * `interrupt_on={"delete": True}`: the HITL middleware from lesson 2.7, configured in one argument. It
    applies to the main agent and is inherited by declarative subagents (which can override it).
  * path-scoped approval: `FilesystemPermission(operations=["write"], paths=["/published/**"],
    mode="interrupt")` pauses only for writes under /published/. Drafts are free; publishing needs a person.
    (`mode="deny"` refuses outright, lesson 7.5.)
  * resuming is the same `Command(resume={"decisions": [...]})` as lesson 2.7: approve, edit or reject.

Use case: a support lead drafts a customer status note with the agent, then approves publishing it.

Run:  uv run poe lesson 7.4
"""

from collections.abc import Callable

from deepagents import FilesystemPermission
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import deep_agent, files_of

PROMPT = "You help the Nimbus Notes support lead write customer status notes. Keep notes under 60 words."
SHIFT_1 = [
    "Draft a status note for eu-west customers about delayed sync (incident INC-2041) in /drafts/eu-west.md.",
    "Looks good. Publish it as /published/eu-west.md.",
]
SHIFT_2 = "Which files do we have?"


def approve(action: dict) -> dict:
    return {"type": "approve"}


def run(llm: BaseChatModel, review: Callable[[dict], dict] = approve) -> dict:
    agent = deep_agent(
        llm,
        system_prompt=PROMPT,
        interrupt_on={"delete": True},  # tool-level: always ask before deleting a file
        permissions=[FilesystemPermission(operations=["write"], paths=["/published/**"], mode="interrupt")],
        checkpointer=InMemorySaver(),  # threads need one
    )
    interrupts = []

    def ask(text: str, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        print(f"\n[{thread_id}] Lead: {text}")
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        while "__interrupt__" in result:  # paused before a guarded call
            decisions = []
            for action in result["__interrupt__"][0].value["action_requests"]:
                interrupts.append({"tool": action["name"], "args": action["args"]})
                decisions.append(review(action))
                path = action["args"].get("file_path", "")
                print(f"  needs approval: {action['name']}({path}) -> {decisions[-1]['type']}")
            result = agent.invoke(Command(resume={"decisions": decisions}), config)
        print(f"[{thread_id}] Agent: {result['messages'][-1].text}")
        return result

    for text in SHIFT_1:
        shift_1 = ask(text, "shift-1")
    shift_2 = ask(SHIFT_2, "shift-2")

    header("Each thread has its own messages and files")
    threads = {}
    for name, state in (("shift-1", shift_1), ("shift-2", shift_2)):
        threads[name] = {"messages": len(state["messages"]), "files": sorted(files_of(state))}
        print(f"  {name}: {threads[name]['messages']} messages, files {threads[name]['files']}")
    return {"interrupts": interrupts, "threads": threads, "files": files_of(shift_1)}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
