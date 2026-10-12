"""Running a deep agent: a harness for long, multi-step work, built on `create_agent`.

Academy: Introduction to Deep Agents M1 "Overview", "Running a Deep Agent".

Concepts:
  * `create_deep_agent(model, tools, system_prompt, ...)` returns the same kind of compiled graph as
    `create_agent` (lesson 1.6). The difference is the harness it adds around your tools:
      - files: `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep` on a virtual filesystem.
        By default the files live in the agent's state (`result["files"]`), per thread.
      - delegation: `task` hands a self-contained job to a subagent (lesson 7.11)
      - summarization that offloads old history to a file (lesson 7.8)
  * planning: `TodoListMiddleware` adds `write_todos`, a checklist the agent keeps in state and updates as it
    works. Long tasks stay on track because the plan is re-read every turn, not remembered.
  * why files: a deep agent writes notes and drafts instead of holding everything in its context window,
    then reads back only what it needs. That's what lets it work for many steps.
  * a budget: deep agents allow ~10,000 steps, so every lesson's agent stops after a few model calls
    (`docshelp.deep.budget`, the call limit from lesson 2.5).

Use case: triage the open Windows tickets and leave a triage note for the next shift.

Run:  uv run poe lesson 7.1
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import deep_agent, files_of, ops_tools
from docshelp.ingest import get_retriever

TASK = (
    "Triage the open Windows tickets. Plan first, check the docs for known fixes, then write a short triage "
    "note to /triage.md for the next shift: one line per ticket with the likely fix."
)


def run(llm: BaseChatModel, retriever: BaseRetriever | None = None) -> dict:
    agent = deep_agent(llm, tools=ops_tools(retriever), planning=True, system_prompt="You are Nimbus Notes support.")
    header(f"Task: {TASK}")
    steps = []
    state: dict = {}
    for mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": TASK}]}, stream_mode=["updates", "values"]
    ):
        if mode == "values":
            state = chunk
            continue
        for update in chunk.values():
            for message in (update or {}).get("messages", []):
                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        steps.append(call["name"])
                        print(f"  {call['name']}({', '.join(f'{k}=' for k in call['args'])})")
    header("Plan (write_todos)")
    for todo in state.get("todos", []):
        print(f"  [{todo['status']}] {todo['content']}")
    files = files_of(state)
    header("Files the agent wrote")
    for path, content in files.items():
        print(f"--- {path}\n{content}")
    header("Answer")
    print(state["messages"][-1].text)
    return {"steps": steps, "todos": state.get("todos", []), "files": files, "answer": state["messages"][-1].text}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
