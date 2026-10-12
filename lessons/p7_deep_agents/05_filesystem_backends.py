"""The deep agent's environment: filesystem backends decide where each file really lives.

Academy: Introduction to Deep Agents M2 "The Deep Agent Environment", "Filesystem Backends".

Concepts:
  * the file tools always look the same to the model; a *backend* decides what they touch:
      StateBackend       the thread's state (default). Scratch space: per thread, checkpointed, then gone.
      FilesystemBackend  a real directory. `virtual_mode=True` makes `root_dir` the agent's "/" and blocks
                         `..` and absolute paths, so the agent can't wander outside it.
      StoreBackend       the LangGraph Store (Part 4): survives across threads. `namespace` picks the scope,
                         e.g. per team or per customer.
      CompositeBackend   routes by path prefix to any of the above: one tree, several homes.
  * permissions are separate from backends: `FilesystemPermission(operations=["write"], paths=["/docs/**"],
    mode="deny")` makes the docs read-only to the agent's tools, whatever the backend could do.
  * DocsHelp's layout (`docshelp.deep.ops_backend`, used again by the capstone):
      /docs/ (disk, read-only)   /skills/ (disk, read-only)   /memories/ (Store)   /reports/ (disk)   / (state)

Use case: the agent reads a doc, files a report, keeps a team note, scribbles a scratch file, and is stopped
from editing the docs.

Run:  uv run poe lesson 7.5
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.store.memory import InMemoryStore

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import DOCS_READ_ONLY, REPORTS_DIR, deep_agent, files_of, ops_backend, team_memory_namespace

TASK = (
    "Read /docs/03-sync-and-offline.md. Then: save a two-line summary to /reports/offline-summary.md, add "
    "'offline edits merge automatically' to /memories/facts.md, keep a scratch note in /scratch.md, and fix a "
    "typo in /docs/03-sync-and-offline.md if you see one."
)


def run(llm: BaseChatModel, reports_dir: Path | None = None) -> dict:
    reports_dir = reports_dir or REPORTS_DIR
    store = InMemoryStore()
    agent = deep_agent(
        llm, backend=ops_backend(reports_dir), permissions=[DOCS_READ_ONLY], store=store, system_prompt="Be brief."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": TASK}]})
    print(result["messages"][-1].text)

    header("Where the files ended up")
    on_disk = sorted(p.name for p in reports_dir.glob("*.md"))
    in_store = {item.key: item.value["content"] for item in store.search(team_memory_namespace(None))}
    in_state = files_of(result)
    print(f"  disk  {reports_dir.name}/: {on_disk}")
    print(f"  store {team_memory_namespace(None)}: {list(in_store)}")
    print(f"  state (this thread): {list(in_state)}")
    return {"disk": on_disk, "store": in_store, "state": in_state, "messages": result["messages"]}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
