"""Summarization and offloading: keeping a long-running agent's context small without losing anything.

Academy: Introduction to Deep Agents M3 "Summarization and Offload".

Concepts:
  * offloading large tool results: when a tool returns more than `tool_token_limit_before_evict` tokens
    (20,000 by default), `FilesystemMiddleware` saves the full result to `/large_tool_results/<call id>` and
    gives the model a short preview plus the path. The model then `grep`s or `read_file`s just the part it
    needs. Here the limit is lowered to 500 so a ~2,500-token log is offloaded.
  * summarization with offload: Part 2.6's `SummarizationMiddleware` *drops* old messages for a summary. The
    deep-agents version first appends them to `/conversation_history/<session>.md`, and the summary says
    where, so nothing is lost: the agent can read the old turns back.
  * it doesn't rewrite the thread: `result["messages"]` keeps the full log (for replay, evals, the UI); only
    what the *model* receives is compacted. Defaults come from the model's context window; here
    `trigger=("messages", 8)` and `keep=("messages", 2)` make it happen within a short demo.
  * both are configured by passing your own instance in `middleware=[...]`: it replaces the default one of
    the same name. (`create_summarization_tool_middleware` adds a `compact_conversation` tool so the model can
    also compact on demand.)

Use case: a long investigation of one ticket. A huge client log arrives, and the chat keeps going.

Run:  uv run poe lesson 7.8
"""

from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware, SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from docshelp.config import DATA_DIR, describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import deep_agent, files_of

LOG = DATA_DIR / "logs" / "sync-client.log"
TURNS = [
    "Ticket NIM-7A8B9C: why does sync fail? Fetch the client log and count the ERROR lines with grep.",
    "Which error is most common?",
    "What does the log show just before the errors start?",
    "Draft a one-sentence fix for the customer.",
]


@tool
def fetch_client_log(ticket_id: str) -> str:
    """Fetch the sync client log the customer attached to a ticket. Logs are long."""
    return LOG.read_text(encoding="utf-8")


def run(llm: BaseChatModel, turns: list[str] = TURNS) -> dict:
    backend = StateBackend()  # both offloads land in the thread's files
    agent = deep_agent(
        llm,
        tools=[fetch_client_log],
        system_prompt="You investigate Nimbus Notes sync problems. Be brief.",
        checkpointer=InMemorySaver(),
        middleware=[
            FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=500),
            SummarizationMiddleware(llm, backend=backend, trigger=("messages", 8), keep=("messages", 2)),
        ],
    )
    config = {"configurable": {"thread_id": "investigation"}}
    for turn in turns:
        result = agent.invoke({"messages": [{"role": "user", "content": turn}]}, config)
        print(f"\nLead: {turn}\nAgent: {result['messages'][-1].text}")

    files = files_of(result)
    header("Offloaded to files")
    for path, content in files.items():
        print(f"  {path} ({len(content):,} characters)")
    history = next((c for p, c in files.items() if p.startswith("/conversation_history/")), "")
    print(f"\nThe thread still holds all {len(result['messages'])} messages.")
    print("The model only gets the summary and the most recent turns.")
    return {"files": files, "history": history, "messages": result["messages"]}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
