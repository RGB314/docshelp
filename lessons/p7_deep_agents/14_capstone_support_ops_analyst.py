"""Capstone: the Support Ops analyst, a deep agent that turns the ticket queue into an incident report.

Academy: Introduction to Deep Agents M5 "The Sales Assistant", "The Advanced Sales Assistant" (the same
build applied to support operations).

Everything from Part 7 in one agent (`docshelp.deep.build_ops_analyst`):
  * planning with `write_todos` (7.1), a budget, the configured model (7.2) and tools (7.3)
  * one filesystem from four backends (7.5): /docs/ and /skills/ read-only on disk, /memories/ in the
    Store, /reports/ on disk, scratch files in state
  * HITL by path (7.4): writing anything under /reports/ waits for the lead's approval
  * skills (7.9): `log-analysis` and `refund-policy`, read only when needed
  * `AGENTS.md` memory (7.10): team conventions the agent reads every time and updates when told
  * a subagent team (7.11): `ticket-analyst` (structured output) and `docs-researcher`
  * it's also served by `uv run poe server` as `ops_analyst` (7.13), with the server's own persistence

Run:  uv run poe lesson 7.14            (the report lands in data/reports/)
"""

from collections.abc import Callable
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import REPORTS_DIR, build_ops_analyst, read_memory
from docshelp.ingest import get_retriever

REQUEST = (
    "Write an incident report about the Windows sync problems in the ticket queue: which tickets, likely cause, "
    "workaround from the docs, next steps. Save it as /reports/windows-sync.md."
)
FEEDBACK = "Good. From now on, start every report with a one-line TL;DR. Remember that for next time."


def approve(action: dict) -> dict:
    return {"type": "approve"}


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    *,
    reports_dir: Path | None = None,
    review: Callable[[dict], dict] = approve,
) -> dict:
    reports_dir = reports_dir or REPORTS_DIR
    store = InMemoryStore()
    agent = build_ops_analyst(llm, retriever=retriever, store=store, reports_dir=reports_dir)
    config = {"configurable": {"thread_id": "incident-1"}}
    steps, approvals = [], []

    def turn(text: str) -> dict:
        print(f"\nLead: {text}")
        # durability="sync": save each step's checkpoint before the next step starts. The default ("async")
        # queues saves on a thread pool; on small machines a long run like this one can fill the pool with saves
        # waiting on each other, and LangGraph 1.2 then deadlocks. Sync costs a little speed and can't pile up.
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config, durability="sync")
        while "__interrupt__" in result:
            decisions = []
            for action in result["__interrupt__"][0].value["action_requests"]:
                decisions.append(review(action))
                approvals.append((action["args"].get("file_path"), decisions[-1]["type"]))
                print(
                    f"  approval needed: {action['name']} {action['args'].get('file_path')} -> {decisions[-1]['type']}"
                )
            result = agent.invoke(Command(resume={"decisions": decisions}), config, durability="sync")
        print(f"Analyst: {result['messages'][-1].text}")
        return result

    result = turn(REQUEST)
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                steps.append(call["name"] if call["name"] != "task" else f"task:{call['args'].get('subagent_type')}")
    header(f"Steps: {' -> '.join(steps)}")
    reports = {p.name: p.read_text(encoding="utf-8") for p in sorted(reports_dir.glob("*.md"))}
    for name, text in reports.items():
        print(f"--- {reports_dir.name}/{name}\n{text}")

    turn(FEEDBACK)
    header("Team memory (/memories/AGENTS.md)")
    print(read_memory(store))
    return {"steps": steps, "approvals": approvals, "reports": reports, "memory": read_memory(store)}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
