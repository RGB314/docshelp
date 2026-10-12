"""Dynamic subagents: dispatching subagents from code, in loops and in parallel.

Academy: Introduction to Deep Agents M4 "Dynamic Subagents".

Concepts:
  * with both subagents and the interpreter (lesson 7.7), the REPL gets a global `task({...})`. Instead of
    one model-chosen `task` call per item, the agent writes a program: loop over the tickets, start a
    subagent for each (`Promise.all` runs them in parallel), branch on the results, aggregate.
  * `responseSchema` (JSON Schema) makes each subagent return a typed object the code can use directly:
    `results.filter(r => r.severity === "high")` with no text parsing.
  * the model opts in when the request reads like a *workflow* over many items; that word is a deliberate
    lever in the interpreter's prompt.
  * caution: dispatches from code skip `interrupt_on` on the parent, so gate the `eval` tool itself, or put
    approval inside the subagent, when a subagent can do something irreversible.

Use case: triage every open ticket with a triage subagent, in parallel, and report the high-severity ones.

Run:  uv run poe lesson 7.12
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_quickjs import CodeInterpreterMiddleware

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import budget, deep_agent, list_tickets

TRIAGER = {
    "name": "ticket-triager",
    "description": "Triages one support ticket: which team owns it and how severe it is.",
    "system_prompt": (
        "Classify the ticket you are given. team: sync, billing, identity or docs. severity: high if data is "
        "missing or a team is blocked, else normal."
    ),
    "tools": [],
    "middleware": [budget(3)],
}
TASK = (
    "Run a workflow: in one eval, get the open tickets with tools.listTickets({status: 'open'}), triage each "
    "one in parallel with task({subagentType: 'ticket-triager', description: <the ticket JSON>, responseSchema: "
    "{type: 'object', properties: {team: {type: 'string'}, severity: {type: 'string'}}}}), and return the ids "
    "of the high-severity tickets with their teams."
)


def run(llm: BaseChatModel) -> dict:
    agent = deep_agent(
        llm,
        tools=[list_tickets],
        subagents=[TRIAGER],
        middleware=[CodeInterpreterMiddleware(ptc=["list_tickets"], timeout=10)],
        system_prompt="You are the Nimbus Notes Support Ops lead's assistant.",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": TASK}]})
    evals = [m.text for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "eval"]
    header("The workflow's result")
    for output in evals:
        print(output)
    header("Answer")
    print(result["messages"][-1].text)
    return {"evals": evals, "answer": result["messages"][-1].text}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
