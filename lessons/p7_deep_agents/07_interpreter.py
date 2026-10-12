"""The interpreter: a JavaScript REPL inside the agent, for working *with* tool results instead of reading them.

Academy: Introduction to Deep Agents M2 "Interpreter".

Concepts:
  * `CodeInterpreterMiddleware` (package `langchain-quickjs`) adds an `eval` tool that runs JavaScript in
    QuickJS compiled to WebAssembly. Unlike the shell (7.6) it is sandboxed by construction: no files, no
    network, no processes; just a memory limit (64 MB) and a per-call time limit (5 s by default).
  * programmatic tool calling (PTC): `ptc=["list_tickets"]` exposes that tool inside the REPL as
    `tools.listTickets({...})`. One `eval` can call tools in loops or in parallel (`Promise.all`) and return
    only the aggregate, so a hundred tool results never enter the context window.
  * state persists across `eval` calls in a thread (`mode="thread"`, the default): define a variable once,
    use it in the next call. `"turn"` and `"call"` reset it sooner.
  * caution: tools called from code bypass `interrupt_on` (no per-call approval), so only expose tools that
    are safe to run unattended. Read-only `list_tickets` is; `send_reply` would not be.

Use case: break down the ticket queue by platform and priority with one piece of code.

Run:  uv run poe lesson 7.7
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_quickjs import CodeInterpreterMiddleware

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep import deep_agent, list_tickets

TASK = (
    "Using the eval tool, fetch all tickets with tools.listTickets({status: 'any'}) (JSON lines), and count "
    "them by platform and by priority. Then tell me which platform has the most open high-priority tickets."
)


def run(llm: BaseChatModel) -> dict:
    interpreter = CodeInterpreterMiddleware(ptc=["list_tickets"], timeout=10)
    agent = deep_agent(
        llm, tools=[list_tickets], middleware=[interpreter], system_prompt="You are a Support Ops data analyst."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": TASK}]})
    evals = [m.text for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "eval"]
    header("What the code returned")
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
