"""Human in the loop: a person approves, edits or rejects a tool call before it runs.

Academy: Intro to LangChain M3 L3 "Human In The Loop"; LangChain Essentials L9 "Human in the Loop".

Concepts:
  * `HumanInTheLoopMiddleware(interrupt_on={...})`: list the tools that need a human; others run freely
  * the run *pauses*: `invoke` returns `__interrupt__` with the proposed `action_requests`, and the state is
    saved by the checkpointer (required), so the reviewer can answer seconds or days later
  * resume with `Command(resume={"decisions": [...]})`, one decision per action:
      approve   run the call as proposed
      edit      run a corrected call (`edited_action`): here the reviewer raises the priority
      reject    don't run it; the `message` goes back to the model, which carries on
  * `allowed_decisions` limits what a reviewer may do per tool
  * Part 3 shows the same idea one layer down: raw `interrupt()` inside a LangGraph node

Run:  uv run poe lesson 2.7              (scripted reviewer: approve, edit, reject)
      uv run poe lesson 2.7 --review     (you are the reviewer)
"""

import argparse
import uuid
from collections.abc import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.customers import Customer
from docshelp.tools import open_ticket

Review = Callable[[dict, int], dict]  # (action request, case number) -> decision

CASES = [
    "Sync has been stuck on my laptop for two days. Please open a ticket.",
    "After syncing, half of the notes in my 'Research' notebook vanished! Please open a ticket.",
    "Please open a ticket asking you to add a dark mode.",
]


def with_priority(action: dict, priority: str) -> dict:
    """An `edit` decision: the same tool call with a corrected priority."""
    return {"type": "edit", "edited_action": {"name": action["name"], "args": {**action["args"], "priority": priority}}}


def scripted_review(action: dict, case: int) -> dict:
    """What a support lead might decide for each case."""
    if case == 0:
        return {"type": "approve"}
    if case == 1:  # data loss is never "normal": fix the priority, then run it
        return with_priority(action, "urgent")
    return {"type": "reject", "message": "Feature requests go to the ideas board, not tickets. Point them there."}


def terminal_review(action: dict, case: int) -> dict:
    choice = input("  [a]pprove, [e]dit priority, [r]eject? ").strip().lower()[:1]
    if choice == "e":
        return with_priority(action, input("  new priority (low/normal/high/urgent): ").strip() or "high")
    if choice == "r":
        return {"type": "reject", "message": input("  reason for the model: ").strip() or "Rejected by the reviewer."}
    return {"type": "approve"}


def run(llm: BaseChatModel, review: Review = scripted_review, customer: Customer | None = None) -> dict:
    customer = customer or Customer(name="Ada", plan="team", account_id="acct-1001")
    agent = create_agent(
        llm,
        tools=[open_ticket],
        system_prompt="You are DocsHelp, the Nimbus Notes support assistant. Open tickets when asked.",
        middleware=[
            HumanInTheLoopMiddleware(interrupt_on={"open_ticket": {"allowed_decisions": ["approve", "edit", "reject"]}})
        ],
        checkpointer=InMemorySaver(),  # a paused run lives in the checkpointer until it is resumed
        context_schema=Customer,
    )
    interrupts, replies = [], []
    for case, text in enumerate(CASES):
        header(f"Case {case + 1}")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        print(f"User: {text}")
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config, context=customer)
        if "__interrupt__" not in result:
            print("  (the model answered without proposing a ticket)")
        while "__interrupt__" in result:  # paused: collect a decision for every pending action
            decisions = []
            for action in result["__interrupt__"][0].value["action_requests"]:
                print(f"  PAUSED for review: {action['name']}({action['args']})")
                interrupts.append(action)
                decisions.append(review(action, case))
                print(f"  reviewer: {decisions[-1]['type']}")
            result = agent.invoke(Command(resume={"decisions": decisions}), config, context=customer)
        replies.append(result["messages"][-1].text)
        print(f"DocsHelp: {replies[-1]}")
    return {"interrupts": interrupts, "replies": replies}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", action="store_true", help="review each ticket yourself")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), terminal_review if args.review else scripted_review)


if __name__ == "__main__":
    main()
