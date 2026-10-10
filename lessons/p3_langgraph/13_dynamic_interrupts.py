"""Dynamic interrupts: pause from inside a node, only when the data says so.

Academy: Intro to LangGraph M3 L4 "Dynamic Breakpoints"; LangGraph Essentials L5 "Interrupt, Human in
the Loop".

Concepts:
  * `interrupt(payload)` inside a node saves state and stops the run. The caller sees the payload in
    `result["__interrupt__"]` and resumes with `Command(resume=value)`; `interrupt()` then *returns*
    that value. (Breakpoints in lesson 3.12 always stop; this stops only when needed.)
  * on resume the node runs again from its start, and earlier `interrupt()` calls in it return their
    saved answers in order. So keep side effects after the interrupts, or in a separate node.
  * a validation loop: keep asking until the answer is valid (an unknown account id is asked again)
  * a conditional approval: refunds over $100 need a manager; small ones don't pause at all
  * `HumanInTheLoopMiddleware` (lesson 2.7) is built on exactly this `interrupt()`

Run:  uv run poe lesson 3.13                 (scripted human)
      uv run poe lesson 3.13 --interactive   (you answer)
"""

import argparse
import uuid
from collections.abc import Callable
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from docshelp.config import describe, ensure_llm_ready, get_llm, structured_llm
from docshelp.console import header
from docshelp.customers import ACCOUNTS

APPROVAL_LIMIT = 100.0
EMAIL = "Hello, you charged our card twice this month, $240 in total. Please refund the duplicate. - Ada"


class RefundRequest(BaseModel):
    amount: float = Field(description="Refund amount requested, in dollars.")
    reason: str = Field(description="Why the customer wants a refund, in a few words.")
    account_id: str | None = Field(default=None, description="Account id like acct-1234, if mentioned.")


class RefundState(TypedDict, total=False):
    email: str
    amount: float
    reason: str
    account_id: str | None
    status: str


def build(llm: BaseChatModel):
    extractor = structured_llm(llm, RefundRequest)

    def extract(state: RefundState) -> dict:
        return extractor.invoke(f"Extract the refund request from this email:\n{state['email']}").model_dump()

    def ensure_account(state: RefundState) -> dict:
        account_id = state.get("account_id")
        while account_id not in ACCOUNTS:  # ask until the answer is a known account
            account_id = interrupt({"question": "Which account is this refund for?", "got": account_id})
            account_id = str(account_id).strip()
        return {"account_id": account_id}

    def approve(state: RefundState) -> dict:
        if state["amount"] <= APPROVAL_LIMIT:
            return {"status": "refund approved (automatic, under the limit)"}
        answer = interrupt(
            {"question": f"Approve a ${state['amount']:.2f} refund for {state['account_id']} ({state['reason']})?"}
        )
        return {"status": "refund approved" if str(answer).lower().startswith("y") else "refund denied"}

    builder = StateGraph(RefundState)
    builder.add_node(extract)
    builder.add_node(ensure_account)
    builder.add_node(approve)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", "ensure_account")
    builder.add_edge("ensure_account", "approve")
    builder.add_edge("approve", END)
    return builder.compile(checkpointer=InMemorySaver())


def scripted_human() -> Callable[[dict], str]:
    account_answers = iter(["acct-9999", "acct-1001"])  # a typo first, then the right id

    def answer(payload: dict) -> str:
        return next(account_answers) if "account" in payload["question"] else "yes"

    return answer


def terminal_human(payload: dict) -> str:
    return input(f"  {payload['question']} > ")


def run(llm: BaseChatModel, human: Callable[[dict], str] | None = None) -> dict:
    human = human or scripted_human()
    graph = build(llm)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    header("Refund request")
    print(EMAIL)
    result = graph.invoke({"email": EMAIL}, config)
    interrupts = []
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        interrupts.append(payload)
        answer = human(payload)
        print(f"  PAUSED: {payload['question']}  -> human answers {answer!r}")
        result = graph.invoke(Command(resume=answer), config)
    header("Result")
    print(f"  {result['status']} (account {result['account_id']}, ${result['amount']:.2f}, {result['reason']})")
    return {"interrupts": interrupts, "state": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="answer the questions yourself")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), terminal_human if args.interactive else None)


if __name__ == "__main__":
    main()
