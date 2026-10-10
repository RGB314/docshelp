"""User feedback: attach thumbs up / down (and comments) from customers to the exact trace they rate.

Academy: Intro to Agent Observability & Evaluations M4 L1 "User Feedback".

Concepts:
  * feedback = a key ("user_score"), a score, an optional comment, attached to a *run id*. Because it
    points at a trace, a thumbs-down leads straight to the model calls and documents behind that answer.
  * keep the run id of each response: `traced(...)` returns the root run, and `run.id` is what the
    front end sends back with the rating
  * `record_feedback` (`src/docshelp/observability.py`) calls `Client().create_feedback(run_id, key=...,
    score=..., comment=...)` with LangSmith, and always writes `data/feedback.jsonl` (the local twin)
  * browsers shouldn't hold your API key: `Client().create_presigned_feedback_token(run_id, "user_score")`
    gives the front end a one-off URL to post the rating to
  * feedback isn't only from users: evaluators (5.5) and reviewers (5.12) write feedback too, under
    their own keys

Run:  uv run poe lesson 5.11
"""

import uuid
from collections.abc import Callable
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver

from docshelp.agents import Customer, build_support_agent
from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.observability import FEEDBACK_PATH, load_rows, record_feedback, traced

QUESTIONS = [
    "How do I share a notebook with someone outside my workspace?",
    "Why does my tray icon show a red slash after my laptop wakes up?",
    "What's the keyboard shortcut for a new note?",
]


def scripted_customer(question: str, answer: str) -> tuple[int, str | None]:
    """A customer who dislikes answers that don't mention a doc (a stand-in for real ratings)."""
    return (1, None) if ".md" in answer else (0, "Didn't really answer my question.")


def terminal_customer(question: str, answer: str) -> tuple[int, str | None]:
    thumbs = input("  Helpful? [y/n] ").strip().lower().startswith("y")
    return (1, None) if thumbs else (0, input("  What was wrong? ").strip() or None)


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    rate: Callable[[str, str], tuple[int, str | None]] = scripted_customer,
    feedback_path: Path | None = None,
) -> dict:
    agent = build_support_agent(llm, retriever, InMemorySaver(), structured=False)
    customer = Customer(name="Ada", plan="team", account_id="acct-1001")
    records = []
    for question in QUESTIONS:
        header(f"User: {question}")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        with traced("support_request", {"question": question}, metadata={"plan": customer.plan}) as run:
            answer = agent.invoke({"messages": [{"role": "user", "content": question}]}, config, context=customer)
            answer = answer["messages"][-1].text
            run.end(outputs={"answer": answer})
        print(f"DocsHelp: {answer}")
        score, comment = rate(question, answer)
        records.append(record_feedback(run.id, "user_score", score, comment, path=feedback_path))
        print(f"  feedback on run {str(run.id)[:8]}: user_score={score}" + (f" ({comment})" if comment else ""))

    header("Feedback summary")
    scores = [r["score"] for r in records]
    print(f"  user_score: {sum(scores) / len(scores):.0%} positive over {len(scores)} responses")
    negative = [r for r in records if r["score"] == 0]
    for r in negative:
        print(f"  👎 run {r['run_id'][:8]}: {r['comment']}")
    stored = load_rows(feedback_path or FEEDBACK_PATH)
    log = (feedback_path or FEEDBACK_PATH).name
    print(f"  ({len(stored)} feedback records in {log}; lesson 5.12 reviews the 👎 ones)")
    return {"feedback": records, "negative": negative}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", action="store_true", help="rate each answer yourself")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever(), terminal_customer if args.rate else scripted_customer)


if __name__ == "__main__":
    main()
