"""The agent development lifecycle (ADLC), automated: a local twin of LangSmith Engine (lessons 9.10-9.11).

LangSmith Engine watches production traces and closes the loop on its own:

  1. **Find** failing traces (online evaluator scores, errors, user feedback)    `find_failures`
  2. **Build** a fix: an LLM diagnoses the pattern and proposes a new prompt     `propose_fix`
  3. **Test** it: an experiment, old prompt vs new, on the failures plus a       `prompt_target`,
     regression set, with the same evaluators, and a pairwise judge             `ship_decision`
  4. **Deploy** it: a new version in the prompt hub, tagged "prod" only if the
     test passed (the agent pulls "docshelp-support:prod")
  5. **Monitor**: replay traffic with the shipped prompt and compare the online metrics

Each step reuses an earlier part: Part 5's experiments, judges and prompt hub, and Part 9's online evaluators.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from docshelp.config import structured_llm
from docshelp.customers import Customer
from docshelp.monitoring import (
    PROMPT_NAME,
    answer_of,
    build_monitored_agent,
    feedback_by_trace,
    prompt_template,
    prompt_text,
    question_of,
)

FAILURE_CHECKS = {  # feedback key -> the score below which a trace counts as failing
    "helpfulness": 1.0,
    "cites_docs": 1.0,
    "user_sentiment": 0.0,
}


def find_failures(summaries: list[dict], feedback: list[dict]) -> list[dict]:
    """Traces that errored or scored badly, each with the reasons: Engine's "issues" list."""
    scores = feedback_by_trace(feedback)
    failures = []
    for summary in summaries:
        reasons = [f"error: {summary['error']}"] if summary["status"] == "error" else []
        for key, minimum in FAILURE_CHECKS.items():
            score = scores.get(summary["trace_id"], {}).get(key)
            if score is not None and score < minimum:
                reasons.append(f"{key}={score:g}")
        if reasons:
            failures.append(
                {
                    "trace_id": summary["trace_id"],
                    "question": question_of(summary),
                    "answer": answer_of(summary),
                    "reasons": reasons,
                }
            )
    return failures


class PromptFix(BaseModel):
    """A proposed change to the system prompt, with the evidence behind it."""

    diagnosis: str = Field(description="The pattern behind the failures, in two sentences at most.")
    changes: list[str] = Field(description="Each instruction you add or change, one per item.")
    new_prompt: str = Field(description="The complete new system prompt.")


def propose_fix(llm: BaseChatModel, current_prompt: str, failures: list[dict], *, limit: int = 8) -> PromptFix:
    cases = "\n\n".join(
        f"Customer: {f['question']}\nAgent: {f['answer']}\nProblems: {', '.join(f['reasons'])}"
        for f in failures[:limit]
    )
    return structured_llm(llm, PromptFix).invoke(
        "You improve the system prompt of a customer support agent for Nimbus Notes. Its tools: search_docs (the "
        "product docs, cite the file names), quote_price, my_account. Fix the patterns behind these failures; keep "
        "what works.\n\n"
        f"Current prompt:\n{current_prompt}\n\nFailing conversations:\n{cases}"
    )


def propose_candidate(llm: BaseChatModel, hub, summaries: list[dict], feedback: list[dict]) -> dict:
    """Steps 1-2: find failures, propose a fix, and save it in the prompt hub as "candidate" (not "prod")."""
    current = prompt_text(hub)
    failures = find_failures(summaries, feedback)
    if not failures:
        return {"failures": [], "fix": None, "commit": None, "current": current, "error": None}
    try:
        fix = propose_fix(llm, current, failures)
    except Exception as err:  # noqa: BLE001 - no usable proposal (e.g. invalid structured output): change nothing
        return {"failures": failures, "fix": None, "commit": None, "current": current, "error": f"{type(err).__name__}"}
    commit = hub.push(
        PROMPT_NAME, prompt_template(fix.new_prompt), tags=["candidate"], description=f"candidate: {fix.diagnosis[:80]}"
    )
    return {"failures": failures, "fix": fix, "commit": commit, "current": current, "error": None}


def prompt_target(llm: BaseChatModel, retriever: BaseRetriever | None, system_prompt: str) -> Callable[[dict], dict]:
    """An experiment target: the support agent with a given prompt, one fresh conversation per example."""
    agent = build_monitored_agent(llm, retriever, system_prompt=system_prompt)
    counter = iter(range(1_000_000))

    def target(inputs: dict) -> dict:
        config = {"configurable": {"thread_id": f"experiment-{next(counter)}"}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": inputs["question"]}]}, config, context=Customer()
        )
        return {"answer": result["messages"][-1].text}

    return target


def as_offline_evaluator(online: Callable[[dict], dict]) -> Callable[[dict, dict], dict]:
    """Reuse an online evaluator (it reads a trace summary) in an offline experiment (inputs, outputs)."""

    def evaluator(inputs: dict, outputs: dict) -> dict:
        summary = {"inputs": json.dumps(inputs), "outputs": json.dumps(outputs)}
        result = online(summary)
        return {"key": result["key"], "score": result["score"], "comment": result["comment"]}

    evaluator.__name__ = getattr(online, "__name__", "evaluator")
    return evaluator


def ship_decision(old: dict[str, float], new: dict[str, float], wins: dict[str, int]) -> tuple[bool, str]:
    """Ship only if the candidate is no worse on every metric and the pairwise judge prefers it more often."""
    worse = [k for k in old if new.get(k, 0.0) < old[k]]
    if worse:
        return False, f"regressed on {', '.join(worse)}"
    if wins.get("new", 0) <= wins.get("old", 0):
        return False, f"pairwise judge didn't prefer it ({wins})"
    return True, f"no regressions; pairwise {wins}"
