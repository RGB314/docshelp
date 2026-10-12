"""Insights: what are customers asking about, and where does the agent struggle? (lesson 9.8)

LangSmith Insights runs an agent over your traces: it summarises each conversation, then groups them into a
hierarchy of categories with counts and examples. This local twin does the same in two LLM steps over the
trace log:

  1. `summarize_conversation`: each thread becomes a `ConversationSummary` (topic, outcome, sentiment)
  2. `cluster`: all summaries together become a few named `Category`s, each listing its threads

So instead of reading hundreds of traces, you read five categories and decide what to fix first.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from docshelp.config import structured_llm
from docshelp.monitoring import answer_of, question_of


class ConversationSummary(BaseModel):
    topic: str = Field(description="What the customer wanted, in at most eight words.")
    outcome: Literal["resolved", "unresolved", "refused"] = Field(description="How the conversation ended.")
    sentiment: Literal["negative", "neutral", "positive"] = Field(description="The customer's mood overall.")


class Category(BaseModel):
    name: str = Field(description="A short category name, e.g. 'Sync failures'.")
    description: str = Field(description="One sentence: what these conversations have in common.")
    threads: list[str] = Field(description="The ids of the threads in this category.")


class Categories(BaseModel):
    categories: list[Category] = Field(description="Three to six categories; every thread in exactly one.")


def threads_of(summaries: list[dict]) -> dict[str, list[dict]]:
    """Traces grouped into conversations by their `thread_id` metadata, in order."""
    threads: dict[str, list[dict]] = {}
    for summary in summaries:
        threads.setdefault(summary["metadata"].get("thread_id", summary["trace_id"]), []).append(summary)
    return threads


def transcript(turns: list[dict]) -> str:
    return "\n".join(f"Customer: {question_of(t)}\nAgent: {answer_of(t)}" for t in turns)


def summarize_conversation(llm: BaseChatModel, turns: list[dict]) -> ConversationSummary:
    return structured_llm(llm, ConversationSummary).invoke(
        f"Summarise this support conversation for a product team.\n\n{transcript(turns)}"
    )


def cluster(llm: BaseChatModel, summaries: dict[str, ConversationSummary]) -> Categories:
    listing = "\n".join(
        f"- {thread}: {s.topic} (outcome: {s.outcome}, sentiment: {s.sentiment})" for thread, s in summaries.items()
    )
    return structured_llm(llm, Categories).invoke(
        "Group these support conversations into a few categories a product team would act on. "
        f"Use the thread ids exactly as given.\n\n{listing}"
    )


def insights(llm: BaseChatModel, summaries: list[dict]) -> dict:
    """The whole report: per-thread summaries, and categories with counts, most common first."""
    per_thread = {}
    for thread, turns in threads_of(summaries).items():
        try:
            per_thread[thread] = summarize_conversation(llm, turns)
        except Exception as err:  # noqa: BLE001 - small models sometimes return invalid structure: skip that one
            print(f"  (couldn't summarise {thread}: {type(err).__name__})")
    try:
        categories = cluster(llm, per_thread).categories if per_thread else []
    except Exception as err:  # noqa: BLE001 - report the summaries even if grouping them failed
        print(f"  (couldn't group the conversations: {type(err).__name__})")
        categories = []
    known = set(per_thread)
    report = []
    for category in categories:
        members = [t for t in category.threads if t in known]  # models sometimes invent ids: drop those
        unresolved = sum(per_thread[t].outcome != "resolved" for t in members)
        report.append(
            {
                "name": category.name,
                "description": category.description,
                "threads": members,
                "count": len(members),
                "unresolved": unresolved,
            }
        )
    return {"threads": per_thread, "categories": sorted(report, key=lambda c: -c["count"])}
