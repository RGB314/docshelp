"""Capstone: the full deep-research system, from a vague request to a cited report, then graded.

Academy: Project: Deep Research M6 "Multi-Agent Research System".

Brings Part 6's research lessons together in `src/docshelp/deep_research.py::build_deep_research`:
  * scope (6.5): clarify if needed (the run ends with a question; the answer continues the thread), then
    write the brief
  * research (6.8): the supervisor subgraph runs researchers (6.6) in parallel until the brief is covered
  * report: one writer turns the compressed findings into a Markdown report with a Sources section,
    saved to `data/reports/`
  * evaluate (Part 5): an LLM judge scores the report against a rubric (covers the brief, cites sources,
    no unsupported claims). Run it after every change to the prompts or limits.

Run:  uv run poe lesson 6.9               (offline sources)
      uv run poe lesson 6.9 --web         (plus web search)
"""

import argparse
import asyncio
import re
import uuid
from collections.abc import Callable
from datetime import date
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from openevals.llm import create_llm_as_judge

from docshelp.config import DATA_DIR, describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep_research import build_deep_research, make_research_tools
from docshelp.ingest import get_retriever

REQUEST = "I need a research report on Nimbus Notes for new support hires: what customers most often struggle with."
REPORTS_DIR = DATA_DIR / "reports"
RUBRIC = """Grade this research report against its brief.

<brief>
{inputs}
</brief>

<report>
{outputs}
</report>

A good report answers every part of the brief, cites a [source] for its facts, has a clear structure,
and makes no claims the sources don't support. Score from 0 (useless) to 1 (excellent)."""


def scripted_user(question: str) -> str:
    return "Cover sync problems, billing questions and sharing permissions, with the fix for each. One page."


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    answer: Callable[[str], str] = scripted_user,
    *,
    web: bool = False,
    reports_dir: Path | None = None,
) -> dict:
    graph = build_deep_research(llm, make_research_tools(retriever, web=web), InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    header(f"User: {REQUEST}")
    # The supervisor runs researchers with asyncio, so the whole graph runs with ainvoke.
    state = asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": REQUEST}]}, config))
    for _ in range(3):  # scoping may ask a few clarifying questions
        if state.get("final_report"):
            break
        question = state["messages"][-1].text
        reply = answer(question)
        print(f"Scoper: {question}\nUser:   {reply}")
        state = asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": reply}]}, config))

    if not state.get("final_report"):
        print("Scoping kept asking questions; answer them interactively or give a more specific request.")
        return {"brief": "", "report": "", "path": None, "grade": None}
    report = state["final_report"]
    reports_dir = reports_dir or REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", state["research_brief"].lower())[:40].strip("-") or "report"
    path = reports_dir / f"{date.today()}-{slug}.md"
    path.write_text(report, encoding="utf-8")
    header(f"Report ({len(state.get('notes', []))} sets of findings) saved to {path.name}")
    print(report)

    header("Report quality (LLM judge against the rubric)")
    judge = create_llm_as_judge(prompt=RUBRIC, judge=llm, feedback_key="report_quality", continuous=True)
    grade = judge(inputs=state["research_brief"], outputs=report)
    print(f"  report_quality = {grade['score']}  {grade.get('comment') or ''}"[:300])
    return {"brief": state["research_brief"], "report": report, "path": path, "grade": grade}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", action="store_true", help="also search the web (needs `uv sync --extra web`)")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever(), web=args.web)


if __name__ == "__main__":
    main()
