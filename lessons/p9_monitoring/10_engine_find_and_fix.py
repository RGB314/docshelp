"""The improvement loop, part 1: find what's failing in production and have an LLM propose the fix.

Academy: LangSmith Engine "What is the ADLC?", "Creating a Recruiting Agent", "Setting Up Engine", "Build: Engine
Proposes a Fix"; Building Reliable Agents "Intro: The Agent Development Lifecycle"; LangSmith Essentials M5
"Closing the Loop".

Concepts:
  * the agent development lifecycle (ADLC): build -> test -> deploy -> monitor -> back to build, driven by what
    production shows. Parts 1-8 built, tested and deployed; Part 9 monitors; now the loop closes.
  * LangSmith Engine automates it: it watches the project's traces and online evaluator scores, detects issues
    (clusters of failures), proposes a fix (a prompt change, or a code change as a pull request), tests it with
    an experiment, and helps you ship and monitor it. ("The agent" here is DocsHelp, not the course's recruiter.)
  * the local twin is `docshelp/engine.py`. Setting it up means what Engine needs too: traces with online
    evaluators on them (9.4), and the prompt in a hub so a fix is a new version, not a code change (Part 5.10).
  * this lesson does steps 1-2:
      1. find failures: errors, unhelpful answers, uncited answers, unhappy customers (`find_failures`)
      2. propose a fix: an LLM reads the failing conversations and the current prompt, diagnoses the pattern
         and writes a new prompt (`PromptFix`), saved in the hub tagged "candidate", *not* "prod"
    Lesson 9.11 tests it, ships it only if it's better, and monitors the result.

Run:  uv run poe lesson 9.10
"""

import difflib
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.engine import propose_candidate
from docshelp.ingest import get_retriever
from docshelp.monitoring import (
    FEEDBACK_LOG,
    TRACE_LOG,
    ensure_feedback,
    ensure_prompt,
    ensure_traffic,
    run_filter,
    summarize,
)
from docshelp.prompts import prompt_hub


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever | None = None,
    *,
    path: Path = TRACE_LOG,
    feedback_path: Path = FEEDBACK_LOG,
    hub=None,
) -> dict:
    hub = hub or prompt_hub()
    ensure_prompt(hub)
    production = run_filter('has(tags, "production")')
    summaries = [s for s in summarize(ensure_traffic(llm, retriever, path=path)) if production(s)]
    feedback = ensure_feedback(llm, summaries, feedback_path=feedback_path)  # the online evaluators of 9.4

    result = propose_candidate(llm, hub, summaries, feedback)
    header(f"1. Failures: {len(result['failures'])} of {len(summaries)} production traces")
    for failure in result["failures"][:10]:
        print(f"  {', '.join(failure['reasons']):40} {failure['question'][:60]}")
    if result["fix"] is None:
        why = f"the model gave no usable proposal ({result['error']})" if result["error"] else "nothing is failing"
        print(f"No fix proposed: {why}. Production is unchanged.")
        return {"failures": len(result["failures"]), "commit": None}

    fix = result["fix"]
    header("2. The proposed fix")
    print(f"Diagnosis: {fix.diagnosis}")
    for change in fix.changes:
        print(f"  + {change}")
    print("\nPrompt diff (prod -> candidate):")
    for line in difflib.unified_diff(result["current"].split(". "), fix.new_prompt.split(". "), lineterm="", n=0):
        if not line.startswith(("---", "+++", "@@")):
            print(f"  {line}")
    print(f"\nSaved as commit {result['commit']}, tagged 'candidate'. Production still uses 'prod'. Next: lesson 9.11.")
    return {
        "failures": len(result["failures"]),
        "reasons": [f["reasons"] for f in result["failures"]],
        "commit": result["commit"],
        "fix": fix,
    }


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
