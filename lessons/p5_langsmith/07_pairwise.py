"""Pairwise evaluation: show a judge two answers side by side and ask which is better.

Academy: Intro to Agent Observability & Evaluations M2 L5 "Pairwise Experiments"; Building Reliable
Agents M2 L6 "Pairwise Evaluations".

Concepts:
  * scoring each answer alone ("is this correct?") misses quality differences when both are correct.
    Comparing two answers to the same question ("which one helps more?") is easier and more consistent,
    for humans and for LLM judges.
  * position bias: judges tend to prefer whichever answer they read first, so the pair is shown in
    alternating order (`swap`) and the verdict mapped back
  * with LangSmith: run both experiments, then `evaluate((experiment_a, experiment_b), evaluators=[...])`.
    A comparative evaluator receives both runs for an example and returns a score per run, and the UI
    shows the preference next to the two experiments. Without LangSmith, the same judge runs locally.

Run:  uv run poe lesson 5.7               (all 9 examples)
      uv run poe lesson 5.7 --split edge
"""

import argparse
from collections import Counter

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langsmith import evaluate

from docshelp.config import describe, ensure_llm_ready, get_llm, langsmith_enabled
from docshelp.console import header
from docshelp.evals import load_rows, make_graph_target, make_rag_target, pairwise_judge, run_experiment
from docshelp.ingest import get_retriever


def run(llm: BaseChatModel, retriever: BaseRetriever, split: str | None = None) -> dict:
    rows = load_rows(split)
    names = ("rag-chain", "support-graph")
    header("Run both versions over the dataset (no scoring yet)")
    a = run_experiment(make_rag_target(llm, retriever), rows, [], prefix=names[0])
    b = run_experiment(make_graph_target(llm, retriever), rows, [], prefix=names[1])

    header("Pairwise judge: which answer helps the customer more?")
    prefer = pairwise_judge(llm)
    tally: Counter = Counter()
    for n, (ra, rb) in enumerate(zip(a["rows"], b["rows"])):
        answer_a, answer_b = ra["outputs"].get("answer", ""), rb["outputs"].get("answer", "")
        winner, reason = prefer(ra["question"], answer_a, answer_b, swap=n % 2 == 1)  # swap every other pair
        label = {"first": names[0], "second": names[1]}.get(winner, "tie")
        tally[label] += 1
        print(f"  {ra['question']}\n    -> {label}: {reason[:120]}")

    header("Preference tally")
    for label in (*names, "tie"):
        print(f"  {label:14} {tally[label]}")

    if langsmith_enabled():
        def preferred(runs, example) -> dict:  # a comparative evaluator: both runs for one example
            winner, reason = prefer(
                example.inputs["question"], runs[0].outputs["answer"], runs[1].outputs["answer"], swap=False
            )
            return {"key": "preferred", "scores": {runs[0].id: winner == "first", runs[1].id: winner == "second"},
                    "comment": reason}

        evaluate((a["name"], b["name"]), evaluators=[preferred], randomize_order=True)
        print("\nIn LangSmith: open the dataset's Pairwise Experiments tab to see each preference.")
    return {"tally": dict(tally)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["base", "edge"], help="compare on one split only")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()} for the app and the judge")
    run(get_llm(), get_retriever(), args.split)


if __name__ == "__main__":
    main()
