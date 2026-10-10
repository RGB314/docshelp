"""Experiments: run two versions of DocsHelp over the dataset, score them, and compare.

Academy: Intro to Agent Observability & Evaluations M2 L3 "Experiments", M2 L4 "Analyzing Experiment
Results"; Building Reliable Agents M2 L3 "Running Experiments"; LangSmith Essentials M2 "Experiments".

Concepts:
  * an *experiment* = one target (a version of your app) run over a dataset, with every evaluator's
    score attached to every example: `langsmith.evaluate(target, data=..., evaluators=[...])`
  * comparing experiments is how you decide whether a change helped. Here: the plain RAG chain
    (lesson 1.5) against the support graph (lesson 3.15). Stage 4 used to do this; it's now this lesson.
  * analysing results: averages per metric, then the individual examples where versions disagree or
    fail, with the judge's reasoning. That's where you find what to fix.
  * `metadata` (model, prompt version) records what was tested. `num_repetitions` reruns each example
    to average out model randomness. `max_concurrency` runs examples in parallel.
  * with LangSmith the experiments are uploaded under the dataset (select two → Compare); without it,
    `evaluate(..., upload_results=False)` runs the same thing locally

Run:  uv run poe lesson 5.6               (all 9 examples)
      uv run poe lesson 5.6 --split edge  (only the edge cases)
"""

import argparse

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm, langsmith_enabled
from docshelp.console import header
from docshelp.evals import (
    cites_source,
    compare,
    expected_source,
    load_rows,
    make_correctness_judge,
    make_graph_target,
    make_rag_target,
    run_experiment,
)
from docshelp.ingest import get_retriever


def run(llm: BaseChatModel, retriever: BaseRetriever, split: str | None = None) -> dict:
    rows = load_rows(split)
    evaluators = [expected_source, cites_source, make_correctness_judge(llm)]
    targets = {"rag-chain": make_rag_target(llm, retriever), "support-graph": make_graph_target(llm, retriever)}
    experiments = {}
    for name, target in targets.items():
        header(f"Experiment '{name}' on {len(rows)} examples")
        experiments[name] = run_experiment(target, rows, evaluators, prefix=name, metadata={"split": split or "all"})
        for row in experiments[name]["rows"]:
            mark = "✓" if row["scores"].get("correctness") else "✗"
            print(f"  {mark} {row['question']}")

    header("Comparison (average score per evaluator)")
    print(compare(experiments))

    header("Analysis: questions where the two versions disagree on correctness")
    a, b = experiments.values()
    disagreements = [
        (ra["question"], ra["scores"].get("correctness"), rb["scores"].get("correctness"))
        for ra, rb in zip(a["rows"], b["rows"])
        if ra["scores"].get("correctness") != rb["scores"].get("correctness")
    ]
    for question, score_a, score_b in disagreements:
        print(f"  {question}\n    rag-chain={score_a}  support-graph={score_b}")
    if not disagreements:
        print("  (none: both versions got the same questions right and wrong)")
    if langsmith_enabled():
        print("\nIn LangSmith: Datasets & Experiments → docshelp-qa → select both experiments → Compare.")
    return {"experiments": experiments, "disagreements": disagreements}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["base", "edge"], help="evaluate only one split")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()} for the app and the judge")
    run(get_llm(), get_retriever(), args.split)


if __name__ == "__main__":
    main()
