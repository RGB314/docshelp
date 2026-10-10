"""Annotation queues: route doubtful answers to a human reviewer, and turn corrections into test data.

Academy: Intro to Agent Observability & Evaluations M4 L2 "Annotation Queues".

Concepts:
  * an annotation queue is a to-do list of runs for reviewers (support leads, domain experts). Each item
    shows the trace; the reviewer scores it against a *rubric* and can write a corrected answer.
  * filling the queue: a rule picks which runs need eyes. Here, answers that cite no document
    (the `cites_source` check from lesson 5.5); in production also thumbs-down feedback (5.11) or low
    judge scores. Part 9 automates these rules.
  * closing the loop: a reviewed answer, with its correction, becomes a new dataset example (5.4), so
    the next experiment (5.6) tests exactly the case that went wrong
  * with LangSmith: `create_annotation_queue(name=..., rubric_instructions=...)`,
    `add_runs_to_annotation_queue(queue_id, run_ids=[...])`, then review in the UI, then
    `create_example(..., source_run_id=...)`. The local twin keeps the queue in `data/annotation_queue.jsonl`.

Run:  uv run poe lesson 5.12               (scripted reviewer)
      uv run poe lesson 5.12 --review      (you review)
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langsmith import Client

from docshelp.config import DATA_DIR, describe, ensure_llm_ready, get_llm, langsmith_enabled
from docshelp.console import header
from docshelp.evals import DATASET_NAME, cites_source, make_rag_target
from docshelp.ingest import get_retriever
from docshelp.observability import record_feedback, traced

QUEUE_PATH = DATA_DIR / "annotation_queue.jsonl"
RUBRIC = (
    "correct: is every fact right? (y/n). helpful: 1-5. "
    "If not correct, write the answer DocsHelp should have given."
)
QUESTIONS = [
    "How long do deleted notes stay in the trash?",
    "Can I get my data hosted in Japan?",
    "Which plan includes version history?",
]


def scripted_reviewer(item: dict) -> dict:
    """A support lead's review: uncited answers get corrected (stand-in for a human)."""
    return {"correct": False, "helpful": 2, "correction": f"(reviewed answer for: {item['question']})"}


def terminal_reviewer(item: dict) -> dict:
    print(f"  Rubric: {RUBRIC}")
    correct = input("  correct? [y/n] ").strip().lower().startswith("y")
    helpful = int(input("  helpful 1-5: ").strip() or 3)
    correction = None if correct else input("  corrected answer: ").strip()
    return {"correct": correct, "helpful": helpful, "correction": correction}


def run(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    reviewer: Callable[[dict], dict] = scripted_reviewer,
    queue_path: Path | None = None,
) -> dict:
    queue_path = queue_path or QUEUE_PATH
    answer = make_rag_target(llm, retriever)
    header("1. Answer questions; a rule sends uncited answers to the queue")
    queue = []
    for question in QUESTIONS:
        with traced("support_request", {"question": question}) as run:
            outputs = answer({"question": question})
            run.end(outputs=outputs)
        cited = cites_source(outputs)["score"]
        print(f"  {'cited ' if cited else 'QUEUED'}  {question}")
        if not cited:
            queue.append({"run_id": str(run.id), "question": question, "answer": outputs["answer"],
                          "reason": "no source cited"})
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("a", encoding="utf-8") as f:
        for item in queue:
            f.write(json.dumps(item) + "\n")
    if langsmith_enabled() and queue:
        client = Client()
        q = client.create_annotation_queue(name="docshelp-review", rubric_instructions=RUBRIC)
        client.add_runs_to_annotation_queue(q.id, run_ids=[item["run_id"] for item in queue])
        print("  (also queued in LangSmith → Annotation queues → docshelp-review)")

    header(f"2. Review {len(queue)} queued answers")
    new_examples = []
    for item in queue:
        print(f"Q: {item['question']}\nA: {item['answer']}")
        review = reviewer(item)
        record_feedback(item["run_id"], "reviewer_correct", review["correct"])
        record_feedback(item["run_id"], "reviewer_helpful", review["helpful"])
        if review.get("correction"):
            new_examples.append(
                {"question": item["question"], "answer": review["correction"], "source_run_id": item["run_id"]}
            )

    header("3. Corrections become dataset examples")
    for example in new_examples:
        print(f"  + {example['question']}  ->  {example['answer']}")
        if langsmith_enabled():
            Client().create_example(
                inputs={"question": example["question"]},
                outputs={"answer": example["answer"]},
                dataset_name=DATASET_NAME,
                split="reviewed",
                source_run_id=example["source_run_id"],  # links the example to the trace it came from
            )
    if new_examples and not langsmith_enabled():
        print("  (with LangSmith they join the 'reviewed' split; locally, add them to data/eval_dataset.json)")
    return {"queue": queue, "new_examples": new_examples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", action="store_true", help="review the queued answers yourself")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever(), terminal_reviewer if args.review else scripted_reviewer)


if __name__ == "__main__":
    main()
