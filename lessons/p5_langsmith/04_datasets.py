"""Datasets: the questions (and good answers) you test every version of the agent against.

Academy: Intro to Agent Observability & Evaluations M2 L1 "Datasets"; Building Reliable Agents M2 L1
"Evaluating Agents", M2 L2 "Creating Datasets"; LangSmith Essentials M2 "Datasets and Offline Evals".

Concepts:
  * an *example* = inputs (the question) + reference outputs (a good answer, the doc that holds it) +
    metadata. `data/eval_dataset.json` holds DocsHelp's examples; `src/docshelp/evals.py` loads them.
  * splits group examples (here "base" and "edge" cases) so you can evaluate a subset
  * versions: LangSmith versions a dataset on every change, and you can tag one ("v1") and evaluate
    against it later. Locally, a content hash plays that role.
  * where examples come from: hand-written by experts, taken from real traces (then corrected by a
    reviewer, lesson 5.12), or *synthetic*: an LLM writes variations of real questions (below)
  * with LangSmith: `create_dataset`, `create_examples(..., split=...)`, `list_examples(splits=[...])`,
    `update_dataset_tag(...)`. Without it, the same examples become in-memory `Example` objects.

Run:  uv run poe lesson 5.4
"""

from collections import Counter
from datetime import datetime, timezone

from langchain_core.language_models import BaseChatModel
from langsmith import Client

from docshelp.config import describe, ensure_llm_ready, get_llm, langsmith_enabled
from docshelp.console import header
from docshelp.evals import DATASET_NAME, ensure_dataset, load_rows, to_examples, version_of


def paraphrase(llm: BaseChatModel, question: str) -> str:
    return llm.invoke(
        f"Rewrite this customer support question the way a different, less technical customer might ask it. "
        f"Reply with the question only.\n\n{question}"
    ).text.strip()


def run(llm: BaseChatModel, synthetic: int = 2) -> dict:
    rows = load_rows()
    header(f"Dataset '{DATASET_NAME}': {len(rows)} examples, version {version_of(rows)}")
    for split, count in Counter(r["split"] for r in rows).items():
        print(f"  split {split!r}: {count} examples")
    example = to_examples(rows[:1])[0]
    print(f"\nOne example:\n  inputs:  {example.inputs}\n  outputs: {example.outputs}\n  metadata: {example.metadata}")
    print(f"\nEdge cases only: {[r['question'] for r in load_rows('edge')]}")

    header("Synthetic examples: an LLM rephrases real questions (review before adding them!)")
    candidates = []
    for row in rows[:synthetic]:
        question = paraphrase(llm, row["question"])
        candidates.append({**row, "question": question, "split": "synthetic"})
        print(f"  {row['question']}\n    -> {question}")
    print(f"Adding them would create version {version_of(rows + candidates)}.")

    if langsmith_enabled():
        client = Client()
        ensure_dataset(client, to_examples(rows), DATASET_NAME)
        client.update_dataset_tag(dataset_name=DATASET_NAME, as_of=datetime.now(timezone.utc), tag="v1")
        edge = list(client.list_examples(dataset_name=DATASET_NAME, splits=["edge"]))
        print(f"\nIn LangSmith: '{DATASET_NAME}' tagged v1, {len(edge)} edge-case examples.")
        print("See Datasets & Experiments in the LangSmith UI.")
    else:
        print("\n(LangSmith not configured: the dataset stays local. Lessons 5.6-5.8 evaluate against it.)")
    return {"rows": rows, "version": version_of(rows), "candidates": candidates}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
