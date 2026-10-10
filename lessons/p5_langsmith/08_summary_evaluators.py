"""Summary evaluators: metrics over a whole experiment, like precision, recall and F1.

Academy: Intro to Agent Observability & Evaluations M2 L6 "Summary Evaluators".

Concepts:
  * a normal evaluator scores one example. Some metrics only exist across all examples: precision ("of
    the emails we escalated, how many needed it?"), recall ("of the emails that needed a ticket, how many
    did we catch?"), F1, or "pass rate ≥ 90%".
  * a summary evaluator receives every example's `outputs` and `reference_outputs` at once and returns
    one score: `evaluate(..., summary_evaluators=[f1])`. LangSmith shows it at the experiment level.
  * the task here: decide whether a support email needs a ticket (`data/triage_dataset.json`), the
    same decision the capstones made with `needs_ticket`. Missing one is worse than an extra ticket,
    so recall matters most.

Run:  uv run poe lesson 5.8
"""

import json

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from docshelp.config import DATA_DIR, describe, ensure_llm_ready, get_llm, structured_llm
from docshelp.console import header
from docshelp.evals import make_examples, run_experiment

TRIAGE_PATH = DATA_DIR / "triage_dataset.json"


class TicketDecision(BaseModel):
    needs_ticket: bool = Field(
        description="True for bugs, data loss, billing errors or outages; false for how-to questions."
    )


def make_target(llm: BaseChatModel):
    decide = structured_llm(llm, TicketDecision)

    def triage(inputs: dict) -> dict:
        prompt = f"Does this Nimbus Notes support email need a ticket for a human engineer?\n\n{inputs['email']}"
        return {"needs_ticket": decide.invoke(prompt).needs_ticket}

    return triage


def correct(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["needs_ticket"] == reference_outputs["needs_ticket"]


def counts(outputs: list[dict], reference_outputs: list[dict]) -> tuple[int, int, int]:
    pairs = [(o["needs_ticket"], r["needs_ticket"]) for o, r in zip(outputs, reference_outputs)]
    true_pos = sum(1 for got, want in pairs if got and want)
    false_pos = sum(1 for got, want in pairs if got and not want)
    false_neg = sum(1 for got, want in pairs if not got and want)
    return true_pos, false_pos, false_neg


def precision(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    tp, fp, _ = counts(outputs, reference_outputs)
    return {"key": "precision", "score": tp / (tp + fp) if tp + fp else 0.0}


def recall(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    tp, _, fn = counts(outputs, reference_outputs)
    return {"key": "recall", "score": tp / (tp + fn) if tp + fn else 0.0}


def f1(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    p, r = precision(outputs, reference_outputs)["score"], recall(outputs, reference_outputs)["score"]
    return {"key": "f1", "score": 2 * p * r / (p + r) if p + r else 0.0}


def run(llm: BaseChatModel) -> dict:
    items = json.loads(TRIAGE_PATH.read_text(encoding="utf-8"))
    pairs = [({"email": i["email"]}, {"needs_ticket": i["needs_ticket"]}) for i in items]
    examples = make_examples("docshelp-triage", pairs)
    header(f"Triage experiment on {len(examples)} emails")
    experiment = run_experiment(
        make_target(llm), examples, [correct], summary_evaluators=[precision, recall, f1],
        prefix="triage", dataset_name="docshelp-triage",
    )
    for row in experiment["rows"]:
        got, want = row["outputs"].get("needs_ticket"), row["reference"]["needs_ticket"]
        print(f"  {'✓' if got == want else '✗'} ticket={got!s:5} expected={want!s:5}  {row['question'][:60]}")
    header("Summary evaluators (one score for the whole experiment)")
    for key, score in experiment["summary"].items():
        print(f"  {key:9} {score:.0%}")
    return experiment


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm())


if __name__ == "__main__":
    main()
