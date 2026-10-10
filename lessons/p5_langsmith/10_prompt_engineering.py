"""The prompt engineering lifecycle: version a prompt, evaluate a change, promote it with a tag.

Academy: Intro to Agent Observability & Evaluations M3 L1 "Playground", M3 L2 "Prompt Hub",
M3 L3 "Prompt Engineering Lifecycle", M3 L4 "Polly".

Concepts:
  * prompts are versioned artifacts. Push each version to a prompt hub (`src/docshelp/prompts.py`:
    LangSmith's Prompt Hub, or the local twin) and pull it by name, tag or commit: `docshelp-rag:prod`.
  * the lifecycle: draft a change → evaluate it against the dataset (lesson 5.6) → promote the winner by
    moving the "prod" tag → keep the old commit for rollback. No code change, no redeploy.
  * Playground (LangSmith UI): open a prompt, run it on dataset examples with different models and
    settings side by side, then save the edit as a new commit. Polly, LangSmith's built-in assistant,
    can suggest prompt edits and explain traces there. Both work on the commits this lesson pushes.
  * the prompt here is the RAG prompt from lesson 1.5; v2 asks for citations, and the `cites_source`
    evaluator measures whether that worked

Run:  uv run poe lesson 5.10
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.evals import averages, cites_source, compare, expected_source, load_rows, make_rag_target, run_experiment
from docshelp.ingest import get_retriever
from docshelp.prompts import prompt_hub

NAME = "docshelp-rag"
V1 = ChatPromptTemplate.from_messages(
    [("system", "Answer using only this Nimbus Notes documentation. Be concise.\n\n{context}"), ("human", "{question}")]
)
V2 = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using only this Nimbus Notes documentation. Be concise. End with the source file "
                   "name in square brackets, e.g. [07-troubleshooting.md].\n\n{context}"),
        ("human", "{question}"),
    ]
)


def run(llm: BaseChatModel, retriever: BaseRetriever, split: str | None = "edge", hub_root: Path | None = None) -> dict:
    hub = prompt_hub(hub_root)
    header("1. Push two versions of the prompt")
    v1 = hub.push(NAME, V1, tags=["prod"], description="first version")
    v2 = hub.push(NAME, V2, description="ask for a citation")
    print(f"  v1 = {v1} (tagged prod), v2 = {v2}")

    header("2. Evaluate both on the dataset")
    rows = load_rows(split)
    experiments = {}
    for label, commit in (("v1", v1), ("v2", v2)):
        prompt = hub.pull(f"{NAME}:{commit}")  # always run what's in the hub, not a local copy
        target = make_rag_target(llm, retriever, prompt)
        experiments[label] = run_experiment(target, rows, [cites_source, expected_source], prefix=f"rag-prompt-{label}",
                                            metadata={"prompt_commit": commit})
    print(compare(experiments))

    header("3. Promote the winner")
    winner = max(("v1", "v2"), key=lambda label: averages(experiments[label]).get("cites_source", 0))
    winner_commit = {"v1": v1, "v2": v2}[winner]
    if winner_commit != v1:
        hub.tag(NAME, winner_commit, "prod")
    print(f"  prod -> {winner} ({winner_commit}). Rolling back is the same call with the old commit.")
    for entry in hub.history(NAME):
        print(f"  {entry['commit']}  {entry['description']:22} {', '.join(entry['tags'])}")
    prod = hub.pull(f"{NAME}:prod")
    return {"winner": winner, "prod_prompt": prod, "experiments": experiments}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever(), split=None)


if __name__ == "__main__":
    main()
