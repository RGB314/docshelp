"""Stage 4 - LangSmith: tracing, datasets and evaluation.

What LangSmith gives you here:
  * tracing: with LANGSMITH_TRACING=true every LangChain/LangGraph call in *every* stage is recorded
    automatically (inputs, outputs, latency, tokens, cost, and each nested step). `@traceable` adds
    your own plain-Python functions to the trace tree.
  * datasets: versioned input / reference-output examples, stored in LangSmith
  * evaluation: run an app over a dataset, score it with evaluators (here an LLM-as-judge plus a
    code check), and compare experiments side by side in the UI

This script compares the simple RAG chain from stage 2 against the LangGraph agent from stage 3
on the same questions.

Without a LangSmith key it still runs the same evaluation locally and prints the scores.

Run:  uv run poe stage4        (Docker: docker compose run --rm app poe stage4)
"""

import json
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client, evaluate, traceable
from pydantic import BaseModel, Field

from docshelp.config import DATA_DIR, describe, ensure_llm_ready, get_llm, langsmith_enabled, structured_llm
from docshelp.graph import build_graph
from docshelp.ingest import format_docs, get_retriever

DATASET_NAME = "docshelp-qa"
EXAMPLES = json.loads((DATA_DIR / "eval_dataset.json").read_text(encoding="utf-8"))

ensure_llm_ready()
llm = get_llm()
retriever = get_retriever()


# ---------------------------------------------------------------------------
# The two systems under test. Each takes the dataset "inputs" and returns "outputs".
# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using only this Nimbus Notes documentation. Be concise.\n\n{context}"),
        ("human", "{question}"),
    ]
)
rag_chain = RAG_PROMPT | llm | StrOutputParser()


@traceable(name="rag_chain_target")
def rag_chain_target(inputs: dict) -> dict:
    docs = retriever.invoke(inputs["question"])
    answer = rag_chain.invoke({"context": format_docs(docs), "question": inputs["question"]})
    return {"answer": answer, "sources": [d.metadata["source"] for d in docs]}


graph = build_graph(llm=llm, retriever=retriever, checkpointer=InMemorySaver())


@traceable(name="langgraph_agent_target")
def graph_target(inputs: dict) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = graph.invoke({"messages": [HumanMessage(inputs["question"])]}, config)
    return {
        "answer": state["messages"][-1].text,
        "sources": [d["source"] for d in state.get("documents", [])],
    }


# ---------------------------------------------------------------------------
# Evaluators: (inputs, outputs, reference_outputs) -> score
# ---------------------------------------------------------------------------


class Verdict(BaseModel):
    reasoning: str = Field(description="Brief comparison of the answer against the reference.")
    correct: bool = Field(
        description="True if the answer is factually consistent with the reference and answers the question."
    )


judge = structured_llm(llm, Verdict)


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge: is the answer consistent with the reference answer?"""
    verdict = judge.invoke(
        "You are grading a support assistant.\n"
        f"Question: {inputs['question']}\n"
        f"Reference answer: {reference_outputs['answer']}\n"
        f"Assistant answer: {outputs['answer']}"
    )
    return {"key": "correctness", "score": verdict.correct, "comment": verdict.reasoning}


def retrieved_expected_source(outputs: dict, reference_outputs: dict) -> bool:
    """Code evaluator: did retrieval surface the doc that holds the answer?"""
    return reference_outputs["source"] in outputs["sources"]


EVALUATORS = [correctness, retrieved_expected_source]
TARGETS = {"rag-chain": rag_chain_target, "langgraph-agent": graph_target}


# ---------------------------------------------------------------------------


def ensure_dataset(client: Client) -> None:
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset '{DATASET_NAME}' already exists in LangSmith.")
        return
    dataset = client.create_dataset(DATASET_NAME, description="Nimbus Notes support questions")
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": e["question"]},
                "outputs": {"answer": e["answer"], "source": e["source"]},
            }
            for e in EXAMPLES
        ],
    )
    print(f"Created dataset '{DATASET_NAME}' with {len(EXAMPLES)} examples.")


def run_with_langsmith() -> None:
    client = Client()
    ensure_dataset(client)
    for name, target in TARGETS.items():
        print(f"\nRunning experiment '{name}'...")
        results = evaluate(
            target,
            data=DATASET_NAME,
            evaluators=EVALUATORS,
            experiment_prefix=name,
            metadata={"model": describe()},
            max_concurrency=4,
        )
        print_summary(name, [row["evaluation_results"]["results"] for row in results])
    print(
        "\nOpen LangSmith → Datasets & Experiments → docshelp-qa, select both experiments and "
        "click Compare to see per-question differences, traces, latency and token costs."
    )


def run_locally() -> None:
    print(
        "LangSmith is not configured (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env).\n"
        "Running the same evaluation locally; nothing is uploaded.\n"
    )
    for name, target in TARGETS.items():
        print(f"Running '{name}'...")
        rows = []
        for e in EXAMPLES:
            inputs = {"question": e["question"]}
            reference = {"answer": e["answer"], "source": e["source"]}
            outputs = target(inputs)
            verdict = correctness(inputs, outputs, reference)
            rows.append(
                [
                    _Result("correctness", verdict["score"]),
                    _Result("retrieved_expected_source", retrieved_expected_source(outputs, reference)),
                ]
            )
            mark = "✓" if verdict["score"] else "✗"
            print(f"  {mark} {e['question']}")
        print_summary(name, rows)


class _Result:
    def __init__(self, key: str, score):
        self.key, self.score = key, score


def print_summary(name: str, rows: list[list]) -> None:
    totals: dict[str, list[float]] = {}
    for results in rows:
        for r in results:
            totals.setdefault(r.key, []).append(float(r.score or 0))
    summary = ", ".join(f"{k}={sum(v) / len(v):.0%}" for k, v in totals.items())
    print(f"  => {name}: {summary}")


def main() -> None:
    print(f"Using {describe()} for the app and the judge")
    if langsmith_enabled():
        run_with_langsmith()
    else:
        run_locally()


if __name__ == "__main__":
    main()
