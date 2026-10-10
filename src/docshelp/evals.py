"""Datasets, targets, evaluators and experiments for DocsHelp (Part 5).

Everything uses LangSmith's own evaluation API (`langsmith.evaluate`). With a LangSmith key, datasets
and experiments are uploaded and can be compared in the UI. Without one, the same `evaluate()` runs
locally (`upload_results=False`) over in-memory `Example`s: the local twin.
"""

from __future__ import annotations

import hashlib
import json
import uuid
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client, evaluate
from langsmith.schemas import Example
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from pydantic import BaseModel, Field

from docshelp.config import DATA_DIR, describe, langsmith_enabled, structured_llm
from docshelp.ingest import format_docs
from docshelp.observability import offline_client

EVAL_PATH = DATA_DIR / "eval_dataset.json"
DATASET_NAME = "docshelp-qa"
NAMESPACE = uuid.UUID("6f1c2a52-0d2e-4b8e-9a51-6b3f1d7c0e11")  # stable example ids across runs


# --- datasets ---------------------------------------------------------------------------------------


def load_rows(split: str | None = None) -> list[dict]:
    rows = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    return [r for r in rows if split is None or r.get("split") == split]


def version_of(rows: list[dict]) -> str:
    """A content hash: any change to the examples gives a new version (LangSmith versions on every edit)."""
    return hashlib.sha1(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:8]


def to_examples(rows: list[dict]) -> list[Example]:
    """Q&A rows as LangSmith `Example`s: inputs, reference outputs, and the split in metadata."""
    pairs = [({"question": r["question"]}, {"answer": r["answer"], "source": r["source"]}) for r in rows]
    return make_examples(DATASET_NAME, pairs, [r.get("split", "base") for r in rows])


def make_examples(name: str, pairs: list[tuple[dict, dict]], splits: list[str] | None = None) -> list[Example]:
    """Any (inputs, reference outputs) pairs as LangSmith `Example`s of dataset `name`."""
    now = datetime.now(timezone.utc)
    return [
        Example(
            id=uuid.uuid5(NAMESPACE, f"{name}:{json.dumps(inputs, sort_keys=True)}"),
            dataset_id=uuid.uuid5(NAMESPACE, name),
            inputs=inputs,
            outputs=outputs,
            metadata={"dataset_split": [splits[n] if splits else "base"]},
            created_at=now,
            modified_at=now,
        )
        for n, (inputs, outputs) in enumerate(pairs)
    ]


def ensure_dataset(client: Client, examples: list[Example], name: str) -> str:
    """Create the dataset in LangSmith once (with splits); return its name."""
    if client.has_dataset(dataset_name=name):
        return name
    dataset = client.create_dataset(name, description=f"DocsHelp evaluation dataset '{name}'")
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": e.inputs, "outputs": e.outputs, "split": (e.metadata or {}).get("dataset_split", ["base"])[0]}
            for e in examples
        ],
    )
    return name

# --- targets: (inputs) -> outputs -------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using only this Nimbus Notes documentation. Be concise and cite sources.\n\n{context}"),
        ("human", "{question}"),
    ]
)


def make_rag_target(llm: BaseChatModel, retriever: BaseRetriever, prompt: ChatPromptTemplate = RAG_PROMPT):
    """The RAG chain from lesson 1.5 as an evaluation target."""
    chain = prompt | llm | StrOutputParser()

    def rag_chain(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        answer = chain.invoke({"context": format_docs(docs), "question": inputs["question"]})
        return {"answer": answer, "sources": [d.metadata.get("source", "?") for d in docs]}

    return rag_chain


def make_graph_target(llm: BaseChatModel, retriever: BaseRetriever):
    """The support graph from lesson 3.15 as an evaluation target."""
    from docshelp.graph import build_graph

    graph = build_graph(llm, retriever, InMemorySaver())

    def support_graph(inputs: dict) -> dict:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = graph.invoke({"messages": [HumanMessage(inputs["question"])]}, config)
        return {"answer": state["messages"][-1].text, "sources": [d["source"] for d in state.get("documents", [])]}

    return support_graph


def make_agent_target(llm: BaseChatModel, retriever: BaseRetriever):
    """The hand-built ReAct agent (lesson 3.3), returning its answer *and* its trajectory of messages."""
    from docshelp.react import build_react_agent
    from docshelp.tools import make_search_docs, quote_price

    agent = build_react_agent(llm, [make_search_docs(retriever), quote_price])

    def react_agent(inputs: dict) -> dict:
        messages = agent.invoke({"messages": [HumanMessage(inputs["question"])]})["messages"]
        tools = [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
        return {"answer": messages[-1].text, "tools": tools, "messages": messages}

    return react_agent


# --- evaluators: (inputs, outputs, reference_outputs) -> score ---------------------------------------


def expected_source(outputs: dict, reference_outputs: dict) -> dict:
    """Code evaluator: did retrieval surface the document that holds the answer?"""
    return {"key": "expected_source", "score": reference_outputs["source"] in outputs.get("sources", [])}


def cites_source(outputs: dict) -> dict:
    """Code evaluator: does the answer cite a doc file, as the prompt asks?"""
    return {"key": "cites_source", "score": ".md" in outputs.get("answer", "")}


def make_correctness_judge(llm: BaseChatModel):
    """LLM-as-judge (openevals' correctness rubric) comparing the answer with the reference answer."""
    judge = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, judge=llm, feedback_key="correctness")

    def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        return judge(
            inputs=inputs["question"], outputs=outputs["answer"], reference_outputs=reference_outputs["answer"]
        )

    return correctness


# --- experiments ------------------------------------------------------------------------------------


def run_experiment(
    target: Callable[[dict], dict],
    rows: list[dict] | list[Example],
    evaluators: Sequence[Callable],
    *,
    prefix: str,
    summary_evaluators: Sequence[Callable] = (),
    metadata: dict | None = None,
    repetitions: int = 1,
    dataset_name: str = DATASET_NAME,
) -> dict:
    """Run `target` over the examples and score it. Uploads to LangSmith if configured, else local.

    `rows` are Q&A rows (`load_rows()`) or ready-made `Example`s (`make_examples()`).
    """
    online = langsmith_enabled()
    examples = rows if rows and isinstance(rows[0], Example) else to_examples(rows)
    data: Any = ensure_dataset(Client(), examples, dataset_name) if online else examples
    with warnings.catch_warnings():
        # Local runs use `upload_results=False`, which langsmith flags as beta; the API is stable enough here.
        warnings.filterwarnings("ignore", message="'upload_results' parameter is in beta")
        results = evaluate(
            target,
            data=data,
            evaluators=list(evaluators),
            summary_evaluators=list(summary_evaluators),
            experiment_prefix=prefix,
            metadata={"model": describe(), **(metadata or {})},
            num_repetitions=repetitions,
            max_concurrency=0,  # one example at a time: easy to follow; raise it for speed
            upload_results=online,
            client=None if online else offline_client(),
        )
    rows_out = []
    for row in results:
        rows_out.append(
            {
                "inputs": row["example"].inputs,
                "question": next(iter(row["example"].inputs.values()), ""),
                "reference": row["example"].outputs or {},
                "outputs": row["run"].outputs or {},
                "error": row["run"].error,
                "scores": {r.key: r.score for r in row["evaluation_results"]["results"]},
            }
        )
    summary = {r.key: r.score for r in (results._summary_results or {}).get("results", [])}
    return {"name": results.experiment_name, "rows": rows_out, "summary": summary}


def averages(experiment: dict) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for row in experiment["rows"]:
        for key, score in row["scores"].items():
            if score is not None:
                totals.setdefault(key, []).append(float(score))
    return {key: sum(v) / len(v) for key, v in totals.items()}


def compare(experiments: dict[str, dict]) -> str:
    """A side-by-side table of average scores, like LangSmith's experiment comparison view."""
    keys = sorted({k for e in experiments.values() for k in averages(e)})
    lines = [f"  {'experiment':22}" + "".join(f"{k:>18}" for k in keys)]
    for name, experiment in experiments.items():
        avg = averages(experiment)
        lines.append(f"  {name:22}" + "".join(f"{avg.get(k, 0):>18.0%}" for k in keys))
    return "\n".join(lines)


# --- pairwise ---------------------------------------------------------------------------------------


class Preference(BaseModel):
    reasoning: str = Field(description="Compare the two answers in one or two sentences.")
    preferred: Literal["A", "B", "tie"] = Field(description="Which answer better helps the customer.")


def pairwise_judge(llm: BaseChatModel):
    """Judge two answers to the same question. Returns 'first', 'second' or 'tie'.

    The pair is shown in both orders across examples (`swap`), because judges tend to favour the first
    answer they read (position bias).
    """
    judge = structured_llm(llm, Preference)

    def prefer(question: str, first: str, second: str, *, swap: bool) -> tuple[str, str]:
        a, b = (second, first) if swap else (first, second)
        verdict = judge.invoke(
            "Which reply better answers this Nimbus Notes support question? Prefer correct, specific, concise "
            f"answers.\n\nQuestion: {question}\n\n<answer A>\n{a}\n</answer A>\n\n<answer B>\n{b}\n</answer B>"
        )
        if verdict.preferred == "tie":
            return "tie", verdict.reasoning
        first_won = (verdict.preferred == "A") != swap
        return ("first" if first_won else "second"), verdict.reasoning

    return prefer
