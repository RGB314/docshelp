"""Run types and other ways to trace: `@traceable`, `trace()`, and hiding personal data.

Academy: Intro to Agent Observability & Evaluations M1 L2 "Types of Runs", M1 L3 "Alternative Ways to
Trace".

Concepts:
  * run types tell LangSmith how to display a step: `llm`, `chain`, `tool`, `retriever`, `prompt`,
    `parser`, `embedding`. LangChain sets them for you; for your own functions you choose.
  * `@traceable(run_type=..., name=...)` traces any Python function, nested correctly under whatever
    called it. Here a hand-written RAG pipeline: retriever → prompt → llm → parser.
  * a `retriever` run returning `[{"page_content", "type": "Document", "metadata"}]` renders as documents
  * `trace("name")` (a context manager) traces a block that isn't a function
  * `process_inputs` / `process_outputs` change what is *recorded* (here: mask email addresses) without
    changing what the code receives. `LANGSMITH_HIDE_INPUTS=true` hides inputs everywhere.
  * metadata and tags per call: `fn(..., langsmith_extra={"metadata": {...}, "tags": [...]})`
  * also available, not shown here: `wrap_openai` / `wrap_anthropic` for vendor SDKs used without
    LangChain, and OpenTelemetry export (`LANGSMITH_OTEL_ENABLED=true`) to send traces to other tools too

Run:  uv run poe lesson 5.2
"""

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langsmith import traceable
from langsmith.run_helpers import trace

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.observability import print_tree, traced

QUESTION = "I'm ada@acme.example. How long is the workspace export link valid?"
EMAIL = re.compile(r"\S+@\S+")


def mask_emails(inputs: dict) -> dict:
    """Recorded in the trace instead of the real inputs."""
    return {k: EMAIL.sub("[email]", v) if isinstance(v, str) else v for k, v in inputs.items()}


def build_pipeline(llm: BaseChatModel, retriever: BaseRetriever):
    @traceable(run_type="retriever", name="search_docs")
    def retrieve(query: str) -> list[dict]:
        return [
            {"page_content": d.page_content, "type": "Document", "metadata": d.metadata}
            for d in retriever.invoke(query)
        ]

    @traceable(run_type="prompt", name="build_prompt")
    def build_prompt(question: str, docs: list[dict]) -> str:
        context = "\n\n".join(f"[{d['metadata'].get('source', '?')}] {d['page_content']}" for d in docs)
        return f"Answer briefly from these docs, citing sources.\n\n{context}\n\nQuestion: {question}"

    @traceable(run_type="parser", name="clean_answer")
    def parse(text: str) -> str:
        return text.strip()

    @traceable(run_type="chain", name="rag_pipeline", process_inputs=mask_emails)
    def answer(question: str) -> str:
        docs = retrieve(EMAIL.sub("", question))
        with trace("generate", run_type="chain"):  # a traced block, not a function
            reply = llm.invoke(build_prompt(question, docs))  # LangChain adds its own `llm` run here
        return parse(reply.text)

    return answer


def run(llm: BaseChatModel, retriever: BaseRetriever, traces_path=None) -> dict:
    pipeline = build_pipeline(llm, retriever)
    header("A hand-written RAG pipeline, traced with @traceable")
    with traced("support_request", {"question": mask_emails({"q": QUESTION})["q"]}, path=traces_path) as root:
        answer = pipeline(QUESTION, langsmith_extra={"metadata": {"channel": "email"}, "tags": ["rag"]})
        root.end(outputs={"answer": answer})
    print(f"DocsHelp: {answer}")
    header("Run types in the trace")
    print_tree(root)
    pipeline_run = root.child_runs[0]
    print(f"\nRecorded inputs of rag_pipeline: {pipeline_run.inputs}   (the email is masked)")
    return {"run": root, "answer": answer}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
