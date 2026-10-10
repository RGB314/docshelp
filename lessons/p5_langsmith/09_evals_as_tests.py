"""Evals as tests: run evaluations with pytest, tracked by LangSmith's pytest plugin.

Academy: Building Reliable Agents M2 (evaluation in the development workflow); LangSmith Essentials M2
"Datasets and Offline Evals".

Concepts:
  * the evals in `evals/test_support_evals.py` are ordinary pytest tests with
    `@pytest.mark.langsmith`. They log inputs, outputs, reference outputs and feedback through
    `langsmith.testing`, then `assert` like any test.
  * with LangSmith configured, each run is uploaded as an experiment on a dataset named after the test
    file, with no `evaluate()` code to write. Without it, `LANGSMITH_TEST_TRACKING=false` keeps them local.
  * why pytest: evals fit the workflow you already have. Run them in CI on every prompt or model change;
    a failing eval blocks a regression the way a failing unit test does.
  * keep them separate from unit tests: they call a real model, cost time, and can be flaky. Here they
    live in `evals/` (run with `uv run poe evals`), while `uv run poe test` stays offline.

Run:  uv run poe lesson 5.9      (or: uv run poe evals)
"""

import os
from contextlib import contextmanager

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import PROJECT_ROOT, describe, ensure_llm_ready, get_llm, langsmith_enabled
from docshelp.console import header
from docshelp.ingest import get_retriever


class Models:
    """A pytest plugin that hands the lesson's model and retriever to the evals' fixtures."""

    provides_docshelp_models = True  # evals/conftest.py looks for this

    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever):
        self.llm, self.retriever = llm, retriever
        self.outcomes: dict[str, str] = {}

    def pytest_runtest_logreport(self, report):  # a pytest hook: record each test's result
        if report.when == "call" or report.outcome == "skipped":
            self.outcomes[report.nodeid.split("::")[-1]] = report.outcome


@contextmanager
def tracking(enabled: bool):
    old = os.environ.get("LANGSMITH_TEST_TRACKING")
    os.environ["LANGSMITH_TEST_TRACKING"] = "true" if enabled else "false"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("LANGSMITH_TEST_TRACKING", None)
        else:
            os.environ["LANGSMITH_TEST_TRACKING"] = old


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    plugin = Models(llm, retriever)
    header(f"pytest evals/ ({'tracked in LangSmith' if langsmith_enabled() else 'local only'})")
    with tracking(langsmith_enabled()):
        exit_code = pytest.main(
            [str(PROJECT_ROOT / "evals"), "-q", "-p", "no:cacheprovider", "--rootdir", str(PROJECT_ROOT)],
            plugins=[plugin],
        )
    header("Results")
    for name, outcome in plugin.outcomes.items():
        print(f"  {outcome:8} {name}")
    print("\nFailing evals are information, not crashes: read the failure, then fix the prompt, retriever or tool.")
    return {"exit_code": int(exit_code), "outcomes": plugin.outcomes}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
