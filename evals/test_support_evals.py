"""DocsHelp evals written as pytest tests, tracked by LangSmith's pytest plugin (lesson 5.9).

`@pytest.mark.langsmith` turns a test into an evaluation: with LangSmith configured, each test case
becomes an example in a dataset named after this file, every run becomes an experiment, and what you
log with `langsmith.testing` (inputs, outputs, reference outputs, feedback) appears in the UI.
Without LangSmith (`LANGSMITH_TEST_TRACKING=false`), they are ordinary pytest tests.

Run:  uv run poe evals
"""

import pytest
from langsmith import testing as t

CASES = [
    ("How long is the workspace export download link valid?", "05-import-export.md"),
    ("Which plan do I need for Okta SSO?", "06-security.md"),
]


@pytest.mark.langsmith
@pytest.mark.parametrize("question, source", CASES)
def test_retrieves_the_doc_with_the_answer(rag, question, source):
    t.log_inputs({"question": question})
    t.log_reference_outputs({"source": source})
    outputs = rag({"question": question})
    t.log_outputs(outputs)
    found = source in outputs["sources"]
    t.log_feedback(key="expected_source", score=found)
    assert found, f"{source} not in {outputs['sources']}"


@pytest.mark.langsmith
def test_prices_come_from_the_pricing_tool(agent):
    question = "What does the Team plan cost for 12 members, billed monthly?"
    t.log_inputs({"question": question})
    outputs = agent({"question": question})
    t.log_outputs({"answer": outputs["answer"], "tools": outputs["tools"]})
    used_tool = "quote_price" in outputs["tools"]
    t.log_feedback(key="used_pricing_tool", score=used_tool)
    assert used_tool, "the agent guessed a price instead of calling quote_price"
