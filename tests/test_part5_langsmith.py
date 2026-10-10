"""Part 5 lessons (LangSmith), driven offline: tracing and evals use the local twins, never the network."""

import pytest

from docshelp import observability

from .conftest import FakeChatModel, tool_call


@pytest.fixture(autouse=True)
def no_langsmith(monkeypatch, tmp_path):
    """Force local mode (even if a developer's .env enables LangSmith) and keep logs out of data/."""
    for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(observability, "TRACES_PATH", tmp_path / "traces.jsonl")
    monkeypatch.setattr(observability, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")


def run_types(run):
    return [run.run_type] + [t for child in run.child_runs for t in run_types(child)]


def test_5_1_tracing_basics_builds_a_local_run_tree(lesson, fake_retriever, tmp_path):
    llm = FakeChatModel(
        replies=[tool_call("quote_price", {"plan": "team", "members": 12, "billing": "annual"}), "$921.60 per year."]
    )
    path = tmp_path / "t.jsonl"

    result = lesson("5.1").run(llm, fake_retriever, traces_path=path)

    run = result["run"]
    assert run.name == "support_request" and run.outputs == {"answer": "$921.60 per year."}
    assert {"llm", "tool", "chain"} <= set(run_types(run))
    assert observability.total_tokens(run) == 30  # two model calls x 15 tokens (FakeChatModel's usage)
    rows = observability.load_rows(path)
    assert rows[0]["parent_id"] is None and all(r["trace_id"] == rows[0]["trace_id"] for r in rows)
    assert rows[0]["metadata"]["plan"] == "team"


def test_5_2_traceable_run_types_and_masked_inputs(lesson, fake_retriever, tmp_path):
    llm = FakeChatModel(replies=["  7 days [05-import-export.md]  "])

    result = lesson("5.2").run(llm, fake_retriever, traces_path=tmp_path / "t.jsonl")

    assert result["answer"] == "7 days [05-import-export.md]"  # the parser step stripped it
    assert {"retriever", "prompt", "parser", "llm"} <= set(run_types(result["run"]))
    pipeline = result["run"].child_runs[0]
    assert "[email]" in str(pipeline.inputs) and "ada@acme.example" not in str(pipeline.inputs)
    assert pipeline.tags == ["rag"] or "rag" in pipeline.tags


def test_5_3_threads_group_turns_and_analysis(lesson, fake_retriever, tmp_path):
    llm = FakeChatModel(replies=["File > Export.", "Yes.", "Yes, on desktop.", "Yes, on mobile too."])

    result = lesson("5.3").run(llm, fake_retriever, traces_path=tmp_path / "t.jsonl")

    assert len(result["threads"]) == 2
    assert all(t["turns"] == 2 and t["tokens"] == 30 for t in result["threads"].values())
    assert result["llm_calls"] == 4


def test_5_4_datasets_splits_versions_and_synthetic_examples(lesson):
    llm = FakeChatModel(replies=["How long until deleted notes are gone for good?", "Can I get a refund?"])

    result = lesson("5.4").run(llm)

    assert len(result["rows"]) == 9 and {r["split"] for r in result["rows"]} == {"base", "edge"}
    assert [c["split"] for c in result["candidates"]] == ["synthetic", "synthetic"]
    assert result["candidates"][0]["answer"] == result["rows"][0]["answer"]  # same reference answer


def judge_says(*scores):
    """Scripted openevals judge verdicts (it asks for a JSON-schema titled "score")."""
    return {"score": [{"score": s, "reasoning": "scripted"} for s in scores]}


def test_5_5_code_llm_and_trajectory_evaluators(lesson, fake_retriever):
    llm = FakeChatModel(
        replies=[tool_call("quote_price", {"plan": "team", "members": 12, "billing": "monthly"}), "$96.00/month."],
        structured=judge_says(True, False, True),  # good answer, bad answer, trajectory
    )

    result = lesson("5.5").run(llm, fake_retriever)

    assert result["code"]["good"]["score"] is True and result["code"]["bad"]["score"] is False
    assert result["correctness"] == {"good": True, "bad": False}
    assert result["trajectory_match"] is True  # the agent called quote_price, like the reference
    assert result["trajectory_judge"] is True


def test_trajectory_match_fails_when_the_agent_skips_the_tool(lesson):
    from agentevals.trajectory.match import create_trajectory_match_evaluator

    module = lesson("5.5")
    lazy = [module.HumanMessage(module.PRICE_QUESTION), module.AIMessage("About $100.")]  # guessed, no tool
    match = create_trajectory_match_evaluator(trajectory_match_mode="superset", tool_args_match_mode="ignore")
    assert match(outputs=lazy, reference_outputs=module.REFERENCE_TRAJECTORY)["score"] is False


def test_5_6_experiments_compare_two_versions(lesson, fake_retriever):
    from docshelp.graph import Grade, Route

    edge = 3  # the "edge" split
    llm = FakeChatModel(
        replies=["An answer [08-api.md]"] * edge * 2,  # rag chain, then the graph's generate node
        structured={
            "score": [{"score": s, "reasoning": "scripted"} for s in (True, False, False, True, True, True)],
            Route: [Route(intent="docs_question", standalone_question="q")] * edge,
            Grade: [Grade(relevant=True, reason="ok")] * edge,
        },
    )

    result = lesson("5.6").run(llm, fake_retriever, split="edge")

    rag, graph = result["experiments"]["rag-chain"], result["experiments"]["support-graph"]
    assert len(rag["rows"]) == len(graph["rows"]) == edge
    assert all(set(r["scores"]) == {"expected_source", "cites_source", "correctness"} for r in rag["rows"])
    assert len(result["disagreements"]) == 2  # examples 1 and 2 were judged differently


def test_5_7_pairwise_with_position_swap(lesson, fake_retriever):
    from docshelp.evals import Preference
    from docshelp.graph import Grade, Route

    llm = FakeChatModel(
        replies=["rag answer"] * 3 + ["graph answer"] * 3,
        structured={
            Route: [Route(intent="docs_question", standalone_question="q")] * 3,
            Grade: [Grade(relevant=True, reason="ok")] * 3,
            # "A" twice: shown in normal order for example 1, swapped for example 2; then a tie
            Preference: [Preference(reasoning="r", preferred=p) for p in ("A", "A", "tie")],
        },
    )

    result = lesson("5.7").run(llm, fake_retriever, split="edge")

    assert result["tally"] == {"rag-chain": 1, "support-graph": 1, "tie": 1}


def test_5_8_summary_evaluators_precision_recall_f1(lesson):
    module = lesson("5.8")
    labels = [True, False, True, False, True, False, True, False]  # the dataset's answers, in order
    # Predict: catch 3 of the 4 real tickets, plus 1 false alarm.
    predictions = [True, True, True, False, False, False, True, False]
    decisions = [module.TicketDecision(needs_ticket=p) for p in predictions]
    llm = FakeChatModel(structured={module.TicketDecision: decisions})

    result = module.run(llm)

    assert [r["reference"]["needs_ticket"] for r in result["rows"]] == labels
    assert result["summary"]["precision"] == pytest.approx(3 / 4)
    assert result["summary"]["recall"] == pytest.approx(3 / 4)
    assert result["summary"]["f1"] == pytest.approx(3 / 4)


def test_5_9_evals_run_as_pytest_with_an_injected_model(lesson, fake_retriever):
    llm = FakeChatModel(
        replies=[
            "7 days [05-import-export.md]",  # rag, case 1
            "Enterprise [06-security.md]",  # rag, case 2
            tool_call("quote_price", {"plan": "team", "members": 12, "billing": "monthly"}),  # agent
            "$96.00 per month.",
        ]
    )

    result = lesson("5.9").run(llm, fake_retriever)

    assert len(result["outcomes"]) == 3  # every eval ran, through the plugin-injected model
    assert result["outcomes"]["test_prices_come_from_the_pricing_tool"] == "passed"
    assert not llm.replies


def test_5_10_prompt_lifecycle_promotes_the_better_version(lesson, fake_retriever, tmp_path):
    llm = FakeChatModel(replies=["No citation."] * 3 + ["Cited. [08-api.md]"] * 3)  # v1 runs first, then v2

    result = lesson("5.10").run(llm, fake_retriever, hub_root=tmp_path)

    assert result["winner"] == "v2"
    assert "square brackets" in result["prod_prompt"].messages[0].prompt.template  # prod now pulls v2


def test_local_prompt_hub_versions_tags_and_rollback(tmp_path):
    from langchain_core.prompts import ChatPromptTemplate

    from docshelp.prompts import LocalPromptHub

    hub = LocalPromptHub(tmp_path)
    one = hub.push("p", ChatPromptTemplate.from_template("v1 {x}"), tags=["prod"])
    two = hub.push("p", ChatPromptTemplate.from_template("v2 {x}"))
    assert hub.push("p", ChatPromptTemplate.from_template("v2 {x}")) == two  # same content, same commit
    assert hub.pull("p").format(x="!") == "Human: v2 !"  # latest
    assert hub.pull("p:prod").format(x="!") == "Human: v1 !"
    hub.tag("p", two, "prod")
    hub.tag("p", one, "prod")  # rollback
    assert hub.pull("p:prod").format(x="!") == "Human: v1 !"
    with pytest.raises(KeyError):
        hub.pull("p:nope")


def test_5_11_feedback_is_attached_to_the_rated_run(lesson, fake_retriever, tmp_path):
    llm = FakeChatModel(
        replies=["Use Share [04-sharing-and-permissions.md]", "No idea.", "Ctrl+N [01-getting-started.md]"]
    )
    path = tmp_path / "fb.jsonl"

    result = lesson("5.11").run(llm, fake_retriever, feedback_path=path)

    assert [r["score"] for r in result["feedback"]] == [1.0, 0.0, 1.0]
    assert len(result["negative"]) == 1 and result["negative"][0]["comment"]
    traced_ids = {r["id"] for r in observability.load_rows(observability.TRACES_PATH) if r["parent_id"] is None}
    assert {r["run_id"] for r in result["feedback"]} <= traced_ids  # every rating points at a real trace
    assert len(observability.load_rows(path)) == 3


def test_5_12_annotation_queue_turns_corrections_into_examples(lesson, fake_retriever, tmp_path):
    llm = FakeChatModel(replies=["30 days [07-troubleshooting.md]", "Sure, anywhere.", "Team and up."])

    result = lesson("5.12").run(llm, fake_retriever, queue_path=tmp_path / "queue.jsonl")

    assert [item["question"] for item in result["queue"]] == [
        "Can I get my data hosted in Japan?", "Which plan includes version history?"
    ]
    assert len(result["new_examples"]) == 2 and all(e["source_run_id"] for e in result["new_examples"])
    keys = [r["key"] for r in observability.load_rows(observability.FEEDBACK_PATH)]
    assert keys.count("reviewer_correct") == 2 and keys.count("reviewer_helpful") == 2


def test_examples_have_stable_ids_and_versions_change_with_content():
    from docshelp.evals import load_rows, to_examples, version_of

    rows = load_rows()
    assert [e.id for e in to_examples(rows)] == [e.id for e in to_examples(rows)]
    assert version_of(rows) != version_of(rows[:-1])
