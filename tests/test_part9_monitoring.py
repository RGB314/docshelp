"""Part 9 (monitoring and the improvement loop): every lesson's `run()` with a scripted model, offline."""

import tomllib
from datetime import datetime, timedelta

import pytest

from docshelp import dashboard, monitoring
from docshelp.config import PROJECT_ROOT
from docshelp.engine import PromptFix
from docshelp.evals import Preference
from docshelp.insights import Categories, Category, ConversationSummary
from docshelp.monitoring import Helpfulness, Sentiment
from docshelp.prompts import LocalPromptHub

from .conftest import FakeChatModel, tool_call

ANSWER = "Use File > Export > PDF (05-import-export.md)."


def traffic_model(**kwargs) -> FakeChatModel:
    """Searches the docs once per conversation turn, then answers: what a good model does."""
    return FakeChatModel(default=ANSWER, **kwargs)


@pytest.fixture
def paths(tmp_path):
    return {"path": tmp_path / "traces.jsonl"}


@pytest.fixture
def hub(tmp_path):
    return LocalPromptHub(tmp_path / "prompts")


# --- the shared pieces -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "row", "expected"),
    [
        ('eq(status, "error")', {"status": "error"}, True),
        ('and(eq(status, "error"), gt(latency, 5))', {"status": "error", "latency": 2}, False),
        ("gte(total_tokens, 100)", {"total_tokens": 100}, True),
        ('has(tags, "production")', {"tags": ["production"]}, True),
        ('search("REFUND")', {"inputs": "I want a refund", "outputs": ""}, True),
        ('and(eq(metadata_key, "plan"), eq(metadata_value, "team"))', {"metadata": {"plan": "team"}}, True),
        ('and(eq(metadata_key, "plan"), eq(metadata_value, "team"))', {"metadata": {"plan": "free"}}, False),
        ('and(eq(feedback_key, "helpfulness"), lt(feedback_score, 1))', {"feedback": {"helpfulness": 0.0}}, True),
        ('and(eq(feedback_key, "helpfulness"), lt(feedback_score, 1))', {"feedback": {}}, False),
        ('or(eq(name, "a"), not(eq(run_type, "chain")))', {"name": "b", "run_type": "llm"}, True),
    ],
)
def test_langsmith_filters_run_locally(expression, row, expected):
    assert monitoring.run_filter(expression)(row) is expected


def test_unknown_filter_functions_are_rejected():
    with pytest.raises(ValueError):
        monitoring.run_filter('like(name, "x")')


def test_costs_percentiles_and_significance():
    pricing = {"models": {"big": {"input": 2.0, "output": 10.0}}}
    run = {"run_type": "llm", "model": "big-v2", "input_tokens": 1_000_000, "output_tokens": 100_000}
    assert monitoring.trace_cost({"runs": [run]}, pricing) == pytest.approx(3.0)  # prefix match: 2 + 1
    assert monitoring.trace_cost({"runs": [{**run, "model": "local"}]}, pricing) == 0.0
    assert monitoring.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert monitoring.percentile([1, 2, 3, 4], 50) == 2
    assert monitoring.two_proportion_test(50, 100, 70, 100)["p_value"] < 0.01
    assert monitoring.two_proportion_test(5, 10, 6, 10)["p_value"] > 0.5


def test_variant_assignment_is_stable_and_follows_the_split():
    variants = {"a": 0.5, "b": 0.5}
    assert monitoring.assign_variant("acct-1", variants) == monitoring.assign_variant("acct-1", variants)
    picks = [monitoring.assign_variant(f"acct-{n}", variants) for n in range(1000)]
    assert 400 < picks.count("a") < 600


def test_a_failing_online_evaluator_is_skipped_not_fatal(tmp_path):
    def broken(summary):
        raise ValueError("invalid structured output")

    summaries = [{"trace_id": "t1", "inputs": '{"question": "q"}', "outputs": '{"answer": "see 01-intro.md"}'}]
    evaluators = [monitoring.OnlineEvaluator(broken), monitoring.OnlineEvaluator(monitoring.cites_docs)]

    records = monitoring.run_online_evaluators(summaries, evaluators, feedback_path=tmp_path / "f.jsonl")

    assert [(r["key"], r["score"]) for r in records] == [("cites_docs", 1.0)]


def test_alerts_fire_only_inside_their_window():
    now = datetime(2026, 1, 1, 12, 0)
    old = {
        "start": (now - timedelta(hours=2)).isoformat(),
        "status": "error",
        "latency": 1,
        "total_tokens": 0,
        "trace_id": "t1",
        "tags": [],
        "runs": [],
    }
    recent = {**old, "start": now.isoformat(), "trace_id": "t2", "status": "success"}
    rule = monitoring.AlertRule("errors", "error_rate", 0.1, window_minutes=15)
    assert monitoring.check_alerts([old, recent], [rule], now=now, path=None) == []
    fired = monitoring.check_alerts([old, {**recent, "status": "error"}], [rule], now=now, path=None)
    assert fired[0]["alert"] == "errors" and fired[0]["value"] == 1.0 and fired[0]["traces"] == 1


# --- lessons ---------------------------------------------------------------------------------------------


def test_9_1_traffic_is_traced_with_metadata_and_filterable(lesson, fake_retriever, paths, hub):
    llm = traffic_model(replies=[tool_call("search_docs", {"query": "export pdf"})])
    result = lesson("9.1").run(llm, fake_retriever, limit=3, hub=hub, **paths)

    assert result["new"] == result["total"] == 4  # three conversations, one of them two turns
    summaries = monitoring.summarize(monitoring.load_rows(paths["path"]))
    assert summaries[0]["tools"] == ["search_docs"] and summaries[0]["llm_calls"] == 2
    assert summaries[0]["metadata"]["plan"] == "team" and "production" in summaries[0]["tags"]
    assert result["counts"]["Team plan"] == 1 and result["counts"]["errors"] == 0
    assert hub.history(monitoring.PROMPT_NAME)[0]["tags"] == ["prod"]  # v1 published on first use


def test_9_2_costs_tokens_per_plan_and_what_if(lesson, fake_retriever, paths):
    result = lesson("9.2").run(traffic_model(), fake_retriever, **paths)

    assert result["turns"] == 14  # every conversation in data/traffic.json
    assert result["input_tokens"] == 14 * 10 and result["output_tokens"] == 14 * 5  # FakeChatModel's usage per call
    assert result["cost"] == 0.0  # local models are free
    assert result["what_if"]["large hosted model"] > result["what_if"]["small hosted model"] > 0
    assert set(result["by_plan"]) == {"team", "enterprise", "free"}
    assert isinstance(result["callback"], dict)  # keyed by model name; the scripted model reports none


# --- lessons 9.3-9.11 ----------------------------------------------------------------------------------------

HELPFUL = Helpfulness(helpful=True, reason="It answers.")
UNHELPFUL = Helpfulness(helpful=False, reason="It doesn't answer.")
ANGRY = Sentiment(label="negative", reason="'unacceptable'")
CALM = Sentiment(label="neutral", reason="A plain question.")


def judges() -> dict:
    """Plenty of judge verdicts for whole batches (FakeChatModel pops them in order)."""
    return {Sentiment: [CALM] * 60, Helpfulness: [HELPFUL] * 60}


def test_9_3_percentiles_breakdown_and_an_injected_incident(lesson, fake_retriever, paths):
    result = lesson("9.3").run(traffic_model(), fake_retriever, **paths)

    assert result["incident_error_rate"] == pytest.approx(2 / 7)  # turns 3 and 6 of 7 time out
    assert result["error_kinds"] == {"TimeoutError": 2}
    assert 0 < result["error_rate"] < 0.2
    assert set(result["percentiles"]) == {50, 95, 99} and result["percentiles"][50] <= result["percentiles"][99]
    assert any(name.startswith("llm:") for name in result["breakdown"])


def test_9_4_online_evaluators_sample_filter_and_write_feedback(lesson, fake_retriever, paths, tmp_path):
    feedback_path = tmp_path / "feedback.jsonl"
    llm = traffic_model(structured=judges() | {Sentiment: [ANGRY, CALM, ANGRY] + [CALM] * 20})

    result = lesson("9.4").run(llm, fake_retriever, feedback_path=feedback_path, helpfulness_sample=0.5, **paths)

    assert result["counts"]["user_sentiment"] == result["counts"]["cites_docs"] == 14
    assert 3 <= result["counts"]["helpfulness"] <= 11  # about half, sampled
    assert result["averages"]["cites_docs"] == 1.0  # every scripted answer cites a doc
    assert len(result["negative"]) == 2
    assert len(monitoring.load_rows(feedback_path)) == sum(result["counts"].values())


def test_9_5_rules_route_traces_and_alerts_fire(lesson, fake_retriever, paths, tmp_path):
    llm = FakeChatModel(default="Here's how to do it.")  # no doc citation: the dataset rule has work to do
    monitoring.simulate_traffic(  # some failures for the error rule and alert
        monitoring.build_monitored_agent(llm, fake_retriever),
        monitoring.load_conversations(limit=2),
        fault=lambda q: (_ for _ in ()).throw(TimeoutError("down")),
        **paths,
    )

    result = lesson("9.5").run(
        llm,
        fake_retriever,
        feedback_path=tmp_path / "f.jsonl",
        out_dir=tmp_path,
        alerts_path=tmp_path / "alerts.jsonl",
        **paths,
    )

    assert result["rules"]["errors to review"] == 3 and result["on_call"] == 3
    assert 0 < result["rules"]["uncited answers to a dataset"] < 17  # sampled at 50%
    assert len((tmp_path / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert "Error rate above 10%" in result["alerts"]


def test_9_6_pii_never_reaches_the_model_and_injections_are_blocked_and_tagged(lesson, fake_retriever, paths):
    result = lesson("9.6").run(traffic_model(), fake_retriever, **paths)

    assert len(result["blocked"]) == 2 and result["served"] == 2  # two injections refused before any model call
    assert not result["card_leaked"] and not result["email_leaked"]
    assert result["alerts"] == ["Prompt injection attempts"]


def test_9_7_ab_test_assigns_stably_and_compares_arms(lesson, fake_retriever, paths, tmp_path, hub):
    llm = traffic_model(structured=judges())
    result = lesson("9.7").run(llm, fake_retriever, feedback_path=tmp_path / "f.jsonl", hub=hub, **paths)

    assert sum(result["arms"].values()) == 12 and all(result["arms"].values())
    assert {r["helpful"] for r in result["results"].values()} == {r["traces"] for r in result["results"].values()}
    assert result["test"]["p_value"] == 1.0  # identical rates: no difference to find
    assert "variant-b" in hub.history(monitoring.PROMPT_NAME)[-1]["tags"]


def test_9_8_insights_summarise_threads_and_group_them(lesson, fake_retriever, paths):
    threads = [f"conv-{n:02d}/prod" for n in range(1, 13)]
    llm = traffic_model(
        structured={
            ConversationSummary: [ConversationSummary(topic="sync", outcome="unresolved", sentiment="negative")] * 12,
            Categories: [
                Categories(
                    categories=[
                        Category(name="Sync failures", description="Sync stops.", threads=threads[:8]),
                        Category(
                            name="Billing", description="Plans and charges.", threads=[*threads[8:], "conv-99/prod"]
                        ),
                    ]
                )
            ],
        }
    )
    result = lesson("9.8").run(llm, fake_retriever, **paths)

    assert result["threads"] == 12
    assert [(c["name"], c["count"]) for c in result["categories"]] == [("Sync failures", 8), ("Billing", 4)]
    assert result["categories"][0]["unresolved"] == 8  # invented thread ids are dropped


def test_9_9_dashboard_is_one_static_page(lesson, fake_retriever, paths, tmp_path):
    out = tmp_path / "dashboard.html"
    result = lesson("9.9").run(
        traffic_model(),
        fake_retriever,
        feedback_path=tmp_path / "f.jsonl",
        alerts_path=tmp_path / "a.jsonl",
        out=out,
        **paths,
    )

    page = out.read_text(encoding="utf-8")
    assert result["metrics"]["traces"] == 14
    for section in ("p95 latency", "By plan", "By variant", "Online evaluator scores", "Slowest traces", "Alerts"):
        assert section in page
    assert "<script" not in page and "http" not in page.replace("http-equiv", "")  # works offline


def test_poe_dashboard_task_writes_the_page():
    tasks = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["poe"]["tasks"]
    assert tasks["dashboard"]["script"] == "docshelp.dashboard:main"
    assert dashboard.DASHBOARD_PATH.parent.name == "reports"  # git-ignored with the other reports


FIX = PromptFix(
    diagnosis="Answers don't cite docs and ignore frustration.",
    changes=["Cite the doc file.", "Acknowledge frustration first."],
    new_prompt="You are DocsHelp. Cite the doc file you used. Acknowledge frustration first.",
)


def test_9_10_finds_failures_and_saves_a_candidate_not_prod(lesson, fake_retriever, paths, tmp_path, hub):
    llm = FakeChatModel(default="Try restarting.", structured=judges() | {PromptFix: [FIX]})

    result = lesson("9.10").run(llm, fake_retriever, feedback_path=tmp_path / "f.jsonl", hub=hub, **paths)

    assert result["failures"] == 14  # no answer cites a doc
    assert all("cites_docs=0" in reasons for reasons in result["reasons"])
    tags = {c["commit"]: c["tags"] for c in hub.history(monitoring.PROMPT_NAME)}
    assert tags[result["commit"]] == ["candidate"]
    assert monitoring.prompt_text(hub) == monitoring.SUPPORT_PROMPT_V1  # production untouched


def test_9_11_tests_ships_and_monitors_the_fix(lesson, fake_retriever, paths, tmp_path, hub):
    hub.push(monitoring.PROMPT_NAME, monitoring.prompt_template(monitoring.SUPPORT_PROMPT_V1), tags=["prod"])
    hub.push(monitoring.PROMPT_NAME, monitoring.prompt_template(FIX.new_prompt), tags=["candidate"])
    # Pairwise: the candidate (shown second, then first after the swap) wins every comparison.
    preferences = [Preference(reasoning="Cites docs.", preferred=p) for p in ["B", "A"] * 20]
    llm = traffic_model(structured=judges() | {Preference: preferences})

    result = lesson("9.11").run(llm, fake_retriever, feedback_path=tmp_path / "f.jsonl", hub=hub, **paths)

    assert result["shipped"], result["reason"]
    assert result["wins"]["new"] == result["questions"] and result["wins"]["old"] == 0
    assert monitoring.prompt_text(hub) == FIX.new_prompt  # "prod" moved to the candidate
    assert set(result["monitor"]) == {"before", "after"}


def test_9_11_does_not_ship_a_candidate_that_loses(lesson, fake_retriever, paths, tmp_path, hub):
    hub.push(monitoring.PROMPT_NAME, monitoring.prompt_template(monitoring.SUPPORT_PROMPT_V1), tags=["prod"])
    hub.push(monitoring.PROMPT_NAME, monitoring.prompt_template(FIX.new_prompt), tags=["candidate"])
    preferences = [Preference(reasoning="Old is better.", preferred=p) for p in ["A", "B"] * 20]
    llm = traffic_model(structured=judges() | {Preference: preferences})

    result = lesson("9.11").run(llm, fake_retriever, feedback_path=tmp_path / "f.jsonl", hub=hub, **paths)

    assert not result["shipped"] and "pairwise" in result["reason"]
    assert monitoring.prompt_text(hub) == monitoring.SUPPORT_PROMPT_V1
