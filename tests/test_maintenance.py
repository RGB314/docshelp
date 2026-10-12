"""`uv run poe maintenance`: the parsers and the report, offline (no network, `gh` or `uv` calls)."""

import json

from docshelp import maintenance

CATALOGUE_HTML = """
<a href="/courses/intro-to-langgraph">Intro to LangGraph</a>
<a href="https://academy.langchain.com/courses/brand-new-course?ref=nav">New</a>
<a href="/courses/intro-to-langgraph">again</a>
"""

CI_LOG = """
job\t2026-10-12T11:47:47Z tests/test_part6_agent_projects.py ............ [ 26%]
job\t2026-10-12T11:52:51Z Timeout (0:05:00)!
job\t2026-10-12T11:52:51Z Thread 0x00007f4b73fff6c0 (most recent call first):
job\t2026-10-12T11:52:51Z   File "/usr/lib/python3.12/threading.py", line 355 in wait
job\t2026-10-12T11:52:51Z   File "/ci/work/docshelp/lessons/p7_deep_agents/14_capstone.py", line 59 in turn
job\t2026-10-12T11:52:51Z   File "/ci/work/docshelp/tests/test_part7_deep_agents.py", line 467 in test_7_14_capstone
"""


def test_catalogue_slugs_and_new_courses():
    assert maintenance.course_slugs(CATALOGUE_HTML) == ["brand-new-course", "intro-to-langgraph"]
    assert maintenance.new_courses(CATALOGUE_HTML, {"intro-to-langgraph": "Intro"}) == ["brand-new-course"]


def test_the_catalogue_registry_lists_mapped_courses():
    registry = json.loads(maintenance.CATALOGUE_PATH.read_text(encoding="utf-8"))
    curriculum = (maintenance.PROJECT_ROOT / "CURRICULUM.md").read_text(encoding="utf-8")
    for title in registry["courses"].values():
        assert f"### {title}" in curriculum, f"{title} has no section in CURRICULUM.md"


def test_catalogue_check_reports_a_network_failure_instead_of_crashing():
    def offline(url):
        raise OSError("no network")

    result = maintenance.check_catalogue(fetch=offline)
    assert not result["ok"] and "no network" in result["note"]


def test_dependency_updates_are_parsed_from_uv_output():
    output = "Resolved 151 packages in 1.7s\nUpdate langgraph v1.2.12 -> v1.2.14\nUpdate pyjwt v2.15.0 -> v2.15.1\n"
    assert maintenance.parse_updates(output) == ["langgraph v1.2.12 -> v1.2.14", "pyjwt v2.15.0 -> v2.15.1"]


def test_hang_dumps_name_the_stuck_test():
    assert maintenance.hang_dumps(CI_LOG) == ["tests/test_part7_deep_agents.py::test_7_14_capstone"]
    assert maintenance.hang_dumps("all green\n") == []


def test_the_curriculum_is_consistent():
    assert maintenance.check_curriculum() == {"ok": True, "unmapped": [], "planned": 0}


def test_report_and_main(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(maintenance, "REPORT_PATH", tmp_path / "maintenance.md")
    monkeypatch.setattr(maintenance, "check_catalogue", lambda: {"ok": True, "note": "", "new": ["x"], "seen": 7})
    monkeypatch.setattr(maintenance, "check_dependencies", lambda: {"ok": True, "note": "", "updates": ["a v1 -> v2"]})

    maintenance.main(["--no-ci"])

    text = (tmp_path / "maintenance.md").read_text(encoding="utf-8")
    assert "https://academy.langchain.com/courses/x" in text and "- a v1 -> v2" in text
    assert "Not checked: skipped (--no-ci)" in text and "every lesson is mapped" in text
    assert "Saved to" in capsys.readouterr().out
