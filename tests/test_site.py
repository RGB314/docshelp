"""The project website generator (src/docshelp/site.py): sources, links, lesson pages, and a strict build."""

import importlib.util

import pytest

from docshelp import site


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("site-src")
    config = site.generate(out)
    return out / "docs", config


def test_every_doc_and_part_gets_a_page(generated):
    docs, config = generated
    for page in ("index.md", "curriculum.md", "roadmap.md", "maintenance.md", "contributing.md"):
        assert (docs / page).is_file()
    parts = sorted(int(p.stem.split("-")[1]) for p in (docs / "lessons").glob("part-*.md"))
    assert parts == list(range(1, 10))
    text = config.read_text(encoding="utf-8")
    assert "lessons/part-9.md" in text and "Part 9: Production" in text


def test_lesson_pages_come_from_the_docstrings(generated):
    docs, _ = generated
    page = (docs / "lessons" / "part-1.md").read_text(encoding="utf-8")
    assert page.startswith("# Part 1: LangChain essentials")
    assert "## 1.6 " in page and "{#lesson-1-6}" in page
    assert "**Academy:**" in page and "```bash\nuv run poe lesson 1.6" in page
    assert f"{site.REPO}/blob/main/lessons/p1_langchain/06_tools_and_agents.py" in page


def test_links_point_at_site_pages_or_github():
    assert site.site_target("CURRICULUM.md#part-1-self-check") == "curriculum.md#part-1-self-check"
    assert site.site_target("lessons/p7_deep_agents/14_capstone_support_ops_analyst.py") == "lessons/part-7.md#lesson-7-14"
    assert site.site_target("src/docshelp/deep.py") == f"{site.REPO}/blob/main/src/docshelp/deep.py"
    assert site.site_target("skills/") == f"{site.REPO}/tree/main/skills"
    assert site.site_target("README.md", page_dir="lessons/") == "../index.md"
    text = site.rewrite_links("See `src/docshelp/deep.py`, `data/monitoring/traces.jsonl` and [x](https://a.b).")
    assert f"[`src/docshelp/deep.py`]({site.REPO}/blob/main/src/docshelp/deep.py)" in text
    assert "`data/monitoring/traces.jsonl`" in text and "[`data/monitoring" not in text  # generated: not linked
    assert "[x](https://a.b)" in text


def test_self_checks_become_collapsible_questions():
    html = "<details>\n<summary>1. What is <code>x</code>?</summary>\n\nIt is y.\nTwo lines.\n</details>"
    assert site.details_to_admonitions(html) == '??? question "1. What is `x`?"\n\n    It is y.\n    Two lines.\n'


def test_the_readme_loses_its_hand_made_contents_only(generated):
    docs, _ = generated
    home = (docs / "index.md").read_text(encoding="utf-8")
    assert "## Contents" not in home and "## How the pieces fit together" in home and "```mermaid" in home


@pytest.mark.skipif(importlib.util.find_spec("mkdocs") is None, reason="mkdocs-material is a dev dependency")
def test_the_site_builds_strictly(generated, tmp_path):
    _, config = generated
    assert site.mkdocs("build", "--strict", "-q", "-f", str(config), "-d", str(tmp_path / "site")) == 0
    assert (tmp_path / "site" / "lessons" / "part-8" / "index.html").is_file()
