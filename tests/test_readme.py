"""The README's table of contents must stay in sync with its headings."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
TOC_START, TOC_END = "<!-- toc -->", "<!-- tocstop -->"


def github_anchor(heading: str) -> str:
    """GitHub's heading-anchor rule: lowercase, drop punctuation (keep hyphens/underscores), spaces -> hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def headings(level_pattern: str) -> list[str]:
    body = re.sub(r"```.*?```", "", README, flags=re.S)  # ignore '#' lines inside code blocks
    return re.findall(rf"^{level_pattern} (.+)$", body, flags=re.M)


def toc_links() -> list[str]:
    assert TOC_START in README and TOC_END in README, "README needs a <!-- toc --> ... <!-- tocstop --> block"
    block = README.split(TOC_START, 1)[1].split(TOC_END, 1)[0]
    return re.findall(r"\]\(#([^)]+)\)", block)


def test_every_toc_link_points_at_a_real_heading():
    anchors = {github_anchor(h) for h in headings("#{2,3}")}
    broken = [link for link in toc_links() if link not in anchors]
    assert broken == []


def test_every_section_is_listed_in_the_toc():
    sections = [h for h in headings("##") if h != "Contents"]  # the ToC doesn't list itself
    missing = [h for h in sections if github_anchor(h) not in toc_links()]
    assert missing == []


def test_anchor_rule_matches_github():
    assert github_anchor("Configure `.env` (all options)") == "configure-env-all-options"
    assert github_anchor("Option C: Dev Container / GitHub Codespaces") == "option-c-dev-container--github-codespaces"
