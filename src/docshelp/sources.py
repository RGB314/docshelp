"""Knowledge sources besides the docs: the changelog, the status page and past tickets (Part 4).

Real support agents combine several sources. These are deliberately simple (keyword matching over local
files) so the lessons can focus on how the graph combines them, not on search quality.
"""

from __future__ import annotations

import json
import re

from docshelp.config import DATA_DIR
from docshelp.mcp_server import STATUS

CHANGELOG_PATH = DATA_DIR / "changelog.md"
SAMPLE_TICKETS_PATH = DATA_DIR / "sample_tickets.jsonl"
STOPWORDS = {"the", "a", "an", "is", "are", "my", "on", "in", "of", "to", "and", "for", "it", "i", "do", "how", "why"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def search_changelog(query: str, k: int = 3) -> list[str]:
    """Changelog bullet points sharing the most words with the query, each prefixed with its release."""
    entries, release = [], ""
    for line in CHANGELOG_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            release = line[3:].strip()
        elif line.startswith("- "):
            entries.append(f"[{release}] {line[2:].strip()}")
        elif line.startswith("  ") and entries:
            entries[-1] += " " + line.strip()
    wanted = _words(query)
    scored = sorted(((len(wanted & _words(e)), e) for e in entries), key=lambda pair: -pair[0])
    return [entry for score, entry in scored[:k] if score > 0]


def status_summary() -> str:
    return "; ".join(f"{region}: {status}" for region, status in STATUS.items())


def load_sample_tickets() -> list[dict]:
    return [json.loads(line) for line in SAMPLE_TICKETS_PATH.read_text(encoding="utf-8").splitlines() if line]
