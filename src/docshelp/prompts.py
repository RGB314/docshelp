"""Versioned prompts: LangSmith's Prompt Hub when configured, a local twin otherwise (Part 5.10).

A prompt hub stores every version (*commit*) of a prompt and lets you point *tags* such as "prod" at
one. Code pulls `"docshelp-rag:prod"` instead of hard-coding the prompt, so a prompt change is a
reviewed, evaluated, reversible release, not a code deploy. Part 9's improvement loop ships prompt
fixes this way.

  LangSmith:  Client().push_prompt(name, object=prompt, tags=[...]) / Client().pull_prompt("name:tag")
  local twin: LocalPromptHub(): the same push / pull / tag, stored as JSON in data/prompts/
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

from langchain_core._api import LangChainBetaWarning
from langchain_core.load import dumpd, load
from langchain_core.prompts import ChatPromptTemplate

from docshelp.config import DATA_DIR, langsmith_enabled

PROMPTS_DIR = DATA_DIR / "prompts"


class LocalPromptHub:
    """Prompt versions and tags in local JSON files, with the same identifiers as LangSmith ("name:tag")."""

    def __init__(self, root: Path | None = None):
        self.root = root or PROMPTS_DIR

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def _read(self, name: str) -> dict:
        path = self._path(name)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"commits": [], "tags": {}}

    def _write(self, name: str, data: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(name).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def push(
        self, name: str, prompt: ChatPromptTemplate, *, tags: list[str] | None = None, description: str = ""
    ) -> str:
        """Save a new version; returns its commit hash (unchanged prompts keep their hash)."""
        data = self._read(name)
        serialized = dumpd(prompt)
        commit = hashlib.sha1(json.dumps(serialized, sort_keys=True).encode()).hexdigest()[:8]
        if not any(c["commit"] == commit for c in data["commits"]):
            data["commits"].append(
                {"commit": commit, "prompt": serialized, "description": description,
                 "created_at": datetime.now(timezone.utc).isoformat()}
            )
        for tag in tags or []:
            data["tags"][tag] = commit
        self._write(name, data)
        return commit

    def tag(self, name: str, commit: str, tag: str) -> None:
        """Point a tag (e.g. "prod") at a commit: promotion and rollback are both just this."""
        data = self._read(name)
        if not any(c["commit"] == commit for c in data["commits"]):
            raise KeyError(f"{name} has no commit {commit}")
        data["tags"][tag] = commit
        self._write(name, data)

    def pull(self, identifier: str) -> ChatPromptTemplate:
        """"name" (latest), "name:<tag>" or "name:<commit>"."""
        name, _, ref = identifier.partition(":")
        data = self._read(name)
        if not data["commits"]:
            raise KeyError(f"no prompt named {name!r}")
        commit = data["tags"].get(ref, ref) if ref else data["commits"][-1]["commit"]
        entry = next((c for c in data["commits"] if c["commit"] == commit), None)
        if entry is None:
            raise KeyError(f"{identifier!r} matches no commit or tag")
        with warnings.catch_warnings():  # load is marked beta in langchain-core; the format is stable
            warnings.simplefilter("ignore", LangChainBetaWarning)
            return load(entry["prompt"], allowed_objects="core")  # only langchain-core classes, never arbitrary code

    def history(self, name: str) -> list[dict]:
        data = self._read(name)
        tags_by_commit: dict[str, list[str]] = {}
        for tag, commit in data["tags"].items():
            tags_by_commit.setdefault(commit, []).append(tag)
        return [
            {
                "commit": c["commit"],
                "description": c["description"],
                "tags": sorted(tags_by_commit.get(c["commit"], [])),
            }
            for c in data["commits"]
        ]


class LangSmithPromptHub:
    """The same interface over LangSmith's Prompt Hub (needs LANGSMITH_API_KEY)."""

    def __init__(self):
        from langsmith import Client

        self.client = Client()

    def push(
        self, name: str, prompt: ChatPromptTemplate, *, tags: list[str] | None = None, description: str = ""
    ) -> str:
        url = self.client.push_prompt(name, object=prompt, tags=tags or None, description=description or None)
        return url.rstrip("/").split("/")[-1].split(":")[-1]  # the commit hash at the end of the URL

    def tag(self, name: str, commit: str, tag: str) -> None:
        self.client.push_prompt(name, object=self.pull(f"{name}:{commit}"), tags=[tag])

    def pull(self, identifier: str) -> ChatPromptTemplate:
        return self.client.pull_prompt(identifier)

    def history(self, name: str) -> list[dict]:
        commits = self.client.list_prompt_commits(name)
        return [{"commit": c.commit_hash[:8], "description": "", "tags": []} for c in commits]


def prompt_hub(root: Path | None = None) -> LocalPromptHub | LangSmithPromptHub:
    return LangSmithPromptHub() if langsmith_enabled() else LocalPromptHub(root)
