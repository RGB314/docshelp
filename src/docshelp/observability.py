"""Tracing and feedback: LangSmith when it's configured, a local twin when it isn't (Part 5).

LangSmith records a *trace* (a tree of *runs*) for every LangChain/LangGraph call once
`LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set. Everything below uses the real `langsmith` SDK
in both modes:

  * with LangSmith: runs go to your project (`LANGSMITH_PROJECT`, default "default")
  * without it: the same SDK builds the same run trees in "local" mode, with a client that never sends
    anything, and they are written to `data/traces.jsonl`. Every lesson works with no account.

Either way a copy of each trace is saved locally, so Part 9's dashboards and insights can read it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langsmith import Client
from langsmith.run_helpers import trace, tracing_context
from langsmith.run_trees import RunTree

from docshelp.config import DATA_DIR, langsmith_enabled

TRACES_PATH = DATA_DIR / "traces.jsonl"
FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"


def offline_client() -> Client:
    """A LangSmith client that never sends anything (sampling rate 0, no server lookup).

    The URL is a placeholder that is never contacted.
    """
    return Client(
        api_url="http://127.0.0.1:9", api_key="local-only", info={}, tracing_sampling_rate=0.0, auto_batch_tracing=False
    )


def client() -> Client:
    """The real LangSmith client when configured, otherwise the offline one."""
    return Client() if langsmith_enabled() else offline_client()


@contextmanager
def tracing(*, metadata: dict | None = None, tags: list[str] | None = None) -> Iterator[None]:
    """Turn tracing on for a block: to LangSmith if configured, otherwise in memory only."""
    if langsmith_enabled():
        with tracing_context(enabled=True, metadata=metadata, tags=tags):
            yield
    else:
        with tracing_context(enabled="local", client=offline_client(), metadata=metadata, tags=tags):
            yield


@contextmanager
def traced(
    name: str,
    inputs: dict,
    *,
    run_type: str = "chain",
    metadata: dict | None = None,
    tags: list[str] | None = None,
    path: Path | None = None,
    save: bool = True,
) -> Iterator[RunTree]:
    """Trace a block as one root run; everything LangChain does inside becomes its children.

    Set the result with `run.end(outputs={...})`. The finished tree is appended to `path`
    (default: `TRACES_PATH`) unless `save=False`.
    """
    with tracing(metadata=metadata, tags=tags), trace(name, run_type=run_type, inputs=inputs) as run:
        yield run
    if save:
        save_trace(run, path or TRACES_PATH)


# --- reading run trees ----------------------------------------------------------------------------


def usage(run: RunTree) -> dict:
    """An LLM run's token usage as the chat model reported it: input_tokens, output_tokens, total_tokens."""
    if run.run_type != "llm" or not run.outputs:
        return {}
    found = run.outputs.get("usage_metadata")
    if found is None:
        try:
            found = run.outputs["generations"][0][0]["message"]["kwargs"].get("usage_metadata")
        except (KeyError, IndexError, TypeError):
            found = None
    return found or {}


def tokens(run: RunTree) -> int | None:
    """Total tokens of an LLM run, from the usage metadata the chat model reported."""
    return usage(run).get("total_tokens")


def latency_ms(run: RunTree) -> float | None:
    return round((run.end_time - run.start_time).total_seconds() * 1000, 1) if run.end_time else None


def _short(value: Any, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def flatten(run: RunTree, parent_id: str | None = None) -> list[dict]:
    """A run tree as rows, one per run, linked by `parent_id`: the shape LangSmith stores too."""
    metadata = (run.extra or {}).get("metadata", {})
    used = usage(run)
    row = {
        "trace_id": str(run.trace_id),
        "id": str(run.id),
        "parent_id": parent_id,
        "name": run.name,
        "run_type": run.run_type,
        "start": run.start_time.isoformat(),
        "latency_ms": latency_ms(run),
        "tokens": tokens(run),
        "input_tokens": used.get("input_tokens"),
        "output_tokens": used.get("output_tokens"),
        "model": metadata.get("ls_model_name"),  # LangSmith prices runs by this name (Part 9)
        "error": run.error,
        "tags": list(run.tags or []),
        "metadata": {k: v for k, v in metadata.items() if not k.startswith(("ls_", "langgraph_", "checkpoint"))},
        "inputs": _short(run.inputs),
        "outputs": _short(run.outputs),
    }
    return [row] + [r for child in run.child_runs for r in flatten(child, str(run.id))]


def print_tree(run: RunTree, depth: int = 0, *, max_depth: int = 6) -> None:
    if depth > max_depth:
        return
    extra = f"  {tokens(run)} tokens" if tokens(run) else ""
    error = f"  ERROR: {run.error[:60]}" if run.error else ""
    print(f"  {'  ' * depth}{run.run_type:9} {run.name:24} {latency_ms(run) or 0:>8.1f} ms{extra}{error}")
    for child in run.child_runs:
        print_tree(child, depth + 1, max_depth=max_depth)


def total_tokens(run: RunTree) -> int:
    return (tokens(run) or 0) + sum(total_tokens(c) for c in run.child_runs)


# --- the local trace log --------------------------------------------------------------------------


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def save_trace(run: RunTree, path: Path | None = None) -> None:
    append_rows(path or TRACES_PATH, flatten(run))


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def thread_id(row: dict) -> str | None:
    """LangSmith groups traces into a *thread* by any of these metadata keys."""
    metadata = row.get("metadata", {})
    return metadata.get("thread_id") or metadata.get("session_id") or metadata.get("conversation_id")


def summarize_threads(rows: list[dict]) -> dict[str, dict]:
    """Per conversation thread: turns (root runs), tokens and total latency, as LangSmith's Threads view."""
    by_trace: dict[str, list[dict]] = {}
    for row in rows:
        by_trace.setdefault(row["trace_id"], []).append(row)
    threads: dict[str, dict] = {}
    for trace_rows in by_trace.values():
        root = next(r for r in trace_rows if r["parent_id"] is None)
        key = thread_id(root) or "(no thread)"
        summary = threads.setdefault(key, {"turns": 0, "tokens": 0, "latency_ms": 0.0, "errors": 0})
        summary["turns"] += 1
        summary["tokens"] += sum(r["tokens"] or 0 for r in trace_rows)
        summary["latency_ms"] += root["latency_ms"] or 0
        summary["errors"] += sum(1 for r in trace_rows if r["error"])
    return threads


# --- feedback -------------------------------------------------------------------------------------


def record_feedback(
    run_id: str, key: str, score: float | bool, comment: str | None = None, *, path: Path | None = None
) -> dict:
    """Attach a score to a run: in LangSmith if configured, and always in the local feedback log."""
    record = {
        "run_id": str(run_id),
        "key": key,
        "score": float(score),
        "comment": comment,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    if langsmith_enabled():
        Client().create_feedback(run_id, key=key, score=float(score), comment=comment)
    append_rows(path or FEEDBACK_PATH, [record])
    return record
