"""LangGraph Studio: see, run and debug your graphs in a visual IDE.

Academy: Intro to LangGraph M1 L3 "LangSmith Studio"; LangSmith Essentials M1 "Debugging with Studio".

Concepts:
  * `langgraph.json` tells the LangGraph CLI which graphs to serve: here the support graph (`docshelp`,
    the Part 3 capstone) and the hand-built agent (`react_agent`, lesson 3.3), both from `server.py`
  * `uv run poe server` runs `langgraph dev`: a local LangGraph Server (API on http://127.0.0.1:2024) with
    its own persistence, and opens Studio, a web UI that connects to that local server. Studio is free;
    you sign in with a free LangSmith account. Your graph, data and model calls stay on your machine.
  * in Studio you can: chat with a graph, watch each node run, open each step's state, edit state and
    re-run from any step (time travel, lesson 3.14), and approve interrupts (lessons 3.12–3.15)
  * the server reloads when you change the code, so Studio is a fast debugging loop
  * Part 8 uses the same server through its API, the SDK and `RemoteGraph`

This lesson prints the graphs Studio will show, then explains how to open it.

Run:  uv run poe lesson 3.4, then  uv run poe server
"""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from docshelp.config import PROJECT_ROOT, describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.ingest import get_retriever
from docshelp.react import build_react_agent
from docshelp.tools import make_search_docs, quote_price

STUDIO_URL = "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"


def run(llm: BaseChatModel, retriever: BaseRetriever) -> dict:
    config = json.loads((PROJECT_ROOT / "langgraph.json").read_text(encoding="utf-8"))
    header("Graphs in langgraph.json")
    for name, target in config["graphs"].items():
        print(f"  {name:12} -> {target}")

    header("What Studio will draw for react_agent")
    mermaid = build_react_agent(llm, [make_search_docs(retriever), quote_price]).get_graph().draw_mermaid()
    print(mermaid)

    header("Open Studio")
    print(
        "1. In one terminal:  uv run poe server     (Docker: docker compose up -d server)\n"
        f"2. Open: {STUDIO_URL}\n"
        "   (sign in with a free LangSmith account if asked; your graphs still run locally)\n"
        "3. Pick a graph at the top left, type a message, and watch the nodes light up.\n"
        "4. Try: open a step's state, edit it, and re-run from there (forking the thread).\n"
        "5. Change a prompt in src/docshelp/react.py and save: the server reloads, Studio picks it up."
    )
    return {"graphs": list(config["graphs"]), "mermaid": mermaid}


def main() -> None:
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever())


if __name__ == "__main__":
    main()
