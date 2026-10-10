"""Deep research, step 2: a research agent that searches, reflects, and compresses what it found.

Academy: Project: Deep Research M3 "Research Agent".

Concepts (`src/docshelp/deep_research.py::build_researcher`):
  * a tool loop like lesson 3.3, tuned for research: several focused searches beat one broad one
  * a `think` tool: the model writes down what it learned and what's missing before deciding to search
    again or stop. It does nothing but record the reflection, and that is enough to make the loop more
    deliberate. (The same trick works in any agent.)
  * a budget: at most a few rounds of tool calls, then it stops, however thorough it wants to be
  * compression: raw search results are long and repetitive. A final step distils them into compact,
    cited findings for whoever reads them next (the supervisor in 6.8). Raw notes are kept separately.
  * sources: the docs and the changelog; `--web` adds free DuckDuckGo search (`uv sync --extra web`)

Run:  uv run poe lesson 6.6           (offline sources)
      uv run poe lesson 6.6 --web     (plus web search)
"""

import argparse

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

from docshelp.config import describe, ensure_llm_ready, get_llm
from docshelp.console import header
from docshelp.deep_research import build_researcher, make_research_tools, research_input
from docshelp.ingest import get_retriever

TOPIC = "How well does Nimbus Notes work offline on desktop, mobile and web, and what changed recently?"


def run(llm: BaseChatModel, retriever: BaseRetriever, *, web: bool = False) -> dict:
    researcher = build_researcher(llm, make_research_tools(retriever, web=web))
    header(f"Topic: {TOPIC}")
    steps = []
    result: dict = {}
    for mode, chunk in researcher.stream(research_input(TOPIC), stream_mode=["updates", "values"]):
        if mode == "updates":
            for node, update in chunk.items():
                for message in (update or {}).get("messages", []):
                    if isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            steps.append(call["name"])
                            print(f"  {call['name']}: {next(iter(call['args'].values()), '')}"[:140])
        else:
            result = chunk
    header("Compressed findings")
    print(result["compressed_research"])
    return {"steps": steps, "compressed_research": result["compressed_research"], "raw_notes": result["raw_notes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", action="store_true", help="also search the web (needs `uv sync --extra web`)")
    args = parser.parse_args()
    ensure_llm_ready()
    print(f"Using {describe()}")
    run(get_llm(), get_retriever(), web=args.web)


if __name__ == "__main__":
    main()
