"""User interfaces for a deployment: Agent Chat UI, Studio, and your own UI on the raw HTTP API.

Academy: Introduction to LangChain M3 L6 "Agent Chat UI"; Introduction to LangSmith Deployment M2 "Connecting
the UI", M3 "UI Walkthrough", M6 "Bring Your Own UI".

Concepts:
  * Agent Chat UI (open source, Next.js) chats with any deployed graph whose state has `messages`. Either
    open https://agentchat.vercel.app, or run it locally: `npx create-agent-chat-app` (needs Node.js), then
    enter Deployment URL `http://127.0.0.1:2024` and Assistant/Graph ID `support_agent`. It streams tokens,
    shows tool calls, lets you edit and regenerate, and shows interrupts (e.g. the `docshelp` graph's ticket
    approval) as forms. A deployed server also needs a LangSmith API key in its settings.
  * UI walkthrough, LangSmith Studio (`uv run poe server` prints its URL): pick a graph, chat in Chat mode
    or send state in Graph mode, watch each node run, inspect and edit state, fork from any checkpoint
    (Part 3.12, 3.14), manage assistants (8.5) and threads, and see interrupts as buttons.
  * Agent Inbox (open source) is the inbox-style UI for ambient agents' interrupts (lesson 8.12).
  * bring your own UI: every UI above uses the same HTTP API, so any frontend can too. This lesson makes the
    calls by hand with `httpx`:
      - `POST /threads`
      - `POST /threads/{id}/runs/stream` with `stream_mode: ["messages-tuple", "updates"]`
      - read the server-sent events (`event:` and `data:` lines, blank line between events)
    and `GET /ui` (8.6) serves a page that does the same in 80 lines of plain JavaScript. React apps use
    `useStream()` from `@langchain/langgraph-sdk/react`, which wraps these calls.

Run:  uv run poe server, then: uv run poe lesson 8.11      (and open http://127.0.0.1:2024/ui)
"""

import argparse
import json

import httpx

from docshelp.config import ensure_llm_ready
from docshelp.console import header
from docshelp.deployment import connect, server_url

QUESTION = "How do I turn on two-factor authentication?"


def read_events(response: httpx.Response):
    """Parse server-sent events: `event: <name>` and `data: <json>` lines, events separated by a blank line."""
    event, data = None, []
    for line in response.iter_lines():
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].strip())
        elif not line and event:
            yield event, json.loads("\n".join(data)) if data else None
            event, data = None, []


def chat_over_http(url: str, question: str, assistant: str = "support_agent") -> dict:
    """What a UI does for one message, with no SDK: create a thread, stream a run, render the events."""
    with httpx.Client(base_url=url, timeout=600) as http:
        thread = http.post("/threads", json={}).json()
        body = {
            "assistant_id": assistant,
            "input": {"messages": [{"role": "user", "content": question}]},
            "stream_mode": ["messages-tuple", "updates"],
        }
        events, tokens, steps = {}, [], []
        with http.stream("POST", f"/threads/{thread['thread_id']}/runs/stream", json=body) as response:
            for event, data in read_events(response):
                events[event] = events.get(event, 0) + 1
                if event == "messages":  # (chunk, metadata): a token, or a whole tool message
                    chunk, metadata = data
                    if chunk.get("type") in ("AIMessageChunk", "ai") and isinstance(chunk.get("content"), str):
                        tokens.append(chunk["content"])
                        print(chunk["content"], end="", flush=True)
                elif event == "updates":  # which node finished: what a UI shows as "Searching docs..."
                    steps += list(data)
    print()
    return {"thread_id": thread["thread_id"], "events": events, "answer": "".join(tokens), "steps": steps}


def run(client, url: str) -> dict:
    header("Bring your own UI: one chat turn over plain HTTP + server-sent events")
    print(f"Customer: {QUESTION}\nDocsHelp: ", end="")
    result = chat_over_http(url, QUESTION)
    print(f"\nEvents received: {result['events']}; nodes: {result['steps']}")
    state = client.threads.get_state(result["thread_id"])
    print(f"The thread is stored like any other: {len(state['values']['messages'])} messages (SDK read).")

    header("The ready-made UIs")
    print(f"  Your page:      {url}/ui")
    print(f"  Agent Chat UI:  https://agentchat.vercel.app  (Deployment URL {url}, Graph ID support_agent)")
    print(f"  Studio:         https://smith.langchain.com/studio/?baseUrl={url}")
    print("  Agent Inbox:    https://dev.agentinbox.ai  (lesson 8.12's interrupts)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=server_url(), help="the LangGraph Server, e.g. http://127.0.0.1:2024")
    args = parser.parse_args()
    ensure_llm_ready()
    run(connect(args.url), args.url)


if __name__ == "__main__":
    main()
