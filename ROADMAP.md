# ROADMAP.md

The build plan for turning DocsHelp into a complete, zero-cost equivalent of the
[LangChain Academy](https://academy.langchain.com/collections) catalogue (17 courses, checked 2026-10-10).
`CURRICULUM.md` maps each Academy lesson to the DocsHelp lesson that covers it. This file covers what is still
to be built, in what order, and how to pick the work up in a new session.

## Status

| Part | Topic | Status |
|---|---|---|
| 1 | LangChain essentials | ✅ Merged (PR #4) |
| 2 | Production-ready agents with LangChain | ✅ Shipped (PR for branch `feat/curriculum-part2-production-agents`) |
| 3 | LangGraph core (absorbs `stages/03`) | ⏭️ Next |
| 4 | LangGraph at scale + long-term memory | Planned |
| 5 | LangSmith observability and evaluation (absorbs `stages/04`) | Planned |
| 6 | Agent projects: ambient inbox and deep research | Planned |
| 7 | Deep Agents (plus dcode, Fleet and MDA twins) | Planned |
| 8 | Deployment (absorbs `stages/05`) | Planned |
| 9 | Production monitoring and the improvement loop | Planned |

Update this table in the same PR that ships a part.

## The maintainer's requirements

These come from the maintainer and are non-negotiable.

- **Cover everything; repeat nothing.** Every concept, feature and idea in the Academy courses has to be
  covered. Each one is taught once, at the point where it's first needed, and afterwards only built on.
- **Use one real-world use case.** Apply every concept to the Nimbus Notes support app. Don't use the courses'
  toy projects (chef, wedding planner and so on). The goal is to learn LangChain and its products as they're
  used in real work.
- **Zero cost:**
  - No paid APIs, and especially no OpenAI.
  - Ollama is the default provider. Groq and Gemini free tiers are optional for the tool-heavy parts.
  - Use the free LangSmith Developer account where it helps. Each paid-only feature (Engine, Fleet, Insights,
    Dashboards, cloud Deployment, Managed Deep Agents) gets a small **local twin** that teaches the same concept.
  - Every lesson also runs with no LangSmith account.
- **One PR per part,** in order, each from its own branch off `main`. CI must be green before the next part
  starts.

## How to resume in a new session

1. Read `AGENTS.md` for the rules, `CURRICULUM.md` for what's covered, and this file.
2. Run `git checkout main`, then `git pull`, then `git checkout -b feat/curriculum-partN-<topic>`.
3. Work test-first:
   - Add `tests/test_partN_<topic>.py`, driven by `tests/conftest.py::FakeChatModel` (and `tool_call()`).
   - Write the lessons in `lessons/pN_<topic>/NN_<slug>.py`. Use Part 1's lessons as the template.
   - Put reusable code in `src/docshelp/`.
4. In the same PR:
   - Replace every "Part N (planned)" row in `CURRICULUM.md` with a lesson path.
   - Add a `### Part N self-check` section to `CURRICULUM.md`.
   - Add a README lesson table.
   - Update the e2e lists in `tests/e2e/`. Keyless runs need no change because lessons are discovered
     automatically. If a lesson can't run on a tiny model, exclude it from `TEXT_ONLY_LESSONS`.
   - Update the status table above.
5. Run `uv run poe test` and `uv run poe e2e`. Both must pass. Then open the PR and wait for all four CI checks.
   The `docker-e2e` check is the only one that runs lessons against a real model (`qwen2.5:0.5b`).

Practical notes:
- In PowerShell, write commit and PR messages to a temp file and pass it with `git commit -F <file>`.
  Here-strings piped to `-F -` fail.
- `load_lesson()` executes the module again on each call. Load it once per test when you compare classes.
- `FakeChatModel` pops replies in call order. Parallel branches pop in a non-deterministic order, so assert on sets.

## Plan per part

Each item is one lesson: a small runnable script. Brackets give the DocsHelp use case. Concepts shared between
courses are taught once and then deepened. HITL goes from Part 2 (middleware) to Part 3 (raw `interrupt()`) to
Part 7 (`interrupt_on`). MCP goes from Part 2 to Part 6 to Part 7. Memory goes from Part 1 to Part 3 to Part 4
to Part 7.

### Part 2: Production-ready agents (Intro to LangChain M2–M3)
Shipped as built. Dependencies added: `langchain-mcp-adapters`, `mcp`.
1. MCP: a stdio server, `src/docshelp/mcp_server.py`, provides tools, a resource and a prompt.
2. Context and custom state: `ToolRuntime`, `Command(update=...)`.
3. Multi-agent: a supervisor delegates to docs, billing and MCP status specialists (`team.py`).
4. Middleware: all six hooks in `AuditLog` (`middleware.py`), plus a `before_agent` guard with `jump_to`.
5. Prebuilt middleware: tool and model retry, call limits and model fallback.
6. Long conversations: summarization, trimming with `RemoveMessage`, and `ContextEditingMiddleware`.
7. HITL: approve, edit or reject on `open_ticket`.
8. Dynamic agents: the model is picked by difficulty and the tools by plan.
9. Capstone: the email assistant (`email_assistant.py`) with a verification gate, dynamic tools from state,
   HITL on `send_reply`, summarization, a call limit and the audit log.

Agent Chat UI is covered in Part 8.

Shared code that later parts should reuse, rather than re-teach:
- `customers.py`: the `Customer` context and the `ACCOUNTS` data
- `tools.py`: `my_account`, `open_ticket`, `send_reply`
- `middleware.py`: `AuditLog`
- `FakeChatModel`, which now also takes exceptions as replies and records the tools offered to each call in `.bound`

Part 3 must show the LangGraph layer *under* these features, not repeat them:
- message trimming and summarization in graph nodes
- `interrupt()` in a node rather than the HITL middleware

### Part 3: LangGraph core (LangGraph Essentials; Intro to LangGraph M1–M3)
Replaces `stages/03`. `graph.py` stays the capstone. New dependency: `langgraph-checkpoint-sqlite`.
1. Nodes and edges.
2. Conditional edges and routers.
3. A ReAct agent built by hand.
4. Studio (`langgraph dev`, free account).
5. State schemas: TypedDict, dataclass and Pydantic.
6. Reducers.
7. Multiple schemas: input, output and private.
8. Trimming and filtering messages.
9. A summarizing chatbot.
10. `SqliteSaver`, so conversations survive restarts.
11. Stream modes: values, updates, messages and custom.
12. Breakpoints, plus editing state and human feedback.
13. Dynamic interrupts.
14. Time travel: replay and fork.

### Part 4: LangGraph at scale + long-term memory (Intro to LangGraph M4–M5)
1. Parallel fan-out and fan-in [docs, changelog and status page].
2. Subgraphs [a ticket-handling subgraph].
3. Map-reduce with `Send` [summarize each retrieved doc].
4. A research assistant [generates a FAQ article from the docs].
5. Short-term versus long-term memory.
6. Store: `InMemoryStore`, the `PostgresStore` concept, and semantic search with fastembed.
7. A memory schema as a profile [plan, OS, devices].
8. A memory schema as a collection [past issues].
9. Capstone: DocsHelp remembers the customer across threads.

### Part 5: LangSmith (Observability & Evals; Building Reliable Agents; LangSmith Essentials M1–M2)
Replaces `stages/04`. New dependency: `agentevals`.
1. Tracing basics.
2. Run types.
3. Other ways to trace: `@traceable`, `wrap_*`, the context manager and OpenTelemetry.
4. Threads, metadata and tags.
5. Datasets: create, version and split.
6. Evaluators:
   - code-based
   - LLM-as-judge
   - trajectory (`agentevals`)
7. Experiments and how to analyse them.
8. Pairwise evaluation.
9. Summary evaluators.
10. The pytest plugin, for evals as tests.
11. Prompt Hub push and pull, the Playground, and versioned prompts.
12. Feedback (`create_feedback`).
13. Annotation queues.

Local twins:
- a JSONL trace recorder (`src/docshelp/observability.py`)
- a local experiment runner with a comparison table
- a local pairwise runner

### Part 6: Agent projects (Project: Ambient Agents; Project: Deep Research)
New optional extra: `web = ["ddgs"]`.
1. Ambient support-inbox agent:
   - It runs on events from `data/inbox/*.json`, not from a chat.
   - A triage router decides: respond, notify or ignore.
   - A response agent drafts the reply.
   - Agent-Inbox-style interrupts let a human accept, edit, respond or ignore.
   - Memory learns triage and tone preferences from the human's edits.
   - A triage-accuracy eval.
2. Deep research [a competitive and product research report about Nimbus Notes]:
   - Scoping: clarifying questions, then a brief.
   - A research agent with a think tool and context compression.
   - The same agent over MCP.
   - A supervisor with parallel researchers.
   - Writing the report.
   - A report-quality eval.
   - Web search uses the optional `ddgs` extra and falls back to the local docs offline.

### Part 7: Deep Agents (Intro to Deep Agents; dcode Essentials; LangSmith Fleet)
New dependency: `deepagents>=0.7.21`. Add it to `pyproject.toml` dependencies, then run `uv lock`.
1. `create_deep_agent`: overview and models.
2. The system prompt.
3. Tools, MCP, threads and HITL through `interrupt_on`.
4. Filesystem backends: State, Filesystem, Store and Composite.
5. Sandbox and local shell, restricted to a temp dir, with an explicit safety lesson.
6. The interpreter.
7. Summarization and offloading large tool results.
8. Skills: `skills/*/SKILL.md` [a refund-policy skill and a log-analysis skill].
9. `AGENTS.md`-style memory.
10. Subagents: a team, dynamic subagents and async subagents.

Capstone: a "Support Ops analyst" deep agent reads `tickets.jsonl` and the docs, then writes an incident report.

Twins:
- **dcode:** run the open-source `deepagents` CLI against this repo.
- **Fleet:** agents from a YAML or Markdown template.
- **Managed Deep Agents:** explained only.

### Part 8: Deployment (Intro to LangGraph M6; Intro to LangSmith Deployment; LangSmith Essentials M3)
Replaces `stages/05`.
1. Concepts, plus `langgraph dev`.
2. Connecting with the SDK and `RemoteGraph`.
3. Threads, runs and streaming.
4. Double texting: reject, enqueue, interrupt and rollback.
5. Assistants [a "concise" and a "detailed" configuration].
6. Custom routes through `http.app` [`/health`, `/tickets`].
7. A Store `index` that uses fastembed through a `path:embed` function.
8. Auth with `@auth.authenticate` and `@auth.on` [customers see only their own threads].
9. The Part 7 deep agent as a second graph.
10. Agent Chat UI against localhost.
11. Docker (`compose.yaml`).

Cloud deployment is explained in the README only.

### Part 9: Monitoring and the improvement loop (Monitoring Production Agents; LangSmith Engine; Essentials M4–M5)
1. Cost: `UsageMetadataCallbackHandler` plus a pricing table.
2. Latency and errors.
3. Sentiment as an online evaluator: a local twin plus the free LangSmith version.
4. A/B testing by metadata.
5. PII and prompt-injection guard middleware, with threshold alerts.
6. Automations and rules.

Twins:
- **Insights:** an LLM clusters the topics in the local trace log.
- **Dashboards:** `uv run poe dashboard` writes a static HTML report.
- **Engine / the agent development lifecycle (ADLC) loop:**
  1. Find the failing traces.
  2. An LLM proposes a prompt fix.
  3. An experiment compares the old prompt with the new one.
  4. Ship the fix as a new prompt version.
  5. Monitor it.

### Exam
The LCAE Practice Exam is covered by the `### Part N self-check` sections in `CURRICULUM.md`.
The TypeScript editions are out of scope.

## Verification for every PR
- `uv run poe test` and `uv run poe e2e` pass locally. Report the results.
- All four CI checks pass.
- Do a manual smoke test with Ollama when it's available, and with a Groq free key for Parts 6–7.
- For Part 8:
  1. Start `uv run poe server`.
  2. Run the SDK lessons.
  3. Point Agent Chat UI at `http://127.0.0.1:2024`.
- For Parts 5 and 9, run once without a LangSmith key and once with a free key.
