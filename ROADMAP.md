# ROADMAP.md

The build plan for turning DocsHelp into a complete, zero-cost equivalent of the
[LangChain Academy](https://academy.langchain.com/collections) catalogue (17 courses, checked 2026-10-10).
`CURRICULUM.md` maps each Academy lesson to the DocsHelp lesson that covers it. This file covers what is still
to be built, in what order, and how to pick the work up in a new session.

## Status

| Part | Topic | Status |
|---|---|---|
| 1 | LangChain essentials | ✅ Merged (PR #4) |
| 2 | Production-ready agents with LangChain | ✅ Merged (PR #6) |
| 3 | LangGraph core (absorbs `stages/03`) | ✅ Merged (PR #7) |
| 4 | LangGraph at scale + long-term memory | ✅ Merged (PR #8) |
| 5 | LangSmith observability and evaluation (absorbs `stages/04`) | ✅ Merged (PR #9) |
| 6 | Agent projects: ambient inbox and deep research | ✅ Merged (PR #10) |
| 7 | Deep Agents (plus dcode, Fleet and MDA twins) | ✅ Merged (PR #11) |
| 8 | Deployment (absorbs `stages/05`) | ✅ Merged (PR #12) |
| 9 | Production monitoring and the improvement loop | ✅ Shipped (branch `feat/curriculum-part9-monitoring`) |

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
- Long sync runs with a checkpointer (deep agents with subagents) can deadlock in LangGraph 1.2 on small CI runners:
  async-durability checkpoint saves fill the thread pool waiting on each other. Use `durability="sync"` there
  (lesson 7.14). `faulthandler_timeout` in pyproject dumps the stacks of any test stuck for 5 minutes.

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
Shipped as built. Replaced `stages/03`. Dependency added: `langgraph-checkpoint-sqlite`.
1. State, nodes and edges.
2. Conditional edges vs `Command(goto=...)`.
3. A ReAct agent by hand (`react.py`).
4. Studio: `langgraph.json` now serves `docshelp` and `react_agent`.
5. State schemas. Pydantic validates input only.
6. Reducers: `InvalidUpdateError`, a custom reducer, and replacing or removing with `add_messages`.
7. Input, output and private schemas.
8. Trimming and filtering.
9. A summarizing chatbot (`chatbot.py`).
10. `SqliteSaver` across "restarts".
11. Stream modes and `astream_events`.
12. Breakpoints, `update_state` and `as_node`.
13. `interrupt()` with a validation loop and conditional approval.
14. Time travel: replay and fork.
15. Capstone: `graph.py`, now with a `RetryPolicy` on `retrieve` and the four error kinds from "Thinking in LangGraph".

Reuse in later parts:
- `build_react_agent`
- `build_summarizing_chatbot`
- `build_graph`

Parallel branches appeared only briefly, in 3.6 (reducers). Part 4 teaches them properly, with fan-out/fan-in and `Send`.

### Part 4: LangGraph at scale + long-term memory (Intro to LangGraph M4–M5)
Shipped as built. Dependency added: `trustcall`. New data: `data/changelog.md` and
`data/sample_tickets.jsonl` (read through `sources.py`).
1. Parallel fan-out and fan-in over docs, changelog and status, with a list join.
2. Two subgraphs with shared and private keys and `output_schema`; streamed with `subgraphs=True`.
3. Map-reduce with `Send`: one note per document.
4. Research assistant (`research.py`): HITL perspectives, interview subgraphs, parallel writers.
5. The Store, cross-thread memory, and `create_agent(store=...)`.
6. Semantic store search with fastembed, plus filters and `index=False`.
7. Profile memory patched with trustcall.
8. Collection memory with inserts and updates.
9. Capstone memory agent (`memory.py`).

Reuse in later parts:
- the `CustomerProfile` and `PastIssue` schemas
- `make_store`, `recall` and `update_*` from `memory.py`
- `build_research_graph`

Part 6 (deep research) scales up 4.4. Part 6's ambient agent learns preferences with the same Store patterns.
Part 8 serves the memory agent with the server's own Store.

### Part 5: LangSmith (Observability & Evals; Building Reliable Agents; LangSmith Essentials M1–M2)
Shipped as built. Replaced `stages/04`. Dependency added: `agentevals` (which brings `openevals`). New data:
`data/eval_dataset.json` gained splits, and `data/triage_dataset.json` was added. New: `evals/` with the `uv run poe evals` task.

The local twins use the real `langsmith` SDK:
- tracing uses `tracing_context(enabled="local")` with `observability.offline_client()`, which never sends
- evaluation uses `evaluate(..., upload_results=False)` over in-memory `Example`s

Lessons:
1. Tracing basics.
2. Run types and `@traceable` / `trace()` / `process_inputs`.
3. Threads and trace analysis.
4. Datasets: splits, versions, synthetic examples.
5. Evaluators: code, LLM judge, trajectory.
6. Experiments: RAG chain vs support graph.
7. Pairwise, with position swap; online it uses `evaluate((a, b))`.
8. Summary evaluators: precision, recall, F1.
9. Evals as tests with the LangSmith pytest plugin.
10. Prompt hub lifecycle, with Playground and Polly explained.
11. User feedback.
12. Annotation queues: corrections become examples.

Reuse in later parts:
- `traced`, `record_feedback` and `load_rows`, which Part 9's dashboards and insights read from `data/traces.jsonl`
- `run_experiment`, `make_examples` and `pairwise_judge`
- `prompt_hub()`, which Part 9's Engine loop ships fixes through

Online-only code paths aren't exercised by CI because they need a key:
- `ensure_dataset`
- `update_dataset_tag`
- comparative `evaluate`
- `create_annotation_queue`
- `push_prompt`

Verify them manually with a free key.

### Part 6: Agent projects (Project: Ambient Agents; Project: Deep Research)
Shipped as built. Dependency added: an optional `web` extra (`ddgs`) for free web search. New data:
`data/inbox/*.json` holds 8 support emails, each with an `eval` block for the labels. Reports go to `data/reports/` (git-ignored).

Ambient agent (`ambient.py`, `build_inbox_agent(hitl=, memory=)`):
1. Triage router, then a hand-built response loop. Its tools are search_docs, search_changelog, quote_price, open_ticket, send_reply and done.
2. Evaluations: triage accuracy, expected tools, an LLM judge on per-email criteria.
3. Agent Inbox protocol: the `HumanInterrupt` and `HumanResponse` TypedDicts are defined locally, because the
   `langgraph.prebuilt.interrupt` versions are deprecated. `process_inbox()` plays the role of the UI.
4. Memory: triage and response preferences in the Store, rewritten from human edits, ignores and responses.

Deep research (`deep_research.py`):
5. Scoping: `ClarifyWithUser` ends the run with a question, then `ResearchBrief`.
6. The researcher: `search_sources` and `think`, a tool budget, compression.
7. The same researcher with tools from `docs_mcp_server.py`.
8. The supervisor: `ConductResearch`, `ResearchComplete` and `think`, with researchers run through `asyncio.gather`, `max_concurrent` and `max_iterations`.
9. Capstone: scope, then supervisor, then report. Graded by an LLM judge. Runs with `ainvoke`.

Loop guards protect real small models:
- `MAX_TOOL_ROUNDS` and `MAX_REVIEWS` in `ambient.py`
- a cap on clarifying questions

Part 8 deploys the inbox agent with cron and Agent Inbox. Ambient Agents M6 is mapped there.

### Part 7: Deep Agents (Intro to Deep Agents; dcode Essentials; LangSmith Fleet)
Shipped as built. Dependencies added: `deepagents` (0.7.24 locked) and `langchain-quickjs` (the WebAssembly
JavaScript interpreter; pure-Python wheels). New data: `data/logs/sync-client.log`. New folders: `skills/`
(refund-policy, log-analysis), `agents/` (templates) and `.deepagents/` (this repo's dcode config).

Shared code: `deep.py` and `agent_templates.py`. Every lesson's agent goes through `deep.deep_agent()`, which adds
a model-call budget and replaces the built-in `general-purpose` subagent with a budgeted one.
1. Running a deep agent: built-in tools, files in state, planning. In 0.7.x `write_todos` is opt-in via `TodoListMiddleware`.
2. Models and the system prompt: a subagent on the fast model; the assembled prompt, captured by middleware; harness profiles.
3. Tools and MCP: Nimbus Ops MCP tools, plus `my_account` with runtime context; `ainvoke`.
4. Threads, checkpointer and HITL: per-thread files, `interrupt_on`, and path-scoped `FilesystemPermission(mode="interrupt")`.
5. Backends: State, Filesystem, Store and Composite (`ops_backend`), with a deny permission on /docs/.
6. LocalShellBackend in a temp dir: minimal env, timeout, an allowlist reviewer, and the safety checklist.
7. Interpreter: `CodeInterpreterMiddleware` with programmatic tool calling.
8. Offloading large tool results, and summarization that saves history to `/conversation_history/`.
9. Skills: progressive disclosure from `skills/`.
10. `AGENTS.md` memory in the Store, edited by the agent and shared across threads.
11. Subagent team: declarative subagents, `response_format`, and a `CompiledSubAgent` (the 3.3 ReAct agent).
12. Dynamic subagents: `task()` from interpreter code, with `responseSchema`.
13. Async subagents and local deployment: the `LocalAgentServer` twin (in-process), or `--url` with `poe server`.
    `langgraph.json` now also serves `docs_researcher` and `ops_analyst`.
14. Capstone: `build_ops_analyst`, every feature in one agent, with an approval-gated report.
15. Agents from templates (Fleet twin), with MDA's project structure mapped in the docstring.
16. dcode: `.deepagents/` loaded by the SDK, read-only, with `.env` denied.

Reuse in later parts:
- `build_ops_analyst(on_server=True)` and `build_docs_researcher`, already served. Part 8 covers the server's
  store and checkpointer for them (Intro to LangSmith Deployment M4 "Memory in Deep Agents").
- `LocalAgentServer` shows the client calls Part 8's SDK lessons make for real.
- `agent_templates` for Part 9's A/B assistants if useful.

### Part 8: Deployment (Intro to LangGraph M6; Intro to LangSmith Deployment; LangSmith Essentials M3; Ambient Agents M6)
Shipped as built. Replaced `stages/05` (the `stage5` task is gone). Dependency added: `pyjwt`. New: `deployment.py`,
`auth.py`, `webapp.py` + `chat.html`, `langgraph.auth.json`, the `server-auth` task and compose service (port 2025).
`langgraph.json` serves 9 graphs (new: `support_agent`, `memory_agent`, `inbox_agent`, `inbox_ingest`,
`agent_from_template`), custom routes and a semantic store index (local fastembed via `deployment.embed`).

Lessons are SDK clients, `run(client, ...)`; `main()` checks the model setup, then `connect()`s (exit with how to start the server).
1. Concepts: the manifest, server info, one assistant per graph, schemas.
2. Sync/async clients, stateless runs, `RemoteGraph` (also as a node in a local graph).
3. Threads, runs, stream modes, background runs, approval over the API, history, search, copy.
4. Double texting: all four strategies on one agent.
5. Assistants from `AssistantContext` (style, model), versions, `set_latest`. The in-memory runtime doesn't
   restore `context` on `set_latest`; the lesson shows and says so.
6. Custom routes `/health`, `/tickets`, `/ui`.
7. The server's Store: SDK access, the memory agent across threads, semantic search and filters.
8. The Ops analyst on the server: approvals over the API, `AGENTS.md` in the server's Store.
9. Deploying without code: a graph factory builds `agents/*.md` templates per assistant (`deepagents deploy` is deprecated).
10. Auth: a local JWT issuer, owner filters, staff role, run context rewritten from the token, Store scoping.
11. UIs: raw HTTP + SSE like a UI does, Agent Chat UI, Studio, Agent Inbox, the `/ui` page.
12. Ambient deployment: `inbox_ingest` (loopback `get_client()`, idempotent `uuid5` threads), crons, the inbox of interrupts.
13. Production: `langgraph dockerfile`, compose, Postgres/Redis, cloud deployment explained.

Gotchas found:
- The server refuses graphs compiled with a checkpointer: `on_server=True` on `build_memory_agent` and `build_inbox_agent`.
- `langgraph dev` flags blocking calls (even `Path.resolve()` in `FilesystemBackend`): build such things in `asyncio.to_thread`.
- `assistants.get_schemas` needs the assistant id, not the graph name.
- Auth handlers must mutate `value["metadata"]` in place.

Reuse in Part 9: `support_agent` assistants for A/B tests, `/health` and the store for dashboards, the fake server for e2e.

### Part 9: Monitoring and the improvement loop (Monitoring Production Agents; LangSmith Engine; Essentials M4–M5; Observability & Evals M5; Building Reliable Agents M3)
Shipped as built. No new dependencies. New data: `data/traffic.json` (12 production-like conversations: angry
customers, injections, pasted PII) and `data/pricing.json` (illustrative tiers; local models cost 0). Lessons log
to `data/monitoring/` (git-ignored). New task: `uv run poe dashboard`.

Shared code:
- `monitoring.py`: `simulate_traffic` (traced turns with thread/plan/variant metadata, optional `fault` injection),
  `ensure_traffic` (each lesson works alone; later ones reuse the log), `summarize` (one row per trace, tags of
  every run), `run_filter` (LangSmith's filter language, incl. metadata and feedback key/value pairs), costs,
  percentiles, online evaluators (`ensure_feedback` scores what each evaluator hasn't), rules, alerts, A/B helpers.
- `guardrails.py`: `PIIMiddleware`, the trace anonymizer, `injection_guard` (tags its own run).
- `insights.py`, `dashboard.py`, `engine.py`: the Insights, Dashboards and Engine twins.
- `observability.flatten` rows gained `model`, `input_tokens`, `output_tokens`; `_append` is now `append_rows`.

Lessons:
1. Production traffic and filters.
2. Cost tracking: per turn and plan, what-if tiers, `get_usage_metadata_callback`.
3. Latency percentiles, per-run breakdown, an injected outage.
4. Online evaluators: sentiment, helpfulness, `cites_docs`; filters and sampling; feedback.
5. Automations (annotation queue, dataset, webhooks) and windowed alerts.
6. Security: PII never reaches the model (proved by a callback), injections refused and tagged.
7. A/B test of two prompt versions from the hub, stable assignment, two-proportion test.
8. Insights twin: summarise threads, then cluster.
9. Dashboard twin: static HTML, no JavaScript.
10. Engine 1-2: failures -> `PromptFix` -> hub tag "candidate".
11. Engine 3-5 (capstone): experiment + pairwise -> `ship_decision` -> move "prod" -> monitor before/after.

Gotchas found:
- `and`/`or`/`not` are Python keywords: `run_filter` renames them before `ast.parse`.
- `get_current_run_tree()` inside LangChain middleware has no `parent_run`: tag your own run, find traces by
  tree filter (LangSmith) or by the union of run tags (local).
- Judges on tiny models sometimes return invalid structure: evaluators, insights, the fix proposal and the
  pairwise judge skip that item instead of failing the lesson.

The curriculum is complete with this part. Keeping it working: bump dependencies in one PR, run
`DOCSHELP_E2E_MODEL=qwen2.5:0.5b uv run poe e2e` (or let `docker-e2e` do it), and re-check the Academy catalogue
for new courses; map any new lesson in `CURRICULUM.md` before writing it.

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
