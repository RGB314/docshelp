# Curriculum: the LangChain Academy, applied to one real app

This repo teaches everything in the [LangChain Academy](https://academy.langchain.com/collections) course
catalogue (17 courses, catalogue checked 2026-10-10). It applies every concept to one app, the **DocsHelp**
support assistant for the made-up product Nimbus Notes, instead of a different toy project per course.
Each concept is taught **once**, at the point where the app first needs it. Later lessons build on that
concept rather than repeating it.

Everything runs at **zero cost**. Models come from local Ollama by default, or from the Groq or Gemini free
tiers. LangSmith uses the free Developer account. Every LangSmith feature that needs a paid plan has a small
local twin that teaches the same idea.

Run `uv run poe lessons` to list the lessons. Run one with `uv run poe lesson <number>`, for example
`uv run poe lesson 1.6`.

## The nine parts

| Part | Theme | Builds on | Status |
|---|---|---|---|
| 1 | LangChain essentials: models, prompts, structured output, multimodal, RAG, tools, agents, streaming, memory | none | ✅ shipped |
| 2 | Production-ready agents: MCP, runtime context and state, multi-agent, middleware, HITL, dynamic agents | 1 | ✅ shipped |
| 3 | LangGraph core: state, nodes, edges, reducers, schemas, message management, persistence, streaming, interrupts, time travel | 1–2 | ✅ shipped |
| 4 | LangGraph at scale and long-term memory: parallelism, subgraphs, map-reduce, the Store, memory schemas | 3 | ✅ shipped |
| 5 | LangSmith: tracing, datasets, evaluators, experiments, prompt engineering, feedback, annotation | 1–4 | ✅ shipped |
| 6 | Agent projects: an ambient support-inbox agent and a deep-research agent | 1–5 | ✅ shipped |
| 7 | Deep Agents: planning, filesystems, sandboxes, skills, memory, subagents, agents-from-config | 1–6 | ✅ shipped |
| 8 | Deployment: LangGraph Server, SDK, assistants, double texting, custom routes, auth, Store, Chat UI | 3–7 | planned |
| 9 | Production: cost, quality, security monitoring, online evals, automations, and the improvement loop | 5, 8 | planned |

## Free vs paid: what runs where

| Academy feature | Here |
|---|---|
| Chat models (OpenAI in the videos) | Any provider via `DOCSHELP_PROVIDER`; Ollama (local) by default, Groq / Gemini free tiers recommended for Parts 6–7 |
| Embeddings | Local fastembed, always free |
| LangSmith tracing, datasets, experiments, Prompt Hub, annotation queues, online evaluators | Free LangSmith Developer account (optional; every lesson also runs locally) |
| LangSmith Studio, `langgraph dev` | Free, local |
| LangSmith Deployment (cloud), Managed Deep Agents | `langgraph dev` / Docker locally; cloud deployment explained only |
| LangSmith Insights, Dashboards, Engine, Fleet | Local twins (Parts 7 and 9) that teach the same concept |

## Coverage matrix

Setup, resources, feedback, transcript and conclusion pages are omitted. Every teaching lesson is listed.
**Covered by** is either the lesson that covers it or the part where it is planned.

### Quickstart: LangChain Essentials (Python)

| Academy lesson | Covered by |
|---|---|
| L1 Create Agent | `lessons/p1_langchain/06_tools_and_agents.py`, `lessons/p1_langchain/02_prompts_and_lcel.py` |
| L2 Models and Messages | `lessons/p1_langchain/01_models_and_messages.py` |
| L3 Streaming | `lessons/p1_langchain/07_streaming.py` |
| L4 Tools | `lessons/p1_langchain/06_tools_and_agents.py` |
| L5 Tools with MCP | `lessons/p2_production_agents/01_mcp.py` |
| L6 Memory | `lessons/p1_langchain/08_short_term_memory.py` |
| L7 Structured Output | `lessons/p1_langchain/03_structured_output.py`, `lessons/p1_langchain/10_capstone_support_agent.py` |
| L8 Dynamic Prompt | `lessons/p1_langchain/09_dynamic_prompt.py` |
| L9 Human in the Loop | `lessons/p2_production_agents/07_human_in_the_loop.py` |
| Getting Set Up: Local Models | [README: Ollama setup](README.md#2-set-up-your-model-provider) |

### Introduction to LangChain (Python)

| Academy lesson | Covered by |
|---|---|
| M1 L1 Foundational Models | `lessons/p1_langchain/01_models_and_messages.py` |
| M1 L2 Tools | `lessons/p1_langchain/06_tools_and_agents.py` |
| M1 L3 Short-Term Memory | `lessons/p1_langchain/08_short_term_memory.py` |
| M1 L4 Multimodal Messages | `lessons/p1_langchain/04_multimodal.py` |
| M1 L5 Personal Chef (project) | `lessons/p1_langchain/10_capstone_support_agent.py` |
| M2 L1 Model Context Protocol | `lessons/p2_production_agents/01_mcp.py` (server: `src/docshelp/mcp_server.py`) |
| M2 L2 Context and State | `lessons/p2_production_agents/02_context_and_state.py` |
| M2 L3 Multi-Agent Systems | `lessons/p2_production_agents/03_multi_agent.py` |
| M2 L4 Wedding Planner (project) | `lessons/p2_production_agents/03_multi_agent.py` (a support team: supervisor plus docs, billing and MCP-backed status specialists) |
| M3 L1 What is Middleware? | `lessons/p2_production_agents/04_middleware.py`, `lessons/p2_production_agents/05_prebuilt_middleware.py` |
| M3 L2 Managing Long Conversations | `lessons/p2_production_agents/06_long_conversations.py` |
| M3 L3 Human In The Loop | `lessons/p2_production_agents/07_human_in_the_loop.py` |
| M3 L4 Dynamic Agents | `lessons/p2_production_agents/08_dynamic_agents.py`, `lessons/p1_langchain/09_dynamic_prompt.py` |
| M3 L5 Email Assistant (project) | `lessons/p2_production_agents/09_capstone_email_assistant.py` |
| M3 L6 Agent Chat UI | Part 8 (planned) |

### Quickstart: LangGraph Essentials (Python)

| Academy lesson | Covered by |
|---|---|
| L1 Nodes, L2 Edges | `lessons/p3_langgraph/01_state_nodes_edges.py` |
| L3 Conditional Edges | `lessons/p3_langgraph/02_conditional_edges_and_command.py` |
| L4 Memory | `lessons/p3_langgraph/03_react_agent_by_hand.py`, `lessons/p3_langgraph/10_external_memory.py` |
| L5 Interrupt, Human in the Loop | `lessons/p3_langgraph/13_dynamic_interrupts.py` |
| L6 Application (email workflow) | `lessons/p3_langgraph/15_capstone_support_graph.py` (graph: `src/docshelp/graph.py`) |

### Introduction to LangGraph (Python)

| Academy lesson | Covered by |
|---|---|
| M1 L1 Motivation, L2 Simple Graph | `lessons/p3_langgraph/01_state_nodes_edges.py` |
| M1 L4 Chain, L5 Router, L6 Agent, L7 Agent with Memory | `lessons/p3_langgraph/03_react_agent_by_hand.py` (graph: `src/docshelp/react.py`), router also in `lessons/p3_langgraph/02_conditional_edges_and_command.py` |
| M1 L3 LangSmith Studio | `lessons/p3_langgraph/04_studio.py` |
| M1 L8 Intro to Deployment | Part 8 (planned) |
| M2 L1 State Schema | `lessons/p3_langgraph/05_state_schemas.py` |
| M2 L2 State Reducers | `lessons/p3_langgraph/06_reducers.py` |
| M2 L3 Multiple Schemas | `lessons/p3_langgraph/07_multiple_schemas.py` |
| M2 L4 Trim and Filter Messages | `lessons/p3_langgraph/08_trim_and_filter.py` |
| M2 L5 Chatbot w/ Summarizing Messages | `lessons/p3_langgraph/09_summarizing_chatbot.py` (graph: `src/docshelp/chatbot.py`) |
| M2 L6 Chatbot w/ External Memory | `lessons/p3_langgraph/10_external_memory.py` |
| M3 L1 Streaming | `lessons/p3_langgraph/11_streaming.py` |
| M3 L2 Breakpoints, L3 Editing State and Human Feedback | `lessons/p3_langgraph/12_breakpoints_and_editing_state.py` |
| M3 L4 Dynamic Breakpoints | `lessons/p3_langgraph/13_dynamic_interrupts.py` |
| M3 L5 Time Travel | `lessons/p3_langgraph/14_time_travel.py` |
| M4 L1 Parallelization | `lessons/p4_langgraph_scale/01_parallelization.py` |
| M4 L2 Sub-graphs | `lessons/p4_langgraph_scale/02_subgraphs.py` |
| M4 L3 Map-reduce | `lessons/p4_langgraph_scale/03_map_reduce.py` |
| M4 L4 Research Assistant | `lessons/p4_langgraph_scale/04_research_assistant.py` (graph: `src/docshelp/research.py`) |
| M5 L1 Short vs Long-Term Memory | `lessons/p4_langgraph_scale/05_store_and_long_term_memory.py` |
| M5 L2 LangGraph Store | `lessons/p4_langgraph_scale/05_store_and_long_term_memory.py`, `lessons/p4_langgraph_scale/06_semantic_memory_search.py` |
| M5 L3 Memory Schema + Profile | `lessons/p4_langgraph_scale/07_memory_profile.py` |
| M5 L4 Memory Schema + Collection | `lessons/p4_langgraph_scale/08_memory_collection.py` |
| M5 L5 Agent with Long-Term Memory | `lessons/p4_langgraph_scale/09_capstone_memory_agent.py` (agent: `src/docshelp/memory.py`) |
| M6 L1 Deployment Concepts, L2 Creating a Deployment, L3 Connecting to a Deployment | Part 8 (planned) |
| M6 L4 Double Texting, L5 Assistants | Part 8 (planned) |

### Introduction to Agent Observability & Evaluations (LangSmith)

| Academy lesson | Covered by |
|---|---|
| M1 L1 Tracing Basics | `lessons/p5_langsmith/01_tracing_basics.py` |
| M1 L2 Types of Runs, L3 Alternative Ways to Trace | `lessons/p5_langsmith/02_run_types_and_traceable.py` |
| M1 L4 Conversational Threads | `lessons/p5_langsmith/03_threads_and_analysis.py` |
| M2 L1 Datasets | `lessons/p5_langsmith/04_datasets.py` |
| M2 L2 Evaluators | `lessons/p5_langsmith/05_evaluators.py` |
| M2 L3 Experiments, L4 Analyzing Experiment Results | `lessons/p5_langsmith/06_experiments.py` |
| M2 L5 Pairwise Experiments | `lessons/p5_langsmith/07_pairwise.py` |
| M2 L6 Summary Evaluators | `lessons/p5_langsmith/08_summary_evaluators.py` |
| M3 L1 Playground, L2 Prompt Hub, L3 Prompt Engineering Lifecycle, L4 Polly | `lessons/p5_langsmith/10_prompt_engineering.py` (hub: `src/docshelp/prompts.py`; Playground and Polly are walked through in the docstring) |
| M4 L1 User Feedback | `lessons/p5_langsmith/11_user_feedback.py` |
| M4 L2 Annotation Queues | `lessons/p5_langsmith/12_annotation_queues.py` |
| M5 L1 Filtering, L2 Online Evaluation, L3 Automations, L4 Monitoring, L5 Custom Dashboards | Part 9 (planned) |

### Building Reliable Agents with LangSmith

| Academy lesson | Covered by |
|---|---|
| Intro: The Agent Development Lifecycle | Part 9 (planned) |
| M1 L1 Observability, L2 Tracing with LangSmith | `lessons/p5_langsmith/01_tracing_basics.py` |
| M1 L3 Analyzing your Agent | `lessons/p5_langsmith/03_threads_and_analysis.py` |
| M2 L1 Evaluating Agents, L2 Creating Datasets | `lessons/p5_langsmith/04_datasets.py`, `lessons/p5_langsmith/09_evals_as_tests.py` |
| M2 L3 Running Experiments | `lessons/p5_langsmith/06_experiments.py` |
| M2 L4 Code-based Eval, L5 LLM-as-Judge | `lessons/p5_langsmith/05_evaluators.py` |
| M2 L6 Pairwise Evaluations | `lessons/p5_langsmith/07_pairwise.py` |
| M3 L1 Moving Towards Production, L2 Insights Agent, L3 Online Evals, L4 Automations | Part 9 (planned) |

### Quickstart: LangSmith Essentials

| Academy lesson | Covered by |
|---|---|
| M1 Debugging with Studio | `lessons/p3_langgraph/04_studio.py` |
| M1 What is Tracing?, Capstone | `lessons/p5_langsmith/01_tracing_basics.py`, `lessons/p5_langsmith/03_threads_and_analysis.py` |
| M2 Datasets and Offline Evals, Experiments, Capstone | `lessons/p5_langsmith/04_datasets.py`, `lessons/p5_langsmith/06_experiments.py`, `lessons/p5_langsmith/09_evals_as_tests.py` |
| M3 Deployments | Part 8 (planned) |
| M4 Online Evaluators and Automations, Annotation Queues, Insights, Dashboards, Capstone | Part 9 (planned) |
| M5 Closing the Loop, Automating the ADLC with LangSmith Engine | Part 9 (planned) |

### Project: Ambient Agents with LangGraph

| Academy lesson | Covered by |
|---|---|
| M1 LangGraph 101 | `lessons/p3_langgraph/01_state_nodes_edges.py` … `lessons/p3_langgraph/15_capstone_support_graph.py` (all of Part 3) |
| M2 Building Agents | `lessons/p6_agent_projects/01_ambient_inbox_agent.py` (agent: `src/docshelp/ambient.py`) |
| M3 Agent Evaluations | `lessons/p6_agent_projects/02_ambient_evaluations.py` |
| M4 Human-in-the-Loop | `lessons/p6_agent_projects/03_ambient_human_in_the_loop.py` |
| M5 Memory | `lessons/p6_agent_projects/04_ambient_memory.py` |
| M6 Deployment | Part 8 (planned) |

### Project: Deep Research with LangGraph

| Academy lesson | Covered by |
|---|---|
| M2 Scoping | `lessons/p6_agent_projects/05_research_scoping.py` (system: `src/docshelp/deep_research.py`) |
| M3 Research Agent | `lessons/p6_agent_projects/06_research_agent.py` |
| M4 Research Agent with MCP | `lessons/p6_agent_projects/07_research_agent_mcp.py` (server: `src/docshelp/docs_mcp_server.py`) |
| M5 Research Supervisor | `lessons/p6_agent_projects/08_research_supervisor.py` |
| M6 Multi-Agent Research System | `lessons/p6_agent_projects/09_capstone_deep_research.py` |

### Introduction to Deep Agents

| Academy lesson | Covered by |
|---|---|
| M1 Overview, Running a Deep Agent | `lessons/p7_deep_agents/01_running_a_deep_agent.py` (harness: `src/docshelp/deep.py`) |
| M1 Models, The System Prompt | `lessons/p7_deep_agents/02_models_and_system_prompt.py` |
| M1 Tools, MCP | `lessons/p7_deep_agents/03_tools_and_mcp.py` |
| M1 Messages, Threads, and Checkpointers; HITL; Test Your Skills | `lessons/p7_deep_agents/04_threads_and_hitl.py` (and the Part 7 self-check below) |
| M2 The Deep Agent Environment, Filesystem Backends | `lessons/p7_deep_agents/05_filesystem_backends.py` |
| M2 Sandboxes and LocalShell | `lessons/p7_deep_agents/06_sandbox_and_local_shell.py` |
| M2 Interpreter | `lessons/p7_deep_agents/07_interpreter.py` |
| M3 Summarization and Offload | `lessons/p7_deep_agents/08_summarization_and_offload.py` |
| M3 Skills | `lessons/p7_deep_agents/09_skills.py` (skills: `skills/`) |
| M3 Memory | `lessons/p7_deep_agents/10_memory.py` |
| M4 Delegation, Building a Subagent Team | `lessons/p7_deep_agents/11_subagent_team.py` |
| M4 Dynamic Subagents | `lessons/p7_deep_agents/12_dynamic_subagents.py` |
| M5 Local Deployment, Asynchronous Subagents | `lessons/p7_deep_agents/13_async_subagents.py` (served graphs: `src/docshelp/server.py`) |
| M5 The Sales Assistant, The Advanced Sales Assistant (projects) | `lessons/p7_deep_agents/14_capstone_support_ops_analyst.py` (the same build as a Support Ops analyst) |
| M6 What is MDA?, MDA Project Structure | `lessons/p7_deep_agents/15_agents_from_templates.py` (explained in the docstring, mapped onto this repo) |

### dcode Essentials

| Academy lesson | Covered by |
|---|---|
| Welcome to dcode, dcode Essentials, Goals | `lessons/p7_deep_agents/16_dcode_project_config.py` |
| Skills, MCP, Plugins, and Marketplaces; Configuring Subagents; dcode for Enterprise | `lessons/p7_deep_agents/16_dcode_project_config.py` (this repo's config: `.deepagents/`) |

### LangSmith Fleet (no-code agents)

| Academy lesson | Covered by |
|---|---|
| L1 Building An Agent, L2 Human-In-The-Loop, L3 What Can Agents Do? | `lessons/p7_deep_agents/15_agents_from_templates.py` (local twin; templates: `agents/`, builder: `src/docshelp/agent_templates.py`) |
| L4 Skills, L5 MCP, L6 Ambient Agents, L7 Sub-Agents, L8 Computer Use (Sandboxes), L9 Templates | `lessons/p7_deep_agents/15_agents_from_templates.py` (computer use: `lessons/p7_deep_agents/06_sandbox_and_local_shell.py`) |

### Introduction to LangSmith Deployment

| Academy lesson | Covered by |
|---|---|
| M1 Concepts, The Agent, Deploy, Use the Tutor | Part 8 (planned) |
| M2 Connecting, Assistants, Connecting the UI, Custom Routes | Part 8 (planned) |
| M3 UI Walkthrough | Part 8 (planned) |
| M4 Memory In LangGraph, Accessing Storage, Memory in Deep Agents | Part 8 (planned) |
| M5 Introduction to Auth, Authentication, Authorization, Auth Provider, Auth on Tutor | Part 8 (planned) |
| M6 Deploying Without Coding Using deepagents-cli, Deploy the Tutor Agent, Bring Your Own UI | Part 8 (planned) |

### Monitoring Production Agents

| Academy lesson | Covered by |
|---|---|
| M1 Token and Cost Breakdowns, Tracking Costs, Viewing Costs | Part 9 (planned) |
| M2 Analyze Trends with Insights | Part 9 (planned) |
| M3 Monitoring Latency and Errors, Monitoring User Sentiment, A/B Testing | Part 9 (planned) |
| M4 Monitoring Security, Setting Up Alerts | Part 9 (planned) |
| Capstone Project | Part 9 (planned) |

### Autonomous Agent Improvement with LangSmith Engine

| Academy lesson | Covered by |
|---|---|
| What is the ADLC?, Creating a Recruiting Agent, Setting Up Engine | Part 9 (planned) |
| Build: Engine Proposes a Fix, Test: Testing Engine's Fix, Deploy: Shipping the Fix, Monitor | Part 9 (planned) |

### Beyond the Academy

Real support apps need these, and the current Academy courses don't teach them on their own:

| Topic | Covered by |
|---|---|
| RAG: loaders, splitters, local embeddings, vector store, retriever, citations | `lessons/p1_langchain/05_rag.py` |
| Provider-neutral model layer (swap vendors via config) | `src/docshelp/providers/` + `lessons/p1_langchain/01_models_and_messages.py` |
| Agent security: identity from context, never from tool arguments; verification gates; credential guard | `lessons/p2_production_agents/02_context_and_state.py`, `lessons/p2_production_agents/09_capstone_email_assistant.py`, `lessons/p2_production_agents/04_middleware.py` |

### LCAE Practice Exam

The practice exam's role is filled by a **self-check** at the end of each part (below). TypeScript editions
of the courses teach the same concepts and are out of scope.

## Self-checks

### Part 1 self-check

<details>
<summary>1. Lesson 1.8 remembers the seat count, yet lesson 1.1 says the model is stateless. Where does the memory live?</summary>

In the checkpointer. After every step it saves the agent's state, including the message list, under the
`thread_id`. The next call loads that state, so the full history is resent to the model. The model itself
still remembers nothing.
</details>

<details>
<summary>2. Why pass the customer's plan as runtime <code>context</code> rather than in the user message?</summary>

The app already knows the plan. The user can't change or spoof the context, it is typed by `context_schema`,
and it doesn't use up conversation history. The dynamic prompt reads it on every model call.
</details>

<details>
<summary>3. In tool calling, who runs the tool?</summary>

Your code. The model only returns `tool_calls`, which hold a name and arguments. `create_agent`'s loop runs the
tool and sends the result back as a `ToolMessage` until the model replies without tool calls.
</details>

<details>
<summary>4. Which stream mode would you use for a "Searching docs…" status line, and which for typing out the answer?</summary>

`custom`, with events sent from inside the tool via `get_stream_writer()`, for the status line. `messages`
for the LLM tokens of the answer. `updates` reports each finished step.
</details>

<details>
<summary>5. What do you gain from <code>response_format=SupportAnswer</code> over plain text?</summary>

The app gets fields it can act on: `sources` to render as citations, and `needs_ticket` to escalate.
Pydantic validates them, so you don't have to parse free text.
</details>

<details>
<summary>6. Lesson 1.5's retriever finds "Refunds" for "get my money back". Why does keyword search fail there?</summary>

The two phrases share no words. Embeddings compare meaning, so the vectors of the question and the Refunds
section end up close together anyway.
</details>

### Part 2 self-check

<details>
<summary>1. Why does <code>my_account</code> take no arguments, when it clearly needs an account id?</summary>

It reads the id from `runtime.context`, which the app sets per run. If the id were an argument, the model
would choose it, and a customer could talk the model into looking up someone else's account. Anything that
decides *whose* data a tool touches belongs in context (or verified state), not in tool arguments.
</details>

<details>
<summary>2. The customer says "I'm on Windows" and a later tool needs that. Context or state? How does the tool save it?</summary>

State: it was learned during the conversation, while context is fixed when the run starts. Declare the field in
a custom `state_schema` (an `AgentState` subclass), and have the tool return `Command(update={"device": ...,
"messages": [ToolMessage(..., tool_call_id=runtime.tool_call_id)]})`. The `ToolMessage` is required, because
every tool call needs an answer.
</details>

<details>
<summary>3. The supervisor asks the billing specialist "and how much would that cost them?" Why does that fail?</summary>

A specialist starts with a fresh message list, so it never saw the conversation and "that" and "them" mean
nothing to it. That isolation is the point of subagents as tools (small contexts), so the supervisor must ask
complete, standalone questions. Facts like the customer's account travel as runtime context instead.
</details>

<details>
<summary>4. With <code>middleware=[A, B]</code>, in which order do the hooks run?</summary>

`before_agent` and `before_model`: A, then B. `after_model` and `after_agent`: B, then A. Wrap hooks nest: A's
`wrap_model_call` wraps B's, which wraps the real call.
</details>

<details>
<summary>5. Summarization, trimming, context editing: which ones change the saved conversation?</summary>

Summarization replaces old messages in state with a summary. Trimming with `RemoveMessage` deletes them from
state. `ContextEditingMiddleware` only changes the request sent to the model, so the saved history keeps the
full tool results.
</details>

<details>
<summary>6. Why does <code>HumanInTheLoopMiddleware</code> need a checkpointer, and what do the three decisions do?</summary>

The run stops mid-loop and returns `__interrupt__`. The checkpointer saves the paused state so
`Command(resume=...)` can continue it later, even in another process. `approve` runs the call as proposed.
`edit` runs a corrected call. `reject` skips the call and returns your message to the model.
</details>

<details>
<summary>7. Lesson 2.5 sets <code>on_failure="error"</code> on the model retry. What breaks with the default, <code>"continue"</code>?</summary>

After the last retry, `"continue"` turns the failure into an ordinary AI message saying the call failed. The
fallback middleware outside it sees a successful call and never tries the backup model. Retry must raise for
the fallback to take over.
</details>

<details>
<summary>8. The capstone hides <code>lookup_account</code> until verification, and the tool also refuses unless state says verified. Why both?</summary>

Defence in depth. Hiding the tool keeps the model from trying, and saves tokens. The check inside the tool
still holds if a misconfigured middleware or a direct call ever reaches it. It also reads the account from
verified state, never from the model.
</details>

### Part 3 self-check

<details>
<summary>1. Two parallel nodes both return <code>{"tags": [...]}</code> and the run fails. Why, and what are the two fixes?</summary>

With no reducer, a key is overwritten, and two writes in one step are ambiguous, so LangGraph raises
`InvalidUpdateError`. You can give the key a reducer (`Annotated[list[str], operator.add]` or a custom merge
function), or have the nodes write to different keys.
</details>

<details>
<summary>2. You declared the state as a Pydantic model with <code>priority: Literal["low", "high"]</code>. A node returns <code>{"priority": "critical"}</code>. What happens?</summary>

Nothing stops it. Pydantic state validates the graph's *input*, not node outputs. Validate model output
where it enters: use structured output, or check the value inside the node (lesson 3.5).
</details>

<details>
<summary>3. When would you route with <code>add_conditional_edges</code>, and when with <code>Command(goto=...)</code>?</summary>

Use conditional edges when routing is a separate, pure decision on the state: it's easy to test and visible in
the graph. Use `Command` when a node's decision and its state update belong together, as in handoffs, a review
step that also edits the ticket, or agents passing control to each other.
</details>

<details>
<summary>4. Which of filtering, <code>RemoveMessage</code> and <code>trim_messages</code> actually shrinks the saved history?</summary>

Only `RemoveMessage`, which the `add_messages` reducer applies to state. Filtering a slice and
`trim_messages` change only what one model call sees. The summarizing chatbot (3.9) combines a summary
with `RemoveMessage`.
</details>

<details>
<summary>5. A node calls <code>interrupt()</code> twice. After the second resume, how many times has the node's code run?</summary>

Three times. Each resume re-runs the node from the top, and earlier `interrupt()` calls return their saved
answers in order. So put side effects (sending emails, writing tickets) after the interrupts, or in a later node.
</details>

<details>
<summary>6. <code>interrupt_before=["tools"]</code> vs <code>interrupt()</code> inside a node: what's the difference?</summary>

A static breakpoint is set at compile time and stops before that node on every run. It's handy for
debugging and for approving every tool call. `interrupt()` stops only when the node's logic decides to,
for example a refund over $100, and it carries a payload that the caller answers with `Command(resume=...)`.
</details>

<details>
<summary>7. How do you correct a tool call's arguments while the graph is paused before <code>tools</code>?</summary>

Call `update_state(config, {"messages": [AIMessage(..., tool_calls=fixed, id=original.id)]})`. Because the id
is the same, `add_messages` replaces the message. Then `invoke(None, config)` resumes. To skip the tool entirely,
write the `ToolMessage` yourself with `as_node="tools"`.
</details>

<details>
<summary>8. Replay vs fork in time travel?</summary>

Replay is `invoke(None, past_snapshot.config)`: it re-runs everything after that checkpoint, unchanged. Fork is
`update_state(past_snapshot.config, change)`: it creates a new branch with modified state, which you then run. The
original history stays as it was in both cases.
</details>

<details>
<summary>9. Why does <code>build_graph</code> put a <code>RetryPolicy</code> on <code>retrieve</code> but handle irrelevant documents with a rewrite loop instead?</summary>

The two errors are different kinds. A flaky store is transient, so retrying the same call is the fix. Irrelevant
documents are something the LLM can recover from: the result goes into state (`docs_relevant`) and the graph
loops through `rewrite` with a better query. Retrying the identical query would return the same documents.
</details>

### Part 4 self-check

<details>
<summary>1. Three parallel branches feed one <code>answer</code> node, but one branch has two steps. How do you make <code>answer</code> wait for all of them?</summary>

Use `add_edge(["trim_docs", "search_changelog", "check_status"], "answer")`. A list of sources means "wait
for all of these". With three separate edges, `answer` could run as soon as the shorter branches finish, and
then run again.
</details>

<details>
<summary>2. When do you need <code>Send</code> instead of ordinary fan-out edges?</summary>

When the number of parallel branches depends on the data, such as one per retrieved document or one per
analyst, or when each branch needs its own payload rather than the shared state. Ordinary edges are fixed
when the graph is built.
</details>

<details>
<summary>3. Two subgraphs both return <code>reviewed_ids</code>. What does the parent need, and what keeps a subgraph from returning its private keys?</summary>

The parent needs a reducer on `reviewed_ids` (for example `operator.add`), because both subgraphs write it in
the same step. Each subgraph's `output_schema` limits what it returns to the parent.
</details>

<details>
<summary>4. Checkpointer vs Store: where would you keep "this customer uses Windows"?</summary>

In the Store, under a namespace like `("profile", account_id)`. The checkpointer holds one thread's state, and
a new conversation starts empty. The Store is shared across threads, and nodes and tools reach it through
`runtime.store`.
</details>

<details>
<summary>5. Why patch a profile with trustcall instead of asking the model to rewrite it?</summary>

A rewrite tends to drop fields the latest message didn't mention, and it can return invalid JSON. trustcall
asks for JSON patches against the existing document and retries with the validation error, so untouched
fields survive (lesson 4.7).
</details>

<details>
<summary>6. Profile or collection: which fits past issues, and which fits "preferred OS"?</summary>

Past issues accumulate and change one at a time, so they fit a collection: one Store item each, searchable
and updatable on its own. The preferred OS has one current value, so it fits the profile.
</details>

<details>
<summary>7. The store has an embedding index. How do you find a customer's open issues that relate to "slow sync", and keep internal notes out of search?</summary>

Call `store.search(("issues", account), query="slow sync", filter={"status": "open"}, limit=5)`. Store internal
notes with `put(..., index=False)`, so they're never embedded and never match a semantic query.
</details>

### Part 5 self-check

<details>
<summary>1. What do you change in the code to start tracing every lesson to LangSmith?</summary>

Nothing. Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` (optionally `LANGSMITH_PROJECT`). LangChain and
LangGraph then trace every call automatically. `@traceable` and `trace()` are only needed for your own
non-LangChain code.
</details>

<details>
<summary>2. Your retrieval function is plain Python. How do you make it show up as a retriever step, with documents, in the trace?</summary>

Decorate it with `@traceable(run_type="retriever")` and return a list of
`{"page_content", "type": "Document", "metadata"}` dicts. LangSmith then renders it as retrieved documents
(lesson 5.2).
</details>

<details>
<summary>3. How does LangSmith know that five traces are one conversation?</summary>

They share a `thread_id` (or `session_id` / `conversation_id`) in their metadata. The Threads view groups
them and sums tokens and latency per conversation (lesson 5.3).
</details>

<details>
<summary>4. Which evaluator type would you use to check (a) the right doc was retrieved, (b) the answer is correct, (c) the agent called the pricing tool?</summary>

(a) A code evaluator, since the answer is exact and cheap to check. (b) An LLM-as-judge against the reference
answer. (c) A trajectory evaluator, `agentevals`' match evaluator in `superset` mode, or an LLM trajectory
judge (lesson 5.5).
</details>

<details>
<summary>5. Why can't precision or F1 be an ordinary evaluator?</summary>

They're defined over the whole set: precision needs every escalated email, not one example's output. That's
what `summary_evaluators` are for. They receive every example's outputs and references at once (lesson 5.8).
</details>

<details>
<summary>6. Your pairwise judge prefers answer A 80% of the time, whichever system wrote it. What's wrong, and how do you fix it?</summary>

That's position bias. Show the pair in alternating (or random) order, and map the verdict back to the system.
`pairwise_judge(..., swap=...)` does this, and LangSmith's `randomize_order=True` too (lesson 5.7).
</details>

<details>
<summary>7. How do you ship a prompt change without a code deploy, and undo it?</summary>

Push the new version to the prompt hub and evaluate it against the dataset. If it wins, move the `prod` tag to
its commit. The app pulls `"docshelp-rag:prod"`, so moving the tag back to the old commit is the rollback
(lesson 5.10).
</details>

<details>
<summary>8. A customer clicks 👎. How does that end up preventing the same mistake next release?</summary>

The rating is stored as feedback on that run's id (5.11). A rule queues the run for review (5.12), the reviewer
writes the correct answer, and it becomes a dataset example. The next experiment (5.6) or `poe evals` (5.9) then
tests that exact case.
</details>

### Part 6 self-check

<details>
<summary>1. What makes the inbox agent "ambient" rather than a chatbot, and why triage before responding?</summary>

It's triggered by events (new emails), runs in the background, and involves a person only when it must. Triage is
one cheap structured call. It keeps the expensive tool-using response agent away from newsletters and
auto-replies, and sends alerts that a person must handle to "notify".
</details>

<details>
<summary>2. In the Agent Inbox protocol, what happens to each of the four responses to a drafted reply?</summary>

`accept` sends it as drafted. `edit` sends the human's version, and with memory updates the response preferences.
`respond` sends nothing: the feedback goes back to the model as a tool result and it drafts again. `ignore` ends
the run, and with memory teaches triage that the email didn't need a reply.
</details>

<details>
<summary>3. Why evaluate triage separately from responses?</summary>

They fail differently and need different measures. Triage is a classification, scored on every email for accuracy
and the direction of its errors. Responses only exist for "respond" emails, and need tool checks plus an LLM judge
against each email's own success criteria. A single end-to-end score would hide which part broke.
</details>

<details>
<summary>4. What does the <code>think</code> tool do, given that it returns its input?</summary>

Nothing, as a tool. Making the model write down what it has learned and what's missing before its next call makes
the loop more deliberate: fewer redundant searches and earlier, better-founded stops. It's cheap structured
reflection.
</details>

<details>
<summary>5. Why do researchers return <em>compressed</em> findings, and why don't they see the conversation?</summary>

The supervisor coordinates several researchers. Raw search results would flood its context, while compressed,
cited findings fit easily. Each researcher gets only a standalone topic, so its own context stays small and
focused, and the topics can run in parallel without depending on each other.
</details>

<details>
<summary>6. Which limits keep a deep-research run from costing too much, and where are they enforced?</summary>

`max_tool_rounds` per researcher (in `should_continue`), `max_concurrent` researchers per supervisor turn (extra
`ConductResearch` calls are told to wait), and `max_iterations` supervisor turns (in `supervisor_tools`). The
scoping step also stops after a few clarifying questions.
</details>

### Part 7 self-check

<details>
<summary>1. What does <code>create_deep_agent</code> add on top of <code>create_agent</code>, and why does each piece matter for long tasks?</summary>

A filesystem (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) to keep notes and drafts outside the
context window; `task` to hand self-contained work to subagents with their own context; summarization that offloads
old history to a file; and patched-up tool calls. Planning (`write_todos`), skills, memory, a shell and an
interpreter are opt-in. Together they let an agent work for many steps without its context overflowing.
</details>

<details>
<summary>2. Files go to <code>/docs/</code>, <code>/memories/</code>, <code>/reports/</code> and <code>/scratch.md</code>. Where does each end up in the capstone, and how long does it last?</summary>

`CompositeBackend` routes by prefix: `/docs/` to the docs folder on disk (and a deny permission makes it read-only),
`/memories/` to the LangGraph Store (every thread, as long as the store lasts), `/reports/` to `data/reports/` on disk
(permanent), and anything else to the thread's state (checkpointed with the thread, gone in a new one).
</details>

<details>
<summary>3. Why is <code>LocalShellBackend</code> with <code>virtual_mode=True</code> still not a sandbox, and what makes running commands acceptable?</summary>

`virtual_mode` only confines the file tools. Shell commands run on the host with your account's permissions. Make it
acceptable with a throwaway directory holding only the needed files, a minimal environment (no keys), a timeout and
output cap, an approval or allowlist on every command, and in production a real remote sandbox.
</details>

<details>
<summary>4. When would you use the interpreter rather than plain tool calls, and what's the catch?</summary>

When the work is loops, joins or aggregation over many tool results: one `eval` calls the tools from code and returns
only the result, saving model turns and context. The catch: tools called from code, and subagents dispatched with
`task()`, skip `interrupt_on`, so expose only tools that are safe without approval.
</details>

<details>
<summary>5. Skills, memory and tools all shape behaviour. What goes where?</summary>

Tools do things. Skills are how-to instructions loaded on demand: only their names and descriptions sit in the
prompt. Memory (`AGENTS.md`) is always in the prompt, so it holds short standing rules and facts, and the agent can
edit it when it learns one.
</details>

<details>
<summary>6. Sync, dynamic and async subagents: what is each for?</summary>

A `task` subagent runs to completion while the main agent waits, with a fresh context and only its answer coming
back. Dynamic subagents are dispatched from interpreter code, in loops and in parallel, with typed results. Async
subagents run on an Agent Protocol server in the background; the main agent gets a task id and checks, updates or
cancels it later.
</details>

<details>
<summary>7. What do a Fleet template, an MDA project and a dcode <code>.deepagents/</code> folder have in common?</summary>

They describe an agent as configuration (instructions, tools, skills, subagents, approvals, MCP servers, triggers)
that a harness turns into a running deep agent. People review files, not code, and the same concepts from this part
apply to all three.
</details>
