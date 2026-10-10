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
| 3 | LangGraph core: state, nodes, edges, reducers, schemas, message management, persistence, streaming, interrupts, time travel | 1–2 | planned |
| 4 | LangGraph at scale and long-term memory: parallelism, subgraphs, map-reduce, the Store, memory schemas | 3 | planned |
| 5 | LangSmith: tracing, datasets, evaluators, experiments, prompt engineering, feedback, annotation | 1–4 | planned |
| 6 | Agent projects: an ambient support-inbox agent and a deep-research agent | 1–5 | planned |
| 7 | Deep Agents: planning, filesystems, sandboxes, skills, memory, subagents, agents-from-config | 1–6 | planned |
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
| L1 Nodes, L2 Edges, L3 Conditional Edges | Part 3 (planned) |
| L4 Memory | Part 3 (planned) |
| L5 Interrupt, Human in the Loop | Part 3 (planned) |
| L6 Application (email workflow) | Part 3 (planned) |

### Introduction to LangGraph (Python)

| Academy lesson | Covered by |
|---|---|
| M1 L1 Motivation, L2 Simple Graph, L4 Chain, L5 Router, L6 Agent, L7 Agent with Memory | Part 3 (planned) |
| M1 L3 LangSmith Studio | Part 3 (planned) |
| M1 L8 Intro to Deployment | Part 8 (planned) |
| M2 L1 State Schema, L2 State Reducers, L3 Multiple Schemas | Part 3 (planned) |
| M2 L4 Trim and Filter Messages, L5 Summarizing Chatbot, L6 External Memory | Part 3 (planned) |
| M3 L1 Streaming, L2 Breakpoints, L3 Editing State and Human Feedback, L4 Dynamic Breakpoints, L5 Time Travel | Part 3 (planned) |
| M4 L1 Parallelization, L2 Sub-graphs, L3 Map-reduce, L4 Research Assistant | Part 4 (planned) |
| M5 L1 Short vs Long-Term Memory, L2 LangGraph Store | Part 4 (planned) |
| M5 L3 Memory Schema + Profile, L4 Memory Schema + Collection, L5 Agent with Long-Term Memory | Part 4 (planned) |
| M6 L1 Deployment Concepts, L2 Creating a Deployment, L3 Connecting to a Deployment | Part 8 (planned) |
| M6 L4 Double Texting, L5 Assistants | Part 8 (planned) |

### Introduction to Agent Observability & Evaluations (LangSmith)

| Academy lesson | Covered by |
|---|---|
| M1 L1 Tracing Basics, L2 Types of Runs, L3 Alternative Ways to Trace, L4 Conversational Threads | Part 5 (planned) |
| M2 L1 Datasets, L2 Evaluators, L3 Experiments, L4 Analyzing Experiment Results | Part 5 (planned) |
| M2 L5 Pairwise Experiments, L6 Summary Evaluators | Part 5 (planned) |
| M3 L1 Playground, L2 Prompt Hub, L3 Prompt Engineering Lifecycle, L4 Polly | Part 5 (planned) |
| M4 L1 User Feedback, L2 Annotation Queues | Part 5 (planned) |
| M5 L1 Filtering, L2 Online Evaluation, L3 Automations, L4 Monitoring, L5 Custom Dashboards | Part 9 (planned) |

### Building Reliable Agents with LangSmith

| Academy lesson | Covered by |
|---|---|
| Intro: The Agent Development Lifecycle | Part 9 (planned) |
| M1 L1 Observability, L2 Tracing with LangSmith, L3 Analyzing your Agent | Part 5 (planned) |
| M2 L1 Evaluating Agents, L2 Creating Datasets, L3 Running Experiments | Part 5 (planned) |
| M2 L4 Code-based Eval, L5 LLM-as-Judge, L6 Pairwise Evaluations | Part 5 (planned) |
| M3 L1 Moving Towards Production, L2 Insights Agent, L3 Online Evals, L4 Automations | Part 9 (planned) |

### Quickstart: LangSmith Essentials

| Academy lesson | Covered by |
|---|---|
| M1 What is Tracing?, Debugging with Studio, Capstone | Part 5 (planned) |
| M2 Datasets and Offline Evals, Experiments, Capstone | Part 5 (planned) |
| M3 Deployments | Part 8 (planned) |
| M4 Online Evaluators and Automations, Annotation Queues, Insights, Dashboards, Capstone | Part 9 (planned) |
| M5 Closing the Loop, Automating the ADLC with LangSmith Engine | Part 9 (planned) |

### Project: Ambient Agents with LangGraph

| Academy lesson | Covered by |
|---|---|
| M1 LangGraph 101 | Part 3 (planned) |
| M2 Building Agents, M3 Agent Evaluations, M4 Human-in-the-Loop, M5 Memory | Part 6 (planned) |
| M6 Deployment | Part 8 (planned) |

### Project: Deep Research with LangGraph

| Academy lesson | Covered by |
|---|---|
| M2 Scoping, M3 Research Agent, M4 Research Agent with MCP | Part 6 (planned) |
| M5 Research Supervisor, M6 Multi-Agent Research System | Part 6 (planned) |

### Introduction to Deep Agents

| Academy lesson | Covered by |
|---|---|
| M1 Overview, Running a Deep Agent, Models, The System Prompt, Tools, MCP | Part 7 (planned) |
| M1 Messages, Threads, and Checkpointers; HITL; Test Your Skills | Part 7 (planned) |
| M2 The Deep Agent Environment, Filesystem Backends, Sandboxes and LocalShell, Interpreter | Part 7 (planned) |
| M3 Summarization and Offload, Skills, Memory | Part 7 (planned) |
| M4 Delegation, Building a Subagent Team, Dynamic Subagents | Part 7 (planned) |
| M5 Local Deployment, The Sales Assistant, Asynchronous Subagents, The Advanced Sales Assistant | Part 7 (planned) |
| M6 What is MDA?, MDA Project Structure | Part 7 (planned) |

### dcode Essentials

| Academy lesson | Covered by |
|---|---|
| Welcome to dcode, dcode Essentials, Goals | Part 7 (planned) |
| Skills, MCP, Plugins, and Marketplaces; Configuring Subagents; dcode for Enterprise | Part 7 (planned) |

### LangSmith Fleet (no-code agents)

| Academy lesson | Covered by |
|---|---|
| L1 Building An Agent, L2 Human-In-The-Loop, L3 What Can Agents Do? | Part 7 (planned) |
| L4 Skills, L5 MCP, L6 Ambient Agents, L7 Sub-Agents, L8 Computer Use (Sandboxes), L9 Templates | Part 7 (planned) |

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
