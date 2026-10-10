# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. `CLAUDE.md` imports this file.

## What this is

DocsHelp is a teaching project: a support assistant for the fictional product "Nimbus Notes" that grows,
lesson by lesson, until it covers everything in the LangChain Academy courses (LangChain, LangGraph, LangSmith,
Deep Agents, deployment). `CURRICULUM.md` maps every Academy lesson to the lesson here that teaches it.
Each concept is taught once and only built on afterwards, never repeated. Everything must run at zero cost
(Ollama by default; paid-only LangSmith features get local twins). Clarity for learners matters as much as
correctness. Keep code small, commented where the concept is non-obvious, and consistent with its part.

**Work in progress:** the curriculum is being built one part per PR. `ROADMAP.md` holds the plan, the current
status and the steps for resuming in a new session. Read it before starting curriculum work.

## Commands

Every command works the same in PowerShell, cmd, Git Bash, macOS and Linux. Don't add shell-specific scripts.

| Task | Command |
|---|---|
| Install (all providers) | `uv sync --all-extras` (plain `uv sync` removes extras) |
| List tasks | `uv run poe` |
| Unit + integration tests (~1 min, offline) | `uv run poe test` |
| End-to-end tests (~2 min, offline) | `uv run poe e2e` |
| Real-model E2E (no keys) | `DOCSHELP_E2E_MODEL=qwen2.5:0.5b uv run poe e2e` with Ollama running and the model pulled |
| Provider setup check | `uv run poe doctor` |
| List / run lessons | `uv run poe lessons`, `uv run poe lesson 1.6` (extra args are passed to the lesson) |
| Run a legacy stage | `uv run poe stage3` … `stage5` (`stage5` needs `uv run poe server`) |
| Docker equivalents | `docker compose build`, then `docker compose run --rm app poe <task>` |

Before finishing any change, run `uv run poe test` and `uv run poe e2e`. Both must pass.

## Layout

- `src/docshelp/providers/`: `LLMProvider` base class (Strategy + Template Method), `ProviderRegistry`
  (Registry/Factory), and one class per vendor in `vendors.py`.
- `src/docshelp/config.py`: `Settings.from_env()` plus the `get_llm()`, `structured_llm()` and
  `ensure_llm_ready()` facades, and local fastembed embeddings.
- `src/docshelp/graph.py`: the LangGraph agent. Dependencies (llm, retriever, checkpointer, provider) are
  injectable for tests.
- `src/docshelp/agents.py`: the `create_agent` support agent that later parts extend.
- `src/docshelp/{ingest,tools,console,tasks,server}.py`: RAG pipeline, tools, UTF-8 output, `poe` helpers
  (including the lesson runner), and the LangGraph Server entry point.
- `lessons/p<part>_<topic>/<NN>_<topic>.py`: the curriculum. `stages/`: stages 3–5, until their parts replace them.
- `tests/`: unit tests (one `test_part<N>_*.py` per part). `tests/e2e/`: process-level and server tests.

## Rules

1. **Provider-neutral app code.** Lessons, stages, `graph.py` and `agents.py` must never name a vendor, model
   or structured-output method. Use `get_llm()`, `get_llm(fast=True)` and `structured_llm(llm, Schema)`. A hygiene test enforces this.
2. **Adding a provider** takes three steps and changes no existing code:
   - add a subclass with `@registry.register` in `providers/vendors.py`
   - add a pyproject extra with the same name
   - add its key variable to `.env.example` with an empty value

   Tests check that the extra and the key line exist.
3. **Configuration comes only from environment variables** (optionally via a git-ignored `.env`):
   - `DOCSHELP_PROVIDER` picks the provider.
   - `DOCSHELP_MODEL`, `DOCSHELP_FAST_MODEL`, `DOCSHELP_VISION_MODEL` and `DOCSHELP_MAX_TOKENS` override the defaults.
   - Each provider reads its own key variable.

   Never hard-code keys, and never read keys anywhere except through a provider's `api_key_envs`. New settings
   go in `Settings.from_env()` and `.env.example`.
4. **No machine-specific values:** no absolute local paths, usernames or personal emails in code or docs. Use
   `pathlib` and `config.PROJECT_ROOT`. `tests/test_repo_hygiene.py` scans the repo for them.
5. **Tests never need network or API keys**, except the opt-in real-model E2E tests, which skip unless
   `DOCSHELP_E2E_MODEL` is set. Use `tests/conftest.py::FakeChatModel` (or `ScriptedLLM` for `graph.py`) and
   the `fake_retriever` fixture. Inject fake environments as a `dict` rather than editing `os.environ`.
6. **Work test-first (red → green → refactor).** Write or extend a failing test, make it pass, then clean up.
7. **Lessons follow one shape.** A docstring whose first line is the title, then the Academy lessons covered,
   the concepts, and the run command. A `run(llm, ...)` with injectable dependencies that tests drive with
   `FakeChatModel`, and a `main()` that calls `ensure_llm_ready()` and wires in real ones. No side effects at
   import. New lessons must be added to `CURRICULUM.md`, and a shipped part keeps no "(planned)" rows there.
   Reusable app code goes in `src/docshelp/`, not in lessons.
8. **Cross-platform by default.**
   - Write helpers in Python (see `tasks.py`), not PowerShell or bash.
   - Keep LF line endings (`.gitattributes`).
   - Keep stdout UTF-8 (`console.ensure_utf8()`, which importing `docshelp.config` already calls).
9. **Model defaults go stale.** When updating a default model, change `providers/vendors.py` and the
   provider table in `README.md` together.
10. **Anthropic specifics:** don't add sampling parameters (`temperature`, `top_p`, `top_k`) or a thinking
    `budget_tokens` to the Anthropic provider, because current Claude models reject them with HTTP 400.

## Git workflow

- `main` is protected. Every change goes through a **pull request** from a branch, and the CI checks
  `test (ubuntu-latest)`, `test (windows-latest)`, `test (macos-latest)` and `docker-e2e` must pass.
  Direct pushes, force pushes and branch deletion are blocked.
- Keep pull requests focused, with a clear description. Never commit `.env`, `.cache/`, `.venv/`,
  `data/tickets.jsonl`, `graph.mmd` or `.claude/settings.local.json`; they're all git-ignored.
