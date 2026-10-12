# DocsHelp, for Deep Agents Code

This is the project memory Deep Agents Code (`dcode`) loads when it works in this repo. The shared rules for
every coding agent are in `AGENTS.md` at the repo root: read it before changing anything.

The essentials:

- Lessons live in `lessons/p<part>_<topic>/<NN>_<slug>.py`. Every lesson has a docstring (title, Academy lessons,
  concepts, run command), a `run(llm, ...)` that tests drive with `FakeChatModel`, and a `main()` that calls
  `ensure_llm_ready()`.
- Reusable code goes in `src/docshelp/`, never in lessons.
- App code never names a model vendor or model; use `get_llm()`.
- Before saying a change is done, run `uv run poe test` and `uv run poe e2e`, and report the results.
- Everything must stay free to run: Ollama by default, no paid APIs.
