---
name: add-a-lesson
description: Add a new lesson to the DocsHelp curriculum, with its test and curriculum entries. Use whenever you create or rename a file under lessons/.
---

# Adding a lesson

1. Write the failing test first, in `tests/test_part<N>_<topic>.py`. Drive the lesson's `run()` with
   `FakeChatModel(replies=[...])` from `tests/conftest.py`; use `tool_call(name, args)` for tool calls.
2. Create `lessons/p<N>_<topic>/<NN>_<slug>.py`:
   - docstring: title line, `Academy:` line naming the courses' lessons, `Concepts:` bullets, `Run:` command
   - `run(llm, ...)`: all dependencies injectable; no side effects at import
   - `main()`: calls `ensure_llm_ready()`, prints `describe()`, wires in `get_llm()` and real dependencies
3. Put anything reusable in `src/docshelp/` and add the module to the hygiene list in
   `tests/test_repo_hygiene.py` (`APP_CODE`).
4. Map the lesson in `CURRICULUM.md` (replace the part's "(planned)" rows) and add it to the part's README table.
5. Run `uv run poe test` and `uv run poe e2e`. The real-model e2e run discovers new lessons by itself.
