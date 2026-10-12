---
name: maintain-curriculum
description: Periodic maintenance of the DocsHelp curriculum. Checks the LangChain Academy catalogue for new or changed courses and lessons, dependency updates, CI hang stack dumps, and curriculum consistency, then proposes or makes the updates. Use when asked to update the repo from the Academy, bump dependencies, or run maintenance.
---

# Maintain the curriculum

Follow `MAINTENANCE.md` in the repo root. It's the source of truth, and this skill only adds how to run it as an agent.

1. **Report:** run `uv run poe maintenance` (add `--no-ci` if `gh` isn't signed in). Read `data/reports/maintenance.md`.
2. **Academy:**
   - Fetch https://academy.langchain.com/collections and the page of each course named in `CURRICULUM.md`'s coverage matrix, plus any new course in the report.
   - Compare their lessons with `CURRICULUM.md`, and list the differences for the user before writing any lesson.
   - New lessons follow `AGENTS.md`: test first, the lesson shape, the DocsHelp use case, zero cost.
3. **Dependencies:** if the report lists updates, bump them on their own branch (`MAINTENANCE.md` step 3) and run both test suites.
4. **CI hangs:** for each stack dump in the report, find the stuck test and the cause from the dump (`MAINTENANCE.md` step 4).
5. **Report back:** say what changed upstream and what you changed, give the test results, and link the PRs. Keep Academy changes, dependency bumps and fixes in separate PRs.

Ask before changing anything outside the repo, and never merge PRs yourself: `main` is protected, and the user merges.
