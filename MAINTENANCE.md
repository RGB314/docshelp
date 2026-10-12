# MAINTENANCE.md

How to keep DocsHelp current with the [LangChain Academy](https://academy.langchain.com/collections) and its
libraries. Run it every month or two, and after any LangChain release you care about. In Claude Code, run the
`/maintain-curriculum` skill, which follows these steps. Any other agent, or a person, can follow them by hand.

## 1. Get the report

```bash
uv run poe maintenance            # writes data/reports/maintenance.md
uv run poe maintenance --no-ci    # skip the CI log scan (it needs the GitHub CLI, signed in)
```

The report covers four things:

1. **Academy catalogue:** course links on the catalogue page that `data/academy_catalogue.json` doesn't list.
2. **Dependency updates:** what `uv lock --upgrade` would change. It's a dry run.
3. **CI hangs:** stack dumps that pytest's `faulthandler_timeout = 300` (in `pyproject.toml`) printed in recent
   failed or cancelled CI runs, and which test was stuck.
4. **Curriculum consistency:** every lesson mapped, and no "(planned)" rows.

## 2. Check the Academy, course by course

The catalogue page builds most of its list with JavaScript, so step 1 only catches what its HTML shows. For
each course heading in `CURRICULUM.md`, and each new course in the report:

1. Open the course page and list its modules and lessons.
2. Compare them with the course's table in `CURRICULUM.md`:
   - **New lesson or course:** decide where it belongs. An existing lesson may cover it, so extend that lesson
     and its docstring, and map the row. If the concept is new, it needs a new lesson in the part that first
     needs it. The rules in `AGENTS.md` apply: teach once, then build on it, use the DocsHelp use case, and
     keep it zero cost.
   - **Renamed or removed lesson:** update the row. Keep a lesson that still teaches something useful.
   - **Changed API in a course video:** check the lesson that covers it against the current library (step 3).
3. Add each reviewed course's slug to `data/academy_catalogue.json` and update `"checked"`.
4. Update the catalogue date at the top of `CURRICULUM.md` and `ROADMAP.md`.

## 3. Update dependencies

1. Run `git checkout -b chore/deps-<date>`, then `uv lock --upgrade` and `uv sync --all-extras`.
2. Run `uv run poe test` and `uv run poe e2e`. Read the changelogs of anything that fails; LangChain,
   LangGraph, deepagents, langgraph-api and langsmith move fastest.
3. Check the provider defaults in `src/docshelp/providers/vendors.py` and the README's provider table
   (AGENTS.md rule 9).
4. Open a PR. CI's `docker-e2e` shards are the first place every lesson runs against a real model. If one
   shard grows much slower than the rest, rebalance the `parts` in `.github/workflows/ci.yml`.

## 4. Act on CI hangs

A stack dump means a test sat for 5 minutes, which is almost always a deadlock, not slowness.

1. Read the dump: the thread inside `tests/...` is the stuck test, and the other threads show what it waits on.
2. Fix the cause: a lock, a thread pool or a subprocess. Don't raise the timeout.
3. Note recurring library gotchas in ROADMAP.md's "Practical notes". One example is LangGraph 1.2's
   checkpoint-save deadlock (`durability="sync"`, lesson 7.14).

## 5. Finish

1. Open one PR per kind of change (Academy updates, dependency bumps, fixes), so each can be reviewed alone.
2. In the PR description, say what the report found and what you changed.
