# CLAUDE.md

@AGENTS.md

## Notes for Claude Code

- The shared rules, commands and workflow are in `AGENTS.md` (imported above). Keep project guidance there,
  so every coding agent sees the same instructions.
- The maintainer works on Windows in both PowerShell and Git Bash, so prefer the `uv run poe <task>` commands,
  which behave identically in both. When a shell-specific command is unavoidable in docs, show PowerShell and
  bash side by side.
- Never put absolute local paths in commands you'd like the user to "always allow"; those get saved to the
  git-ignored `.claude/settings.local.json`. Use `uv` from PATH and project-relative paths.
- Run `uv run poe test` and `uv run poe e2e` before reporting a change as done, and state the results.
