---
name: log-analysis
description: Analyse a Nimbus Notes sync client log to find why sync fails. Use when a customer sends a log file or a ticket mentions sync errors, stuck sync, or missing notes.
---

# Sync log analysis

Client logs are text, one event per line: `<timestamp> <LEVEL> <component> <message>`.

## Steps

1. Count lines per level (ERROR, WARN). Don't read a whole long log into the conversation: grep for
   `ERROR` and `WARN`, or run `count_levels.py` (next to this file) if you can execute code.
2. Group the errors by message. The most frequent one is usually the cause; the rest follow from it.
3. Match the codes:
   - `E401 token expired`: the session expired. Fix: sign out and back in (`07-troubleshooting.md`).
   - `E413 payload too large`: an attachment is over the plan's limit (100 MB Free, 1 GB paid;
     `05-import-export.md`). Fix: remove or split it.
   - `E503 upstream unavailable`: a service incident. Check the status page; nothing to fix locally.
   - `resume after sleep` followed by `offline indicator`: the known tray-icon issue in 4.12 (cosmetic).
4. Report: the cause, the evidence (counts and one example line), the fix, and the doc file that covers it
   (most are in `07-troubleshooting.md` or `03-sync-and-offline.md`).
