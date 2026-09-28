"""Make terminal output behave the same on every OS.

On Windows, Python writes to pipes and to non-native terminals (e.g. Git Bash's mintty) using the
legacy code page (cp1252), which can't encode characters like ✓ or ⏸ and crashes. macOS and Linux
default to UTF-8 already, so switching the standard streams to UTF-8 gives identical behaviour everywhere.
"""

from __future__ import annotations

import sys
from typing import TextIO


def ensure_utf8(*streams: TextIO | None) -> None:
    """Reconfigure the given streams (default: stdout and stderr) to UTF-8. Safe to call repeatedly."""
    for stream in streams or (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if encoding != "utf8":
            reconfigure(encoding="utf-8", errors="replace")
