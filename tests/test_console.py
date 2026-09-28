"""Output must not crash on terminals/pipes whose default encoding isn't UTF-8 (e.g. Windows cp1252)."""

import io
from pathlib import Path

import pytest

from docshelp.console import ensure_utf8

ROOT = Path(__file__).resolve().parents[1]


def test_non_utf8_stream_is_switched_to_utf8():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        stream.write("✓")  # the bug this fixes

    ensure_utf8(stream)
    stream.write("✓ ⏸ ═")
    stream.flush()
    assert raw.getvalue().decode("utf-8") == "✓ ⏸ ═"


def test_streams_that_cannot_be_reconfigured_are_left_alone():
    ensure_utf8(io.StringIO(), None)


def test_config_sets_up_utf8_output_on_import():
    assert "ensure_utf8()" in (ROOT / "src" / "docshelp" / "config.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("path", sorted((ROOT / "stages").glob("*.py")), ids=lambda p: p.name)
def test_every_stage_gets_utf8_output(path):
    source = path.read_text(encoding="utf-8")
    assert "from docshelp.config import" in source or "ensure_utf8()" in source
