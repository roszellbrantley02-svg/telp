"""
lattice/paths.py - where Telp keeps his state on disk.

Everything Telp remembers lives under ONE state directory: <repo>/state by
default, or $TELP_STATE_DIR when set (the test suite points it at a temp
dir so tests never touch real memories).

MEMORY_DB is the one memory every command shares - chat, ask, teach, learn,
see, watch, forget. Earlier versions let chat fall back to a second file
(standalone_lattice.db) until the first learn/see/watch created this one,
which silently orphaned everything taught before; store.merge_legacy_memory
folds such a file back in.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.environ.get("TELP_STATE_DIR")
                 or ROOT / "state").expanduser().resolve()

MEMORY_DB = STATE_DIR / "concept_bridge.db"
LEGACY_MEMORY_DB = STATE_DIR / "standalone_lattice.db"


def state_path(*parts: str) -> Path:
    """A path under the state directory, e.g. state_path("persona.db")."""
    return STATE_DIR.joinpath(*parts)
