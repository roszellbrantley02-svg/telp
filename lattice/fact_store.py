"""
lattice/fact_store.py - facts on disk, in the one memory file.

Facts live in a `facts` table next to `memories` in the same SQLite file,
so the one-memory rule holds: every fact points back at the memory row
(sentence) it was read from, and forgetting that row forgets its facts.
Structured imports (e.g. Wikidata) are facts with no memory row; their
source tag is their citation.

sync() keeps the table in step with the memory incrementally: new rows are
read by the extractor once (a high-water mark in telp_meta remembers how
far it got), facts of deleted rows are dropped, and a new extractor
version re-reads everything. It is safe for several processes (chat,
`telp teach`, the daemon) to sync the same file: facts are unique per
(subject, relation, obj, memory_id).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Iterable

from lattice.facts import Fact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject     TEXT NOT NULL,
    relation    TEXT NOT NULL,
    obj         TEXT NOT NULL,
    source      TEXT,
    text        TEXT,
    memory_id   INTEGER,
    created_at  TEXT,
    qualifiers  TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_unique
    ON facts(subject, relation, obj, IFNULL(memory_id, -1));
CREATE INDEX IF NOT EXISTS idx_facts_memory ON facts(memory_id);
CREATE TABLE IF NOT EXISTS telp_meta (key TEXT PRIMARY KEY, value TEXT);
"""

# rows that are not statements about the world: conversation echoes,
# perception logs, and stories Telp made up
NOT_FACT_SOURCES = ("user_msg", "agent_response", "conversation_turn",
                    "image:", "video:", "story:")

_MARK = "facts:last_memory_id"
_VERSION = "facts:extractor_version"


def _row_to_fact(r) -> Fact:
    subject, relation, obj, source, text, memory_id, created_at, quals = r
    q = tuple(sorted(json.loads(quals).items())) if quals else ()
    return Fact(subject, relation, obj, source or "", text or "", memory_id,
                created_at, q)


class FactStore:
    """The facts table of one memory file."""

    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self.con.executescript(_SCHEMA)
        self.con.commit()

    # ── meta ───────────────────────────────────────────────────────
    def _get(self, key: str, default: str = "") -> str:
        row = self.con.execute("SELECT value FROM telp_meta WHERE key=?",
                               (key,)).fetchone()
        return row[0] if row else default

    def _set(self, key: str, value: str) -> None:
        self.con.execute("INSERT OR REPLACE INTO telp_meta (key, value) "
                         "VALUES (?, ?)", (key, value))

    # ── reading ────────────────────────────────────────────────────
    def all_facts(self) -> list[Fact]:
        rows = self.con.execute(
            "SELECT subject, relation, obj, source, text, memory_id, "
            "created_at, qualifiers FROM facts ORDER BY id").fetchall()
        return [_row_to_fact(r) for r in rows]

    def count(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # ── writing ────────────────────────────────────────────────────
    def add(self, facts: Iterable[Fact]) -> int:
        """Insert facts (duplicates ignored). Returns rows inserted."""
        n = 0
        for f in facts:
            cur = self.con.execute(
                "INSERT OR IGNORE INTO facts (subject, relation, obj, source, "
                "text, memory_id, created_at, qualifiers) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f.subject, f.relation, f.obj, f.source, f.text, f.memory_id,
                 f.created_at,
                 json.dumps(dict(f.qualifiers)) if f.qualifiers else None))
            n += cur.rowcount
        self.con.commit()
        return n

    def remove_source(self, source: str) -> int:
        """Drop structured facts by source tag (e.g. a Wikidata import)."""
        n = self.con.execute("DELETE FROM facts WHERE source=? AND "
                             "memory_id IS NULL", (source,)).rowcount
        self.con.commit()
        return n

    # ── keeping in step with the memory ────────────────────────────
    def sync(self, extract_rows, version: str) -> tuple[int, int]:
        """Read new memory rows into facts and drop facts of deleted rows.

        extract_rows(rows) -> list[Fact], rows being (memory_id, text,
        source, created_at) in id order. Returns (added, removed)."""
        removed = 0
        if self._get(_VERSION) != version:
            # the extractor changed: re-read every sentence
            removed += self.con.execute(
                "DELETE FROM facts WHERE memory_id IS NOT NULL").rowcount
            self._set(_MARK, "0")
            self._set(_VERSION, version)
        removed += self.con.execute(
            "DELETE FROM facts WHERE memory_id IS NOT NULL AND memory_id "
            "NOT IN (SELECT id FROM memories)").rowcount
        last = int(self._get(_MARK, "0") or 0)
        rows = self.con.execute(
            "SELECT id, text, source, created_at FROM memories WHERE id > ? "
            "ORDER BY id", (last,)).fetchall()
        rows = [r for r in rows
                if not (r[2] or "").startswith(NOT_FACT_SOURCES)]
        added = 0
        if rows:
            added = self.add(extract_rows(rows))
        top = self.con.execute("SELECT MAX(id) FROM memories").fetchone()[0]
        if top is not None and top > last:
            self._set(_MARK, str(top))
        self.con.commit()
        return added, removed
