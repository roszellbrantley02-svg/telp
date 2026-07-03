"""
lattice/dictionary_lookup.py — Phase 15: fast retrieval over dict.db.

Wraps the SQLite index built by wiktionary_ingest with a clean
Python API.  Used by the conversation engine.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_TELP_ROOT = Path(__file__).resolve().parents[1]
if str(_TELP_ROOT) not in sys.path:
    sys.path.insert(0, str(_TELP_ROOT))

from lattice.wiktionary_ingest import DEFAULT_DB


# ─── Data classes ────────────────────────────────────────────────


@dataclass
class Sense:
    word:      str
    pos:       str
    sense_idx: int
    gloss:     str
    tags:      list[str]   = field(default_factory=list)
    examples:  list[str]   = field(default_factory=list)


@dataclass
class Entry:
    """All senses + relations + sounds + etymology for one word."""
    word:     str
    senses:   list[Sense]               = field(default_factory=list)
    related:  dict[str, list[str]]      = field(default_factory=dict)
    ipa:      Optional[str]             = None
    audio:    Optional[str]             = None
    etymology: dict[str, str]            = field(default_factory=dict)

    def primary_pos(self) -> Optional[str]:
        return self.senses[0].pos if self.senses else None

    def primary_gloss(self) -> Optional[str]:
        return self.senses[0].gloss if self.senses else None

    def has_pos(self, pos: str) -> bool:
        return any(s.pos == pos for s in self.senses)

    def glosses_for_pos(self, pos: str) -> list[str]:
        return [s.gloss for s in self.senses if s.pos == pos]


# ─── Lookup ──────────────────────────────────────────────────────


class Dictionary:
    """Read-only access to state/wiktionary/dict.db.

    Open once, query many.  All lookups are case-insensitive.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Dictionary not built yet: {self.db_path}.  "
                f"Run: python -m lattice.wiktionary_ingest")
        # check_same_thread=False so the same Dictionary can be reused
        # from the read-only chat loop without per-call open/close
        self._con = sqlite3.connect(self.db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row

    def close(self):
        try: self._con.close()
        except Exception: pass

    # ── Counts (for status reporting) ─────────────────────────

    def stats(self) -> dict:
        c = self._con.cursor()
        out = {}
        for t in ("entries", "related", "sounds", "etym"):
            n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            out[t] = n
        out["distinct_words"] = c.execute(
            "SELECT COUNT(DISTINCT word) FROM entries").fetchone()[0]
        out["db_mb"] = round(self.db_path.stat().st_size / 1e6, 1)
        return out

    # ── Per-word lookups ─────────────────────────────────────

    def lookup(self, word: str) -> Optional[Entry]:
        """Get the full entry for an exact word match (case-insensitive)."""
        w = word.lower().strip()
        if not w:
            return None
        c = self._con.cursor()

        senses_rows = c.execute(
            "SELECT word, pos, sense_idx, gloss, tags, examples "
            "FROM entries WHERE word = ? "
            "ORDER BY pos, sense_idx", (w,)).fetchall()
        if not senses_rows:
            return None
        senses = []
        for r in senses_rows:
            try:
                tags = json.loads(r["tags"]) if r["tags"] else []
            except Exception:
                tags = []
            try:
                examples = json.loads(r["examples"]) if r["examples"] else []
            except Exception:
                examples = []
            senses.append(Sense(
                word=r["word"], pos=r["pos"],
                sense_idx=r["sense_idx"], gloss=r["gloss"],
                tags=tags, examples=examples,
            ))

        related: dict[str, list[str]] = {}
        rel_rows = c.execute(
            "SELECT relation, target FROM related WHERE word = ? "
            "LIMIT 200", (w,)).fetchall()
        for r in rel_rows:
            related.setdefault(r["relation"], []).append(r["target"])
        # Dedupe each relation list, cap at 20
        for k in list(related.keys()):
            seen, out = set(), []
            for x in related[k]:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
                if len(out) >= 20:
                    break
            related[k] = out

        sound_row = c.execute(
            "SELECT ipa, audio FROM sounds WHERE word = ?", (w,)).fetchone()
        ipa = sound_row["ipa"] if sound_row else None
        audio = sound_row["audio"] if sound_row else None

        etym_rows = c.execute(
            "SELECT pos, text FROM etym WHERE word = ?", (w,)).fetchall()
        etymology = {r["pos"]: r["text"] for r in etym_rows}

        return Entry(word=w, senses=senses, related=related,
                          ipa=ipa, audio=audio, etymology=etymology)

    def has(self, word: str) -> bool:
        c = self._con.cursor()
        return bool(c.execute(
            "SELECT 1 FROM entries WHERE word = ? LIMIT 1",
            (word.lower().strip(),)).fetchone())

    # ── Cross-word queries ───────────────────────────────────

    def reverse_related(self, target_word: str,
                                relation: Optional[str] = None,
                                limit: int = 50) -> list[tuple[str, str]]:
        """Find words that have `target_word` as their RELATED.

        E.g. reverse_related("animal", "hypernym") -> all words that
        list "animal" as a hypernym (i.e. all types of animal).
        Returns [(word, relation), ...].
        """
        c = self._con.cursor()
        t = target_word.lower().strip()
        if relation:
            rows = c.execute(
                "SELECT word, relation FROM related "
                "WHERE target = ? AND relation = ? LIMIT ?",
                (t, relation, limit)).fetchall()
        else:
            rows = c.execute(
                "SELECT word, relation FROM related "
                "WHERE target = ? LIMIT ?",
                (t, limit)).fetchall()
        return [(r["word"], r["relation"]) for r in rows]

    def search_prefix(self, prefix: str, limit: int = 25) -> list[str]:
        """Distinct words starting with `prefix`."""
        c = self._con.cursor()
        rows = c.execute(
            "SELECT DISTINCT word FROM entries WHERE word LIKE ? "
            "ORDER BY word LIMIT ?",
            (prefix.lower() + "%", limit)).fetchall()
        return [r["word"] for r in rows]


# ─── Smoke test ──────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="*",
                          default=["bear", "lonely", "forest", "rabbit"])
    args = ap.parse_args()

    d = Dictionary()
    print(f"Dictionary stats: {d.stats()}\n")
    for w in args.words:
        e = d.lookup(w)
        if e is None:
            print(f"  {w!r}: not found")
            continue
        print(f"  {w!r}")
        for s in e.senses[:3]:
            print(f"    [{s.pos}] {s.gloss}")
        if e.related:
            for rel, targets in list(e.related.items())[:3]:
                print(f"    {rel}: {', '.join(targets[:6])}")
        if e.ipa:
            print(f"    /{e.ipa}/")
        print()
