"""
lattice/wiktionary_ingest.py — Phase 15: ingest English Wiktionary.

WHY
---
A dictionary is a knowledge representation an LLM cannot match for
provability.  Each Wiktionary entry IS a frame: word -> {pos,
definitions, synonyms, hypernyms, hyponyms, etymology, examples,
pronunciation, related}.  Telp ingests those frames into a SQLite
index for fast retrieval, then his conversation engine retrieves
+ composes from this knowledge — no sampling, no learned
distribution, no hallucination.

INPUT
-----
state/wiktionary/raw.jsonl — the kaikki.org pre-parsed English
Wiktionary dump (one JSON entry per line, ~2.8 GB / ~1M entries).

OUTPUT
------
state/wiktionary/dict.db — SQLite with:

  entries (word, pos, sense_idx, gloss, ...) — one row per word
    sense, lookup-able by exact match or LIKE prefix
  related  (word, relation, target)         — synonyms, antonyms,
    hypernyms, hyponyms, derived, meronyms, coordinate
  sounds   (word, ipa, audio)               — pronunciation
  etym     (word, text)                     — etymology prose

This is the "long-term memory" layer.  Telp's existing
ReadingLattice (lattice.line_events + word_grounding) stays the
"working memory" — when a conversation needs a word, we look it up
in dict.db, optionally bind a fresh HV for it, use it for the
response.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

_TELP_ROOT = Path(__file__).resolve().parents[1]
if str(_TELP_ROOT) not in sys.path:
    sys.path.insert(0, str(_TELP_ROOT))


# ─── Storage layout ───────────────────────────────────────────────


DEFAULT_RAW   = _TELP_ROOT / "state" / "wiktionary" / "raw.jsonl"
DEFAULT_DB    = _TELP_ROOT / "state" / "wiktionary" / "dict.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  word        TEXT NOT NULL,
  pos         TEXT NOT NULL,
  sense_idx   INTEGER NOT NULL,      -- index within ONE source entry;
                                      -- NOT unique across entries that
                                      -- share (word, pos) — Wiktionary
                                      -- often has multiple per word
                                      -- (different etymologies).  Using
                                      -- it as a tiebreaker only.
  gloss       TEXT,
  tags        TEXT,                  -- JSON array of sense tags
  examples    TEXT                   -- JSON array of example sentences
);

CREATE INDEX IF NOT EXISTS idx_entries_word ON entries(word);

CREATE TABLE IF NOT EXISTS related (
  word        TEXT NOT NULL,
  relation    TEXT NOT NULL,         -- synonym / antonym / hypernym /
                                      -- hyponym / derived / meronym /
                                      -- coordinate / holonym
  target      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_related_word     ON related(word);
CREATE INDEX IF NOT EXISTS idx_related_target   ON related(target);
CREATE INDEX IF NOT EXISTS idx_related_relation ON related(relation);

CREATE TABLE IF NOT EXISTS sounds (
  word        TEXT NOT NULL,
  ipa         TEXT,
  audio       TEXT,
  PRIMARY KEY (word)
);

CREATE TABLE IF NOT EXISTS etym (
  word        TEXT NOT NULL,
  pos         TEXT NOT NULL,
  text        TEXT,
  PRIMARY KEY (word, pos)
);

CREATE TABLE IF NOT EXISTS meta (
  k TEXT PRIMARY KEY,
  v TEXT
);
"""


# ─── Ingestion ───────────────────────────────────────────────────


_RELATION_FIELDS = (
    "synonyms", "antonyms", "hypernyms", "hyponyms",
    "derived", "related", "meronyms", "holonyms",
    "coordinate_terms",
)
_RELATION_LABEL = {
    "synonyms": "synonym",
    "antonyms": "antonym",
    "hypernyms": "hypernym",
    "hyponyms": "hyponym",
    "derived": "derived",
    "related": "related",
    "meronyms": "meronym",
    "holonyms": "holonym",
    "coordinate_terms": "coordinate",
}


def _stream_entries(raw_path: Path) -> Iterator[dict]:
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _extract_ipa(entry: dict) -> Optional[str]:
    sounds = entry.get("sounds") or []
    for s in sounds:
        if isinstance(s, dict) and s.get("ipa"):
            return s["ipa"]
    return None


def _extract_audio(entry: dict) -> Optional[str]:
    sounds = entry.get("sounds") or []
    for s in sounds:
        if isinstance(s, dict) and s.get("audio"):
            return s["audio"]
    return None


def _extract_related_pairs(entry: dict) -> list[tuple[str, str]]:
    """Pull related-word mentions into (relation, target) pairs."""
    pairs = []
    for field in _RELATION_FIELDS:
        items = entry.get(field) or []
        rel = _RELATION_LABEL[field]
        for item in items:
            if isinstance(item, dict):
                target = item.get("word") or item.get("alt")
            else:
                target = str(item)
            if target and isinstance(target, str):
                target = target.strip()
                if target and not target.startswith("(") and len(target) < 80:
                    pairs.append((rel, target))
        # Also walk sense-level relations
    for sense in entry.get("senses") or []:
        for field in _RELATION_FIELDS:
            items = sense.get(field) or []
            rel = _RELATION_LABEL[field]
            for item in items:
                if isinstance(item, dict):
                    target = item.get("word") or item.get("alt")
                else:
                    target = str(item)
                if target and isinstance(target, str):
                    target = target.strip()
                    if target and not target.startswith("(") and len(target) < 80:
                        pairs.append((rel, target))
    # Dedupe preserving order
    seen = set()
    out = []
    for p in pairs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def ingest(raw_path: Path = DEFAULT_RAW,
                db_path:  Path = DEFAULT_DB,
                limit:    Optional[int] = None,
                verbose:  bool = True,
                commit_every: int = 5000) -> dict:
    """Build dict.db from a Wiktionary JSONL dump.

    Returns a stats dict {entries, senses, related, sounds, etym,
    elapsed_secs}.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()   # fresh build

    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    # Insert speed: defer index updates + journal
    con.executescript("""
        PRAGMA synchronous = OFF;
        PRAGMA journal_mode = MEMORY;
        PRAGMA temp_store = MEMORY;
    """)
    cur = con.cursor()

    n_entries = n_senses = n_related = n_sounds = n_etym = 0
    t0 = time.time()
    last_print = t0

    for i, entry in enumerate(_stream_entries(raw_path)):
        if limit is not None and i >= limit:
            break

        word = entry.get("word")
        pos  = entry.get("pos", "unknown")
        if not word:
            continue
        word_l = word.lower()

        # Senses + glosses
        senses = entry.get("senses") or []
        for sidx, sense in enumerate(senses):
            glosses = sense.get("glosses") or []
            if not glosses:
                continue
            gloss = glosses[0]
            tags = sense.get("tags") or []
            examples = []
            for ex in (sense.get("examples") or []):
                if isinstance(ex, dict):
                    txt = ex.get("text")
                    if txt:
                        examples.append(txt)
                elif isinstance(ex, str):
                    examples.append(ex)
            cur.execute(
                "INSERT INTO entries "
                "(word, pos, sense_idx, gloss, tags, examples) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (word_l, pos, sidx, gloss,
                  json.dumps(tags), json.dumps(examples[:5])),
            )
            n_senses += 1

        # Related-word pairs
        for rel, target in _extract_related_pairs(entry):
            cur.execute(
                "INSERT INTO related (word, relation, target) "
                "VALUES (?, ?, ?)",
                (word_l, rel, target.lower()),
            )
            n_related += 1

        # Sounds (IPA + audio)
        ipa = _extract_ipa(entry)
        audio = _extract_audio(entry)
        if ipa or audio:
            cur.execute(
                "INSERT OR REPLACE INTO sounds (word, ipa, audio) "
                "VALUES (?, ?, ?)",
                (word_l, ipa, audio),
            )
            n_sounds += 1

        # Etymology
        etym_text = entry.get("etymology_text")
        if etym_text:
            cur.execute(
                "INSERT OR REPLACE INTO etym (word, pos, text) "
                "VALUES (?, ?, ?)",
                (word_l, pos, etym_text[:8000]),  # truncate huge ones
            )
            n_etym += 1

        n_entries += 1

        if n_entries % commit_every == 0:
            con.commit()
            if verbose and (time.time() - last_print > 5.0):
                rate = n_entries / max(time.time() - t0, 0.001)
                print(f"  ingested {n_entries:>8,} entries  "
                          f"({rate:.0f}/s, senses={n_senses:,})", flush=True)
                last_print = time.time()

    con.commit()

    # Save metadata
    cur.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)",
                  ("ingest_time", time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                              time.gmtime())))
    cur.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)",
                  ("source", str(raw_path)))
    cur.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)",
                  ("n_entries", str(n_entries)))
    con.commit()
    con.close()

    elapsed = time.time() - t0
    stats = {
        "entries":      n_entries,
        "senses":       n_senses,
        "related":      n_related,
        "sounds":       n_sounds,
        "etym":         n_etym,
        "elapsed_secs": round(elapsed, 1),
        "db_path":      str(db_path),
        "db_mb":        round(db_path.stat().st_size / 1e6, 1),
    }
    if verbose:
        print(f"\n  done: {stats}")
    return stats


# ─── CLI ─────────────────────────────────────────────────────────


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--db",  default=str(DEFAULT_DB))
    ap.add_argument("--limit", type=int, default=None,
                          help="cap entries (for testing)")
    args = ap.parse_args()

    stats = ingest(Path(args.raw), Path(args.db),
                          limit=args.limit, verbose=True)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
