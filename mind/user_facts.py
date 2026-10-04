"""autopilot/user_facts.py — persistent memory of the user.

Stores things the user tells Telp about themselves across sessions.
Persists to state/user_facts.db so a chat next week can still
recall "you mentioned you trade MES" or "you said your daughter's
name is Lily."

Architecture mirrors PersonaStore but uses a USER_SELF subspace
(distinct from TELP_SELF).

The fact-capture path is HEURISTIC — we detect first-person
statements ("I'm a programmer", "my name is X", "I have two cats")
and add them.  Not perfect, but cheap and runs every turn.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import bind, hamming_distance, D
from mind.persona import _deterministic_random_vec


# USER_SELF — a subspace separate from TELP_SELF.
USER_SELF = _deterministic_random_vec("USER_SELF")


def bind_user(hv: np.ndarray) -> np.ndarray:
    return bind(USER_SELF, hv.astype(np.int8))


from lattice.paths import state_path  # noqa: E402
USER_FACTS_DB = state_path("user_facts.db")


_USER_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_facts (
    id          INTEGER PRIMARY KEY,
    text        TEXT NOT NULL,
    hv          BLOB NOT NULL,
    source_msg  TEXT,
    captured_at TEXT NOT NULL,
    superseded  INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_user_captured ON user_facts(captured_at);
"""


# ─── Heuristic fact extraction ───────────────────────────────────


# Patterns that signal the user is telling Telp something about themselves.
_FACT_PATTERNS = [
    # "my name is X"
    (re.compile(r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z'\- ]+)", re.IGNORECASE),
        "name"),
    # "i'm X" / "i am X"  (be conservative — only catch identity claims)
    (re.compile(r"\bi'?m\s+(?:a|an)\s+(\w+(?:\s+\w+){0,3})\b", re.IGNORECASE),
        "occupation"),
    (re.compile(r"\bi\s+am\s+(?:a|an)\s+(\w+(?:\s+\w+){0,3})\b", re.IGNORECASE),
        "occupation"),
    # "i live in X" / "i'm from X"
    (re.compile(r"\bi\s+live\s+in\s+([A-Za-z][A-Za-z\- ]+)", re.IGNORECASE),
        "location"),
    (re.compile(r"\bi'?m\s+from\s+([A-Za-z][A-Za-z\- ]+)", re.IGNORECASE),
        "location"),
    # "i have X" — kids, pets, work, etc.
    (re.compile(r"\bi\s+have\s+([\w\s\-]{3,40})\b", re.IGNORECASE),
        "possession"),
    # "i like / love / prefer X"
    (re.compile(r"\bi\s+(?:like|love|prefer|enjoy)\s+([\w\s\-]{3,40})\b",
                  re.IGNORECASE),
        "preference"),
    # "i work at / for X"
    (re.compile(r"\bi\s+work\s+(?:at|for)\s+([A-Za-z][A-Za-z\- &]+)",
                  re.IGNORECASE),
        "workplace"),
    # "my X is Y" - the general attribute ("my favorite color is blue",
    # "my dog's name is Astro")
    (re.compile(r"\bmy\s+([a-z][\w' ]{2,28}?)\s+is\s+"
                r"([\w][\w ,.'-]{0,40})", re.IGNORECASE),
        "attribute"),
    # "remember that X" / "remember X"
    (re.compile(r"\bremember\s+(?:that\s+)?(.+)$", re.IGNORECASE),
        "remember"),
]


# Values that look like a slot but aren't facts about the user:
# "I'm a little tired today" is a mood, not an occupation; "I have a
# question about Rome" is a request; "I like how you explained that" is
# feedback.
_NOT_OCCUPATION = {"little", "bit", "lot", "few", "big", "huge", "kind",
                   "sort", "very", "quite", "really", "pretty", "total",
                   "complete", "fan", "huge", "great", "good", "bad",
                   "new", "first", "second"}
_TIME_WORDS = re.compile(r"\b(today|tonight|now|right now|lately|again|"
                         r"anymore|at the moment|these days)\b", re.I)
_CLAUSE_END = re.compile(r"\s+(?:and|but|or|so|because|who|which|that|with|"
                         r"at|in|for|from|since|when|while|i|i'm)\b.*$", re.I)
_NOT_POSSESSION = re.compile(r"^(?:a\s+question|questions?|a\s+query|no\s+idea|"
                             r"to\b|been\b|got\b|never\b|always\b|just\b|"
                             r"seen\b|heard\b|read\b|done\b|a\s+feeling)",
                             re.I)
_NOT_PREFERENCE = re.compile(r"^(?:how|when|that|this|it|what|your|the\s+way|"
                             r"to\s+(?:think|know|ask|see))\b", re.I)
_NAME_STOP = {"and", "but", "or", "i", "i'm", "im", "who", "from", "so"}


_REQUEST_RX = re.compile(
    r"\b(?:can|could|would|will)\s+you\b|\bplease\b|"
    r"\b(?:tell|show|give|help|teach)\s+me\b|\bexplain\b|"
    r"\bi\s+have\s+(?:a\s+)?questions?\b")


_CLAUSE_SPLIT_RX = re.compile(
    r"(?<=[.!?;])\s+"
    r"|,\s*(?=(?:and\s+|but\s+|so\s+)?(?:can|could|would|will|please|tell|"
    r"show|give|help|explain|teach)\b)", re.I)


def _clauses(msg: str) -> list[str]:
    return [c for c in _CLAUSE_SPLIT_RX.split(msg) if c and c.strip()]


def _clean_name(value: str) -> str:
    """'Eric and I'm a developer' -> 'Eric'; 'Mary Jane Watson' kept."""
    words = value.split()
    out = words[:1]
    for w in words[1:]:
        if w.lower() in _NAME_STOP or not w[:1].isupper():
            break
        out.append(w)
    return " ".join(out)


def extract_user_facts(msg: str) -> list[dict]:
    """Pull self-referential statements out of a user message.

    Returns a list of {text, kind, raw, slot} dicts. `slot` is the text
    prefix shared by every value of the same fact ("User's name is "), so a
    newer value can supersede an older one.
    """
    if not msg:
        return []
    facts = []
    for pat, kind in _FACT_PATTERNS:
        for m in pat.finditer(msg):
            raw = m.group(0).strip()
            captured_value = m.group(1).strip().rstrip(".,!?")
            slot = None
            # Build a clean first-person fact text
            if kind == "name":
                captured_value = _clean_name(captured_value)
                text = f"User's name is {captured_value}."
                slot = "User's name is "
            elif kind == "occupation":
                captured_value = _CLAUSE_END.sub("", captured_value).strip()
                first = captured_value.split()[0].lower() if captured_value else ""
                if (not captured_value or first in _NOT_OCCUPATION
                        or _TIME_WORDS.search(captured_value)):
                    continue
                text = f"User is a {captured_value}."
                slot = "User is a "
            elif kind == "location":
                captured_value = _CLAUSE_END.sub("", captured_value).strip()
                text = f"User lives in / is from {captured_value}."
                slot = "User lives in / is from "
            elif kind == "possession":
                if _NOT_POSSESSION.search(captured_value):
                    continue
                text = f"User has {captured_value}."
            elif kind == "preference":
                if _NOT_PREFERENCE.search(captured_value):
                    continue
                text = f"User likes {captured_value}."
            elif kind == "workplace":
                text = f"User works at {captured_value}."
                slot = "User works at "
            elif kind == "attribute":
                attr = m.group(1).strip().lower()
                if attr in ("name",):        # covered by the name pattern
                    continue
                text = f"User's {attr} is {m.group(2).strip().rstrip('.,!?')}."
                slot = f"User's {attr} is "
            elif kind == "remember":
                text = f"User asked me to remember: {captured_value}."
            else:
                text = raw
            facts.append({"text": text, "kind": kind, "raw": raw,
                          "slot": slot})
    return facts


# ─── Store ───────────────────────────────────────────────────────


class UserFactsStore:
    """Persistent user-fact memory.  HDC-encoded, in USER_SELF subspace."""

    def __init__(self, db_path: Path = None, encoder=None):
        self.db_path = db_path or USER_FACTS_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self._con = sqlite3.connect(str(self.db_path),
                                          check_same_thread=False,
                                          timeout=30.0)
        self._con.executescript(_USER_SCHEMA)
        try:    # migrate pre-Wire-27 stores
            self._con.execute("ALTER TABLE user_facts ADD COLUMN "
                              "superseded INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        self._con.commit()
        self.last_superseded: list[str] = []
        self.last_had_request = False
        self._ids: list[int] = []
        self._texts: list[str] = []
        self._stack: Optional[np.ndarray] = None
        self._reload()

    def _reload(self):
        rows = self._con.execute(
            "SELECT id, text, hv FROM user_facts WHERE superseded=0 "
            "ORDER BY id"
        ).fetchall()
        self._ids   = [r[0] for r in rows]
        self._texts = [r[1] for r in rows]
        if rows:
            self._stack = np.stack(
                [np.frombuffer(r[2], dtype=np.int8) for r in rows]
            )
        else:
            self._stack = None

    def count(self) -> int:
        return len(self._ids)

    def add(self, text: str, source_msg: str = "") -> int:
        if self.encoder is None:
            raise RuntimeError("UserFactsStore: no encoder set")
        hv = bind_user(self.encoder.encode(text))
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._con.execute(
            "INSERT INTO user_facts (text, hv, source_msg, captured_at) "
            "VALUES (?, ?, ?, ?)",
            (text, hv.tobytes(), source_msg, ts),
        )
        self._con.commit()
        mid = cur.lastrowid
        self._ids.append(mid)
        self._texts.append(text)
        if self._stack is None:
            self._stack = hv[None, :].copy()
        else:
            self._stack = np.vstack([self._stack, hv[None, :]])
        return mid

    def capture(self, user_msg: str) -> list[int]:
        """Auto-extract facts from a user message and add them. Returns
        the IDs of facts added.  Dedupes against existing texts to avoid
        repeats."""
        if not user_msg:
            return []
        # questions and hypotheticals are not facts about the user
        # ("if I have 3 apples and eat one, how many are left?")
        low = user_msg.strip().lower()
        if low.startswith(("if ", "what if", "suppose", "imagine", "say ",
                           "forget", "don't remember", "do not remember")):
            return []
        # Judge each clause on its own: "My name is Eric. Can you help me
        # with my essay?" holds a fact AND a request. Questions and requests
        # aren't facts ("I have a question about Rome"), but asking him to
        # remember something is exactly how facts arrive.
        facts = []
        self.last_had_request = False
        for clause in _clauses(user_msg):
            c = clause.strip().lower()
            if not c:
                continue
            if c.endswith("?") or (_REQUEST_RX.search(c)
                                   and "remember" not in c):
                self.last_had_request = True    # caller must still answer
                continue
            facts += extract_user_facts(clause)
        existing = set(self._texts)
        added = []
        self.last_superseded = []
        # a new value for the SAME slot supersedes the old one - kept as
        # history, no longer served ("my favorite color is green" after
        # "...is blue": he remembers both, believes the newer)
        for f in facts:
            if f["text"] in existing:
                continue
            slot = f.get("slot")
            if slot:
                olds = self._con.execute(
                    "SELECT id, text FROM user_facts WHERE superseded=0 "
                    "AND text LIKE ? AND text<>?",
                    (slot + "%", f["text"])).fetchall()
                for oid, otext in olds:
                    self._con.execute(
                        "UPDATE user_facts SET superseded=1 WHERE id=?",
                        (oid,))
                    self.last_superseded.append(otext)
                if olds:
                    self._con.commit()
                    self._reload()
                    existing = set(self._texts)
            try:
                mid = self.add(f["text"], source_msg=user_msg)
                added.append(mid)
                existing.add(f["text"])
            except Exception:
                pass
        return added

    def query(self, text: str, k: int = 5) -> list[dict]:
        """Look up facts relevant to the query, in the USER_SELF
        subspace."""
        if self._stack is None or len(self._ids) == 0 or self.encoder is None:
            return []
        q_hv = bind_user(self.encoder.encode(text))
        xor = np.bitwise_xor(self._stack, q_hv[None, :])
        dists = xor.sum(axis=1)
        order = np.argsort(dists)[:k]
        out = []
        for rank, idx in enumerate(order):
            d = int(dists[idx])
            sim = 1.0 - 2.0 * d / D
            out.append({
                "id":         self._ids[idx],
                "text":       self._texts[idx],
                "distance":   d,
                "similarity": round(sim, 4),
                "rank":       rank + 1,
            })
        return out

    def all_facts(self) -> list[str]:
        """Return all stored user-fact texts (newest last)."""
        return list(self._texts)


def _self_test():
    msgs = [
        "my name is Eric and I'm a developer",
        "I live in Seattle and I have two cats",
        "I like trading futures",
        "remember that I prefer markdown responses",
        "what's the weather like",   # no facts
    ]
    for m in msgs:
        facts = extract_user_facts(m)
        print(f"  {m!r}")
        for f in facts:
            print(f"      [{f['kind']:<11}] {f['text']}")


if __name__ == "__main__":
    _self_test()
