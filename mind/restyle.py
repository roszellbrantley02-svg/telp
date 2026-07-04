"""
mind/restyle.py - truth-preserving paraphrase: HIS voice, provably.

Encyclopedia prose becomes plainer speech through transforms that are
each individually justified, then verified as a whole:

  1. SYNONYM SIMPLIFICATION - a word may be swapped only if the offline
     Wiktionary (650K synonym pairs) lists the replacement as a synonym,
     the replacement is shorter and plainer, and it is not tagged
     archaic/obsolete/dated. The dictionary licenses the swap.
  2. SENTENCE SPLITTING - semicolon chains and long ", and" compounds
     become two sentences. No words change.
  3. THE ROUND-TRIP CHECK - the restyled sentence's embedding must stay
     >= 0.92 cosine to the original, or every change reverts. The neural
     net PROOFREADS meaning-preservation; it never authors a word.

Every accepted rewrite is enumerable: original -> [transforms] -> result.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
_DICT_DB = _TELP_ROOT / "state" / "wiktionary" / "dict.db"

_BAD_TAGS = ("archaic", "obsolete", "dated", "rare", "poetic", "dialect",
             "slang", "vulgar", "informal", "humorous")

_con = None


def _dict_con():
    global _con
    if _con is None and _DICT_DB.exists():
        _con = sqlite3.connect(str(_DICT_DB), check_same_thread=False)
    return _con


# before a number, only a numeric approximator reads right:
# "approximately 65 days" -> "about 65 days", never "loosely 65 days"
_NUMERIC_APPROX = frozenset(("about", "around", "nearly", "roughly",
                             "almost"))


@lru_cache(maxsize=4096)
def simpler_synonym(word: str, before_number: bool = False) -> str | None:
    """A shorter, plainer, dictionary-licensed synonym for `word`,
    or None. Deterministic: candidates ordered by (length, alpha)."""
    con = _dict_con()
    if con is None or len(word) < 8:
        return None
    rows = con.execute(
        "SELECT DISTINCT target FROM related WHERE word=? AND "
        "relation='synonym'", (word,)).fetchall()
    cands = sorted({r[0] for r in rows
                    if r[0].isalpha() and r[0].islower()
                    and 3 <= len(r[0]) <= len(word) - 3},
                   key=lambda w: (len(w), w))
    if before_number:
        cands = [c for c in cands if c in _NUMERIC_APPROX]
    if not cands:
        return None
    # the replacement must be usable in the same POSITION: same part of
    # speech ("approximately"/adv -> "odd"/adj gave "of odd 65 days")
    src_pos = {r[0] for r in con.execute(
        "SELECT DISTINCT pos FROM entries WHERE word=?", (word,))}
    for c in cands:
        ent = con.execute(
            "SELECT DISTINCT pos, COALESCE(tags,'') FROM entries "
            "WHERE word=? LIMIT 8", (c,)).fetchall()
        if not ent:
            continue                        # unknown word: not licensed
        c_pos = {p for p, _ in ent}
        if src_pos and not (src_pos & c_pos):
            continue
        joined = " ".join(t for _, t in ent).lower()
        if any(b in joined for b in _BAD_TAGS):
            continue
        return c
    return None


_WORD_RE = re.compile(r"\b([a-z]{8,})\b")

# v1 swap scope: -ly ADVERBS (frame-free modifiers - substitution cannot
# break syntax) plus a curated safe set. Verbs and nouns carry
# subcategorization frames the round-trip check cannot see ("consider X
# to be" -> "see X to be" passed cosine and broke the grammar).
_SAFE_SWAP = frozenset(("numerous", "commence", "utilize"))


def _apply_swaps(sent: str, max_swaps: int = 2) -> tuple[str, list]:
    swaps = []
    out = sent
    for m in _WORD_RE.finditer(sent):
        if len(swaps) >= max_swaps:
            break
        w = m.group(1)
        if not (w.endswith("ly") or w in _SAFE_SWAP):
            continue
        before_num = bool(re.match(r"\s+\d", sent[m.end():]))
        s = simpler_synonym(w, before_num)
        if s is None or s == w:
            continue
        out = re.sub(rf"\b{w}\b", s, out, count=1)
        swaps.append((w, s))
    return out, swaps


def _apply_splits(sent: str) -> tuple[str, int]:
    n = 0
    # semicolon chains: always safe to split
    if "; " in sent and len(sent) > 90:
        parts = [p.strip() for p in sent.split("; ") if p.strip()]
        parts = [p[0].upper() + p[1:] if p else p for p in parts]
        parts = [p if p.endswith((".", "!", "?")) else p + "." for p in parts]
        sent = " ".join(parts)
        n += len(parts) - 1
    # one long ", and <clause with its own verb>" becomes a new sentence
    if len(sent) > 140:
        m = re.search(r", and (?=(?:the|a|an|it|its|they|their|he|she|his"
                      r"|her|studies|some|this|these)\b)", sent)
        if m:
            head = sent[:m.start()].rstrip(",")
            tail = sent[m.end():].strip()
            if len(head) > 40 and len(tail) > 30:
                if not head.endswith((".", "!", "?")):
                    head += "."
                sent = head + " " + tail[0].upper() + tail[1:]
                n += 1
    return sent, n


def restyle(sent: str, emb_fn, min_cos: float = 0.92) -> tuple[str, int]:
    """Restyle one sentence. Returns (text, n_transforms). On round-trip
    failure every transform reverts and the original stands."""
    try:
        cand, swaps = _apply_swaps(sent)
        cand, n_split = _apply_splits(cand)
        n = len(swaps) + n_split
        if n == 0 or cand == sent:
            return sent, 0
        import numpy as np
        e = np.asarray(emb_fn([sent, cand]))
        if float(e[0] @ e[1]) < min_cos:
            return sent, 0                  # meaning drifted: revert
        return cand, n
    except Exception:
        return sent, 0
