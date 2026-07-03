"""
lattice/justification.py — Phase 17b: explain unusual word choices.

WHY
---
When the imagination engine picks a filler whose novelty is high
("the dragon walked to the headroom"), we don't filter it out
(that suppresses creativity).  Instead, we ATTACH a justification
clause that grounds the unusual choice in the dictionary entry,
so the surreal reads as intentional surrealism, not as a mistake.

Same algebraic frame, augmented surface form:

  bad:    "The dragon walked to the headroom."
  good:   "The dragon walked to the headroom — the space above
            one's head."
  good:   "The dragon saw a flicker — a kind of woodpecker."

The mechanism is pure dictionary lookup.  Telp earns his unusual
choices by pulling their definition into the narrative.  This is
genuinely how surreal fiction works (Carroll, Calvino, kids on the
playground) — weird words carry their definition.

DESIGN
------
Justifier reads from dict.db.  For a word, it composes an
appositive clause from the primary noun-sense gloss, trimmed to a
short readable noun phrase.
"""
from __future__ import annotations

import re
from typing import Optional

from lattice.dictionary_lookup import Dictionary, Entry


# Words that begin a gloss but should be stripped (they say "a..."
# or "the..." which is fine, but "any of various..." reads worse
# than just "various...")
_GLOSS_PREFIXES_TO_DROP = (
    "any of various ", "any of the various ", "any of a class of ",
    "any of a number of ", "any of several ", "any of many ",
    "any of ", "one of ",
)

# Gloss ends — trim at the first occurrence of any of these.
_GLOSS_CUT_PATTERNS = [
    re.compile(r";"),                              # clause separator
    re.compile(r"\.\s"),                           # sentence end
    re.compile(r"\s\([^)]{30,}\)"),                # long parenthetical
    re.compile(r",\s+especially\b", re.IGNORECASE),
    re.compile(r",\s+typically\b", re.IGNORECASE),
    re.compile(r",\s+often\b", re.IGNORECASE),
    re.compile(r",\s+sometimes\b", re.IGNORECASE),
    re.compile(r",\s+used\b", re.IGNORECASE),
    re.compile(r",\s+having\b", re.IGNORECASE),
    re.compile(r",\s+with\b", re.IGNORECASE),
    re.compile(r",\s+from\b", re.IGNORECASE),
    re.compile(r",\s+including\b", re.IGNORECASE),
    re.compile(r",\s+such as\b", re.IGNORECASE),
    re.compile(r",\s+also called\b", re.IGNORECASE),
    re.compile(r",\s+especially\b", re.IGNORECASE),
]

# Maximum char length for a justification clause — keep stories
# from drowning in dictionary prose.
_MAX_CLAUSE_LEN = 90


class Justifier:
    """Turn a dictionary entry into a short inline appositive clause.

    Usage:
      j = Justifier(dictionary)
      clause = j.justify("headroom")
        -> "a small space above the rafters, where dust drifted"

    By default (narrativize=True), the gloss is rewritten into
    fiction register via lattice.narrativize.Narrativizer: taxonomic
    noise stripped, picturable atmosphere added.  Set narrativize=False
    to get the raw dictionary appositive.

    Returns None if no usable clause can be produced (word not in
    dictionary, or gloss too obscure to clean).
    """

    def __init__(self,
                      dictionary: Optional[Dictionary] = None,
                      narrativize: bool = True,
                      rng=None):
        self.dict = dictionary or Dictionary()
        self.narrativize_glosses = narrativize
        if narrativize:
            from lattice.narrativize import Narrativizer
            self.narrativizer = Narrativizer(rng=rng)
        else:
            self.narrativizer = None

    # ── Public API ─────────────────────────────────────────────

    def justify(self, word: str,
                      prefer_pos: str = "noun") -> Optional[str]:
        """Produce a short inline clause defining the word.

        prefer_pos: which POS to prefer when picking the lead sense.
                      For story fillers this is almost always 'noun'.

        If narrativize=True (default), the gloss is rewritten into
        fiction register via Narrativizer.  Otherwise it's trimmed
        as a dictionary appositive.
        """
        if not word:
            return None
        entry = self.dict.lookup(word)
        if entry is None or not entry.senses:
            return None
        gloss = self._pick_gloss(entry, prefer_pos)
        if not gloss:
            return None
        # Phase 18: rewrite into fiction register if narrativization
        # is enabled.  If the narrativizer produces something good,
        # use it.  If it rejects (gloss too messy to clean cleanly),
        # SKIP the justification entirely rather than fall back to
        # the raw long Wiktionary gloss — better silence than a
        # broken appositive.
        if self.narrativizer is not None:
            narrated = self.narrativizer.narrativize(word, gloss)
            if narrated and len(narrated) >= 10:
                return narrated
            return None
        return self._clean_clause(gloss, word)

    # ── Internals ──────────────────────────────────────────────

    def _pick_gloss(self, entry: Entry,
                          prefer_pos: str = "noun") -> Optional[str]:
        """Pick the best gloss for an appositive use.

        Uses the same sense-ranking logic as dictionary_chat:
        avoid alt-of / cross-reference / archaic / obscure senses
        that happen to come first in kaikki's ordering.  Without
        this, "robot" returns "a system of serfdom used in Central
        Europe" (the original 1920s Czech sense) instead of the
        mechanical-being sense.
        """
        if not entry.senses:
            return None

        bad_tags = {"obsolete", "archaic", "rare", "dialectal",
                          "slang", "vulgar", "figurative", "humorous",
                          "informal", "dated", "alt-of", "alternative",
                          "abbreviation", "initialism", "acronym",
                          "misspelling", "nonstandard", "historical",
                          "etymology"}
        cross_ref_starts = (
            "alternative", "misspelling", "plural of",
            "abbreviation of", "initialism of", "acronym of",
            "obsolete form", "archaic form", "rare form",
            "synonym of", "see ", "(see ",
        )

        def rank(s):
            pos_penalty = 0 if s.pos == prefer_pos else 1
            tag_penalty = sum(1 for t in (s.tags or [])
                                      if str(t).lower() in bad_tags)
            g = (s.gloss or "").lower()
            ref_penalty = 10 if g.startswith(cross_ref_starts) else 0
            # Slight preference for shorter, cleaner glosses
            len_penalty = 1 if len(s.gloss or "") > 200 else 0
            return (pos_penalty, tag_penalty + ref_penalty + len_penalty,
                       s.sense_idx)

        best = sorted(entry.senses, key=rank)[0]
        return best.gloss

    def _clean_clause(self, gloss: str, headword: str) -> Optional[str]:
        """Trim, lowercase, and shorten a gloss into an appositive."""
        if not gloss:
            return None
        g = gloss.strip()

        # Cut at the first sub-clause boundary
        for pat in _GLOSS_CUT_PATTERNS:
            m = pat.search(g)
            if m and m.start() > 5:
                g = g[:m.start()]
                break

        # Strip leading "Any of various..." style prefixes
        gl = g.lower()
        for p in _GLOSS_PREFIXES_TO_DROP:
            if gl.startswith(p):
                g = g[len(p):]
                # Re-add an article since we dropped the original
                if not g.lower().startswith(("a ", "an ", "the ",
                                                          "various ")):
                    g = "various " + g
                break

        # Lowercase first letter (it's an appositive, not a sentence)
        if g and g[0].isupper():
            g = g[0].lower() + g[1:]

        # Strip trailing punctuation
        g = g.rstrip(" .,;:").strip()

        # If the gloss begins by repeating the headword ("a frog is
        # any small tailless amphibian..."), it reads naturally as
        # is; otherwise we leave it alone.

        # Length cap — if too long, prefer a cut at a strong
        # boundary AFTER position 30 (so we don't strip the word's
        # essential definition).  Never cut at " of " — that's
        # inside the definition itself ("a body of standing water"
        # cut at " of " would leave only "a body").
        if len(g) > _MAX_CLAUSE_LEN:
            best_cut = None
            for cutchar in (",", "—", "; ", ". "):
                idx = g.find(cutchar, 30)
                if 30 < idx < _MAX_CLAUSE_LEN:
                    if best_cut is None or idx < best_cut:
                        best_cut = idx
            if best_cut is not None:
                g = g[:best_cut].rstrip(" ,.;:—-")
            else:
                # Last resort: word-boundary trim near the cap
                g = g[:_MAX_CLAUSE_LEN].rsplit(" ", 1)[0] + "..."

        # Reject empty / too-short results
        if len(g) < 6:
            return None

        return g


# ─── CLI smoke ───────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    j = Justifier()
    for w in ["headroom", "flicker", "dragon", "kitchen", "thunderstorm",
                "sauropod", "robot", "puddle", "library", "rainbow",
                "haddock", "beetle", "frontcountry"]:
        clause = j.justify(w)
        print(f"  {w:18s} -> {clause}")
