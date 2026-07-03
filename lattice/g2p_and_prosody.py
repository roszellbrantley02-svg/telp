"""
lattice/g2p_and_prosody.py - text -> phonemes + prosody features.

WHY
---
Telp can't read text directly; he reads sounds.  This module
converts text -> ARPABET phoneme sequence using the CMU Pronouncing
Dictionary (~135K English words, free, public domain).

It also extracts PROSODY features the phoneme atoms alone don't
carry:
  - Stress markers from CMU dict (0=unstressed, 1=primary, 2=secondary)
  - Punctuation -> phrase boundaries
  - Rhythm pattern at the line level (sequence of stress levels)

Children's books are RHYTHM-HEAVY ("Brown bear, brown bear" /
"Twinkle, twinkle, little star" / "Itsy bitsy spider").  A toddler
learns the cadence BEFORE the meaning.  Telp should too.

OUTPUTS
-------
  word_to_phonemes("brown") -> [("B", 0), ("R", 0), ("AW", 1), ("N", 0)]
                                   ↑ phoneme    ↑ stress level

  text_to_units("Brown bear, brown bear.") -> [
    {"word": "brown", "phonemes": [...], "stress_pattern": (1,)},
    {"word": "bear",  ...},
    {"phrase_break": ","},
    {"word": "brown", ...},
    {"word": "bear",  ...},
    {"phrase_break": "."},
  ]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# CMU dict via nltk (downloaded at install)
_CMU = None


def _cmu():
    global _CMU
    if _CMU is None:
        import nltk
        from nltk.corpus import cmudict
        try:
            _CMU = cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            _CMU = cmudict.dict()
    return _CMU


# --- G2P (text -> phoneme sequence) -------------------------------


def split_phoneme_stress(arpabet_token: str) -> tuple[str, int]:
    """`AY1` -> ("AY", 1).  `B` -> ("B", 0)."""
    if arpabet_token and arpabet_token[-1] in "012":
        return arpabet_token[:-1], int(arpabet_token[-1])
    return arpabet_token, 0


def word_to_phonemes(word: str) -> list[tuple[str, int]]:
    """Return phoneme + stress sequence for a word.

    Tries CMU dict first.  Falls back to a tiny char-level rule set
    for OOV (very rough but better than nothing for nursery rhyme
    targets).

    NOTE: returns ONLY THE FIRST pronunciation variant.  Use
    word_to_phonemes_all() to get every CMU variant for the same word.
    """
    w = word.lower().strip()
    if not w:
        return []
    d = _cmu()
    if w in d:
        # Take first pronunciation variant
        arp = d[w][0]
        return [split_phoneme_stress(t) for t in arp]
    return _fallback_g2p(w)


def word_to_phonemes_all(word: str) -> list[list[tuple[str, int]]]:
    """Return ALL CMU pronunciation variants for a word.

    CMU dict lists multiple pronunciations for ~10% of English words
    that have regional or stress-pattern variants:
      tomato:  [[T,AH,M,EY,T,OW], [T,AH,M,AA,T,OW]]
      either:  [[IY,DH,ER],       [AY,DH,ER]]

    Returns a list of phoneme sequences, one per variant.  Single-
    variant words return a 1-element list.  OOV words return the
    fallback as a 1-element list.
    """
    w = word.lower().strip()
    if not w:
        return []
    d = _cmu()
    if w in d:
        return [
            [split_phoneme_stress(t) for t in variant]
            for variant in d[w]
        ]
    return [_fallback_g2p(w)]


def _fallback_g2p(word: str) -> list[tuple[str, int]]:
    """Very rough char-level fallback for words not in CMU dict.

    This is intentionally minimal — for v0.1 we don't need to handle
    English's full orthographic chaos.  Just make sure unknown words
    don't crash and produce *something* reasonable.
    """
    # Mostly-correct single-character mappings (no real spelling rules)
    char_to_phoneme = {
        "a": "AE", "b": "B",  "c": "K",  "d": "D",  "e": "EH",
        "f": "F",  "g": "G",  "h": "HH", "i": "IH", "j": "JH",
        "k": "K",  "l": "L",  "m": "M",  "n": "N",  "o": "OW",
        "p": "P",  "q": "K",  "r": "R",  "s": "S",  "t": "T",
        "u": "AH", "v": "V",  "w": "W",  "x": "K",  "y": "Y",
        "z": "Z",
    }
    out: list[tuple[str, int]] = []
    for ch in word.lower():
        if ch in char_to_phoneme:
            out.append((char_to_phoneme[ch], 0))
    return out


# --- Prosody / phrase segmentation --------------------------------


# Punctuation that marks phrase boundaries
_PHRASE_BREAKS = {",": "comma",
                  ";": "semicolon",
                  ":": "colon",
                  ".": "period",
                  "!": "exclamation",
                  "?": "question",
                  "—": "emdash"}


_WORD_RE = re.compile(r"[A-Za-z']+")


def text_to_units(text: str) -> list[dict]:
    """Tokenize text into an ordered list of "units" mixing words +
    phrase breaks.  Words include their phoneme sequence and
    stress-only pattern.

    Returns list of dicts shaped like:
      {"word": str, "phonemes": [(p, stress), ...],
       "stress_pattern": tuple[int, ...]}
      or
      {"phrase_break": str, "char": str}
    """
    units: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            # Blank line = stanza break
            units.append({"phrase_break": "stanza", "char": "\n"})
            continue
        # Walk the line char by char, peeling words and breaks
        i = 0
        while i < len(line):
            ch = line[i]
            if ch.isspace():
                i += 1
                continue
            if ch in _PHRASE_BREAKS:
                units.append({"phrase_break": _PHRASE_BREAKS[ch],
                                  "char": ch})
                i += 1
                continue
            m = _WORD_RE.match(line, i)
            if m:
                w = m.group(0)
                pseq = word_to_phonemes(w)
                stress_only = tuple(s for _, s in pseq
                                            if any(s > 0 for _, s in pseq))
                if not stress_only:
                    stress_only = tuple(s for _, s in pseq)
                units.append({
                    "word": w.lower(),
                    "phonemes": pseq,
                    "stress_pattern": stress_only,
                })
                i = m.end()
                continue
            # Unknown char (digit, weird symbol) — skip
            i += 1
        # End of line = line break (lighter than phrase break)
        units.append({"phrase_break": "line", "char": "\\n"})
    return units


def stress_pattern_of_line(units: list[dict]) -> tuple[int, ...]:
    """Concatenate the stress patterns of all words in a line.
    Used to bind line-level rhythm.
    """
    out = []
    for u in units:
        if "stress_pattern" in u:
            out.extend(u["stress_pattern"])
        elif u.get("phrase_break") in ("line", "stanza"):
            break
    return tuple(out)


# --- CLI smoke test -----------------------------------------------


if __name__ == "__main__":
    test_lines = [
        "Brown bear, brown bear, what do you see?",
        "Twinkle, twinkle, little star.",
        "How I wonder what you are!",
        "Mary had a little lamb",
        "whose fleece was white as snow.",
    ]
    print("=== G2P + prosody smoke test ===\n")
    for line in test_lines:
        units = text_to_units(line)
        print(f"INPUT : {line!r}")
        for u in units:
            if "word" in u:
                phon = " ".join(f"{p}{s}" for p, s in u["phonemes"])
                stress = "".join(str(s) for s in u["stress_pattern"])
                print(f"  word   {u['word']:10}  phon=[{phon}]  stress={stress}")
            else:
                print(f"  break  {u['phrase_break']:10}  ({u['char']!r})")
        sp = stress_pattern_of_line(units)
        print(f"  LINE STRESS PATTERN: {sp}\n")
