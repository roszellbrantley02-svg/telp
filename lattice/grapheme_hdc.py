"""
lattice/grapheme_hdc.py - Phase 12.3: spelling atoms.

WHY
---
Up to now Telp's word identity has been PHONEME-ONLY (sound).  That
means he literally cannot tell `bear` from `bare` — same phonemes
[B EH R] -> same HV -> homophones collapse.  He also has no way to
detect that `running` contains `run` or that `rainbow` contains
`rain`.

This module adds the SPELLING side:

  26 grapheme atoms (a..z), each with:
    - Deterministic SHA-1 HV (uniqueness)
    - Bound feature HVs (vowel/consonant, common letter classes)

  Word-spelling composition:
    spelling_hv("bear") = bundle( bind(POS_i, grapheme_hv(c_i))
                                       for i, c_i in enumerate("bear") )

Combined with the existing phoneme composition, a word's IDENTITY
HV can now bundle BOTH sound and spelling:
    word_identity_hv("bear") = bundle( phoneme_hv_of("bear"),
                                        spelling_hv("bear") )

Now `bear` and `bare` have:
  - identical phoneme component
  - DIFFERENT spelling component
  - -> different word identity HVs -> homophones distinguishable

Same algebra; new atom layer.
"""
from __future__ import annotations
import hashlib

import numpy as np

from train.v5_hdc_prototype import D, bind, bundle, hamming_distance
from lattice.phoneme_hdc import _det_hv, position_hv, similarity


# --- Grapheme feature taxonomy ------------------------------------

VOWELS    = set("aeiou")
# y is a "sometimes vowel" — handled separately
SEMI_VOWEL = set("y")
CONSONANTS = set("bcdfghjklmnpqrstvwxz")

# Letter shape / class features.  These give similar-letter pairs
# (b/d, p/q mirror images; o/c rounded; m/n similar nasals) some
# additional HV similarity beyond random uniqueness.
LETTER_SHAPES = {
    "rounded":   set("ocaegqdb"),
    "ascender":  set("bdfhklt"),
    "descender": set("gjpqy"),
    "tall":      set("bdfhiklt"),
    "short":     set("acemnorsuvwxz"),
}


# --- Grapheme atom HVs --------------------------------------------


_GRAPHEME_HV_CACHE: dict[str, np.ndarray] = {}


def _grapheme_feature_hv(role: str, value: str) -> np.ndarray:
    role_hv  = _det_hv(f"grapheme_role::{role}")
    value_hv = _det_hv(f"grapheme_value::{value}")
    return bind(role_hv, value_hv)


def grapheme_hv(letter: str) -> np.ndarray:
    """HV for a single lowercase letter.  Composed from a unique
    per-letter seed PLUS feature bindings (vowel/consonant class,
    letter-shape features).

    Letters in the same class (e.g. both vowels) share part of their
    HV via the class-bind component; same algebra as phoneme features.
    """
    ch = letter.lower()
    if ch in _GRAPHEME_HV_CACHE:
        return _GRAPHEME_HV_CACHE[ch]

    components = []
    # 1. Per-letter uniqueness
    components.append(_det_hv(f"grapheme_id::{ch}"))
    # 2. Vowel/consonant class
    if ch in VOWELS:
        components.append(_grapheme_feature_hv("class", "vowel"))
    elif ch in SEMI_VOWEL:
        components.append(_grapheme_feature_hv("class", "semivowel"))
    elif ch in CONSONANTS:
        components.append(_grapheme_feature_hv("class", "consonant"))
    else:
        components.append(_grapheme_feature_hv("class", "other"))
    # 3. Shape features
    for shape, members in LETTER_SHAPES.items():
        if ch in members:
            components.append(_grapheme_feature_hv("shape", shape))

    hv = bundle(components)
    _GRAPHEME_HV_CACHE[ch] = hv
    return hv


# --- Spelling composition -----------------------------------------


def spelling_hv(word: str) -> np.ndarray:
    """Compose a word's SPELLING HV from its letters.

    Same position-binding scheme as phoneme composition (cyclic
    permutation rho^i applied to position axis).  Two words sharing
    a common LETTER prefix will share HV structure at those positions.
    """
    w = word.lower()
    if not w:
        return np.zeros(D, dtype=np.int8)
    pairs = []
    for i, ch in enumerate(w):
        if ch.isalpha():
            pairs.append(bind(position_hv(i), grapheme_hv(ch)))
    if not pairs:
        return np.zeros(D, dtype=np.int8)
    return bundle(pairs)


def spelling_hv_unordered(word: str) -> np.ndarray:
    """Spelling HV that's INSENSITIVE to letter order.

    Useful for "rough match" — two anagrams (`silent` / `listen`)
    would have identical unordered HVs.  Bag-of-letters.
    """
    w = word.lower()
    if not w:
        return np.zeros(D, dtype=np.int8)
    pairs = [grapheme_hv(ch) for ch in w if ch.isalpha()]
    if not pairs:
        return np.zeros(D, dtype=np.int8)
    return bundle(pairs)


# --- Sub-word / inside-word detection -----------------------------


def contains_word(big: str, small: str) -> bool:
    """True if `small` appears as a contiguous letter substring of `big`."""
    return small.lower() in big.lower()


def find_sub_words(word: str,
                          vocab: set[str],
                          min_len: int = 3) -> list[str]:
    """Find every word in `vocab` that appears as a contiguous substring
    of `word`.  e.g. find_sub_words("rainbow", {"rain", "bow", "in"})
    returns ["rain", "bow", "in"] (filtered to min_len).
    """
    w = word.lower()
    out = []
    for v in vocab:
        if len(v) < min_len:
            continue
        if v == w:
            continue
        if v in w:
            out.append(v)
    # Sort by length desc so longer matches come first
    out.sort(key=lambda s: -len(s))
    return out


# --- Smoke test ----------------------------------------------------


if __name__ == "__main__":
    print(f"=== Grapheme atom table (26 letters) ===\n")
    # Show feature-based similarities
    pairs = [
        ("a", "e", "both vowels"),
        ("a", "i", "both vowels"),
        ("a", "o", "vowel + rounded"),
        ("b", "d", "rounded ascenders"),
        ("p", "q", "rounded descenders"),
        ("b", "p", "rounded, b ascender, p descender"),
        ("m", "n", "both consonants, both short"),
        ("a", "b", "vowel vs consonant"),
        ("a", "z", "completely different vowel vs consonant"),
        ("o", "c", "both rounded short"),
    ]
    for l1, l2, note in pairs:
        s = similarity(grapheme_hv(l1), grapheme_hv(l2))
        bar = "=" * int(round((s - 0.4) * 50)) if s > 0.4 else ""
        print(f"  '{l1}' vs '{l2}'  sim={s:.3f}  {bar}  {note}")

    print(f"\n=== Spelling similarity ===\n")
    word_pairs = [
        ("bear",  "bare",   "anagram-but-ordered: same letters, different order"),
        ("bear",  "bears",  "share 'bear' prefix"),
        ("bear",  "wear",   "share 'ear' suffix, b->w"),
        ("bear",  "wolf",   "completely different spelling"),
        ("running", "runs",  "share 'run' prefix"),
        ("running", "runner","share 'run' prefix"),
        ("running", "jumping","both end in 'ing'"),
        ("there", "their",  "share 'the' + 'r'"),
        ("there", "they",   "share 'the' prefix"),
        ("to",    "two",    "share 't' onset"),
        ("to",    "too",    "share 'to' prefix"),
    ]
    for w1, w2, note in word_pairs:
        s = similarity(spelling_hv(w1), spelling_hv(w2))
        bar = "=" * int(round((s - 0.4) * 50)) if s > 0.4 else ""
        print(f"  '{w1:10}' vs '{w2:10}'  sim={s:.3f}  {bar}  {note}")

    print(f"\n=== Unordered (anagram) spelling ===\n")
    for w1, w2 in [("bear", "bare"), ("listen", "silent"),
                       ("evil", "live"), ("cat", "dog")]:
        s = similarity(spelling_hv_unordered(w1),
                              spelling_hv_unordered(w2))
        print(f"  '{w1}' vs '{w2}'  unordered_sim={s:.3f}  "
              f"(ordered={similarity(spelling_hv(w1), spelling_hv(w2)):.3f})")

    print(f"\n=== Sub-word detection ===\n")
    vocab = {"rain", "bow", "in", "ow", "sun", "flower", "low",
                "break", "fast", "the", "fox", "snow", "the", "boat"}
    for big in ["rainbow", "sunflower", "breakfast", "snowboat"]:
        subs = find_sub_words(big, vocab)
        print(f"  '{big}' contains: {subs}")
