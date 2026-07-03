"""
lattice/phoneme_hdc.py - Phase 12: developmental language atoms.

WHY
---
Instead of importing a pre-trained tokenizer that mimics LLM
vocabulary, build language from the SAME atoms a human infant
starts with: phonemes.  Every baby on Earth babbles `ba, ma, da`
before they learn ANY words, in ANY language.

ARPABET (the standard American-English phoneme inventory used by
CMU dict) has 39 phonemes.  Each gets:
  1. A deterministic unique HV (SHA-1 seeded from the phoneme name)
  2. PLUS feature bindings — type/place/manner/voicing for consonants,
     height/backness/rounding for vowels

The HV for each phoneme is the BUNDLE of these.  Result:
  - `B` and `P` share the same place (bilabial) and manner (stop) but
    differ in voicing -> their HVs are similar but not identical
  - `B` and `M` share place + voicing but differ in manner (stop vs
    nasal) -> moderately similar
  - `B` and `IY` share almost no features -> nearly orthogonal

This means a baby's first sounds CLUSTER PHONETICALLY in HV space
the same way they cluster acoustically in the ear.

COMPOSITION
-----------
A word HV = bundle( bind(POS_i, phoneme_hv_i) for i in 0..N )
A sentence HV = bundle( bind(POS_j, word_hv_j) for j in 0..M )

Same bind/bundle/permute algebra that runs the trading reasoner.
This is the AGI thesis: ONE substrate, applied at every level.

USAGE
-----
    from lattice.phoneme_hdc import phoneme_hv, compose_word, D
    hv_b = phoneme_hv("B")
    hv_p = phoneme_hv("P")
    sim = 1 - hamming_distance(hv_b, hv_p) / D
    # ~0.65-0.75: B and P share most features

    bear_hv = compose_word(["B", "EH1", "R"])
    wear_hv = compose_word(["W", "EH1", "R"])
    # Similar because they share EH1 and R
"""
from __future__ import annotations
import hashlib
from typing import Optional

import numpy as np

from train.v5_hdc_prototype import D, bind, bundle, hamming_distance


# --- ARPABET phoneme inventory + acoustic features ----------------
#
# Format: phoneme -> dict of features.  Stress digits (0/1/2) on vowels
# are normalized away before lookup (B + EH1 + R becomes B + EH + R for
# phoneme HV purposes; stress can be added as an extra binding later).

PHONEMES = {
    # ── Vowels (monophthongs) ────────────────────────────────────
    "AA": {"class": "vowel", "height": "low",  "backness": "back",    "round": "no"},   # father
    "AE": {"class": "vowel", "height": "low",  "backness": "front",   "round": "no"},   # cat
    "AH": {"class": "vowel", "height": "mid",  "backness": "central", "round": "no"},   # but
    "AO": {"class": "vowel", "height": "mid",  "backness": "back",    "round": "yes"},  # caught
    "EH": {"class": "vowel", "height": "mid",  "backness": "front",   "round": "no"},   # bet
    "ER": {"class": "vowel", "height": "mid",  "backness": "central", "round": "no", "rhotic": "yes"},  # bird
    "IH": {"class": "vowel", "height": "high", "backness": "front",   "round": "no"},   # bit
    "IY": {"class": "vowel", "height": "high", "backness": "front",   "round": "no"},   # beat
    "UH": {"class": "vowel", "height": "high", "backness": "back",    "round": "yes"},  # book
    "UW": {"class": "vowel", "height": "high", "backness": "back",    "round": "yes"},  # boot

    # ── Diphthongs ───────────────────────────────────────────────
    "AW": {"class": "diphthong", "start": "AA", "end": "UH"},  # cow
    "AY": {"class": "diphthong", "start": "AA", "end": "IH"},  # eye
    "EY": {"class": "diphthong", "start": "EH", "end": "IH"},  # bait
    "OW": {"class": "diphthong", "start": "AO", "end": "UH"},  # boat
    "OY": {"class": "diphthong", "start": "AO", "end": "IH"},  # boy

    # ── Stops ────────────────────────────────────────────────────
    "B": {"class": "stop", "place": "bilabial",  "voiced": "yes"},
    "P": {"class": "stop", "place": "bilabial",  "voiced": "no"},
    "D": {"class": "stop", "place": "alveolar",  "voiced": "yes"},
    "T": {"class": "stop", "place": "alveolar",  "voiced": "no"},
    "G": {"class": "stop", "place": "velar",     "voiced": "yes"},
    "K": {"class": "stop", "place": "velar",     "voiced": "no"},

    # ── Fricatives ───────────────────────────────────────────────
    "V":  {"class": "fricative", "place": "labiodental", "voiced": "yes"},
    "F":  {"class": "fricative", "place": "labiodental", "voiced": "no"},
    "DH": {"class": "fricative", "place": "dental",      "voiced": "yes"},   # this
    "TH": {"class": "fricative", "place": "dental",      "voiced": "no"},    # thin
    "Z":  {"class": "fricative", "place": "alveolar",    "voiced": "yes"},
    "S":  {"class": "fricative", "place": "alveolar",    "voiced": "no"},
    "ZH": {"class": "fricative", "place": "palatal",     "voiced": "yes"},   # measure
    "SH": {"class": "fricative", "place": "palatal",     "voiced": "no"},    # ship
    "HH": {"class": "fricative", "place": "glottal",     "voiced": "no"},    # hat

    # ── Affricates ───────────────────────────────────────────────
    "JH": {"class": "affricate", "place": "palatal", "voiced": "yes"},  # judge
    "CH": {"class": "affricate", "place": "palatal", "voiced": "no"},   # church

    # ── Nasals ───────────────────────────────────────────────────
    "M":  {"class": "nasal", "place": "bilabial", "voiced": "yes"},
    "N":  {"class": "nasal", "place": "alveolar", "voiced": "yes"},
    "NG": {"class": "nasal", "place": "velar",    "voiced": "yes"},     # sing

    # ── Liquids ──────────────────────────────────────────────────
    "L":  {"class": "liquid", "place": "alveolar", "voiced": "yes"},
    "R":  {"class": "liquid", "place": "alveolar", "voiced": "yes", "rhotic": "yes"},

    # ── Glides ───────────────────────────────────────────────────
    "W":  {"class": "glide", "place": "bilabial", "voiced": "yes"},
    "Y":  {"class": "glide", "place": "palatal",  "voiced": "yes"},
}


# --- Deterministic HV machinery ----------------------------------


def _det_hv(seed_str: str) -> np.ndarray:
    """SHA-1 seeded deterministic HV.  Same string -> same HV forever."""
    h = hashlib.sha1(f"phoneme_hdc_v1::{seed_str}".encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=D, dtype=np.int8)


# Cache per-phoneme HVs after first compute
_PHONEME_HV_CACHE: dict[str, np.ndarray] = {}
_FEATURE_HV_CACHE: dict[str, np.ndarray] = {}


def _feature_hv(role: str, value: str) -> np.ndarray:
    """HV for a (role, value) acoustic feature binding."""
    key = f"feat::{role}={value}"
    if key not in _FEATURE_HV_CACHE:
        role_hv  = _det_hv(f"role::{role}")
        value_hv = _det_hv(f"value::{value}")
        _FEATURE_HV_CACHE[key] = bind(role_hv, value_hv)
    return _FEATURE_HV_CACHE[key]


def _normalize_phoneme(p: str) -> str:
    """Strip CMU stress digits (B + EH1 + R -> B + EH + R).

    Stress is a separate dimension we can bind later; for the base
    phoneme HV it's noise.
    """
    if p and p[-1] in "012":
        return p[:-1]
    return p


def phoneme_hv(phoneme: str) -> np.ndarray:
    """Get the HV for a phoneme.

    The HV is the bundle of:
      1. A per-phoneme random unique HV (gives identity)
      2. Bound feature HVs (gives acoustic similarity)

    Diphthongs are bundled from their start/end phoneme HVs.
    Unknown phonemes get a single random HV (no features).
    """
    p = _normalize_phoneme(phoneme)
    if p in _PHONEME_HV_CACHE:
        return _PHONEME_HV_CACHE[p]

    if p not in PHONEMES:
        # Unknown / weird phoneme: just random uniqueness
        hv = _det_hv(f"phoneme_unknown::{p}")
        _PHONEME_HV_CACHE[p] = hv
        return hv

    spec = PHONEMES[p]
    components = []
    # Per-phoneme uniqueness so two phonemes with IDENTICAL features
    # are still distinguishable (e.g., AA vs different vowels that
    # collide on the bucketed feature set)
    components.append(_det_hv(f"phoneme_id::{p}"))

    # Diphthong: bundle in the start + end phoneme HVs (recursive)
    if spec.get("class") == "diphthong":
        components.append(phoneme_hv(spec["start"]))
        components.append(phoneme_hv(spec["end"]))
        components.append(_feature_hv("class", "diphthong"))
    else:
        for role, value in spec.items():
            components.append(_feature_hv(role, value))

    hv = bundle(components)
    _PHONEME_HV_CACHE[p] = hv
    return hv


# --- Position HVs (cyclic permutation of a base axis) -------------


_POS_AXIS = _det_hv("position_axis_v1")


def position_hv(i: int) -> np.ndarray:
    """Cyclic permutation of a base axis HV.  Same axis, different shift
    per position.  Cheap, deterministic, gives unique positional bindings
    without needing a per-position lookup table.
    """
    return np.roll(_POS_AXIS, i)


# --- Compose phoneme sequence into word HV -------------------------


def compose_word(phoneme_seq: list[str]) -> np.ndarray:
    """Word HV = bundle of bind(POS_i, phoneme_hv_i) for i in 0..N.

    Two words sharing many of the same phonemes at the same positions
    will have similar HVs.  Two words sharing the same phonemes in
    different positions will be less similar (position changes the
    binding).
    """
    if not phoneme_seq:
        return np.zeros(D, dtype=np.int8)
    pairs = []
    for i, p in enumerate(phoneme_seq):
        pairs.append(bind(position_hv(i), phoneme_hv(p)))
    return bundle(pairs)


def compose_sentence(word_hvs: list[np.ndarray]) -> np.ndarray:
    """Sentence HV = bundle of bind(POS_j, word_hv_j) for j in 0..M.

    Same algebra one level up.
    """
    if not word_hvs:
        return np.zeros(D, dtype=np.int8)
    pairs = []
    for j, w in enumerate(word_hvs):
        pairs.append(bind(position_hv(j), w))
    return bundle(pairs)


# --- Similarity -----------------------------------------------------


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 = identical, ~0.5 = orthogonal/random, 0.0 = perfectly anti."""
    return 1.0 - (hamming_distance(a, b) / D)


# --- CLI smoke test -------------------------------------------------


if __name__ == "__main__":
    print(f"=== Phoneme atom table ({len(PHONEMES)} phonemes) ===\n")

    print("Phoneme HV similarities (should reflect acoustic kinship):")
    pairs = [
        ("B", "P", "voiced/unvoiced bilabial stop pair"),
        ("B", "M", "bilabial voiced (stop vs nasal)"),
        ("B", "D", "voiced stops (bilabial vs alveolar)"),
        ("B", "T", "stops (bilabial voiced vs alveolar unvoiced)"),
        ("B", "IY", "completely different (consonant vs vowel)"),
        ("S", "Z", "voiced/unvoiced alveolar fricative pair"),
        ("S", "SH", "fricatives (alveolar vs palatal)"),
        ("IY", "IH", "high front vowels"),
        ("IY", "UW", "high vowels (front vs back)"),
        ("IY", "AA", "completely different vowels"),
        ("AY", "AW", "diphthongs sharing AA start"),
    ]
    for p1, p2, note in pairs:
        h1 = phoneme_hv(p1)
        h2 = phoneme_hv(p2)
        s = similarity(h1, h2)
        bar = "=" * int(round((s - 0.4) * 50)) if s > 0.4 else ""
        print(f"  {p1:3} vs {p2:3}  sim={s:.3f}  {bar}  {note}")

    print("\n=== Word composition ===")
    word_pairs = [
        ([], "bear",  ["B", "EH1", "R"],
              "wear",  ["W", "EH1", "R"],
              "share EH1+R"),
        ([], "bear",  ["B", "EH1", "R"],
              "bare",  ["B", "EH1", "R"],
              "identical phonemes"),
        ([], "bear",  ["B", "EH1", "R"],
              "beard", ["B", "IH1", "R", "D"],
              "share B+R but different vowel"),
        ([], "cat",   ["K", "AE1", "T"],
              "bat",   ["B", "AE1", "T"],
              "share AE1+T, voiceless/voiced onset"),
        ([], "cat",   ["K", "AE1", "T"],
              "dog",   ["D", "AO1", "G"],
              "completely different"),
    ]
    for _, n1, p1, n2, p2, note in word_pairs:
        w1 = compose_word(p1)
        w2 = compose_word(p2)
        s = similarity(w1, w2)
        bar = "=" * int(round((s - 0.4) * 50)) if s > 0.4 else ""
        print(f"  {n1:7} vs {n2:7}  sim={s:.3f}  {bar}  {note}")
