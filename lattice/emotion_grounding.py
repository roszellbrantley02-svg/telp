"""
lattice/emotion_grounding.py - Phase 12.5: words carry feeling.

WHY
---
WordNet gives Telp the dictionary meaning of "scared" or "happy"
but NOT their EMOTIONAL VALUE.  A toddler learns very early that
some words feel GOOD (mom, cookie, hug) and some feel BAD (spider,
fall, broken).  This is the affect dimension that makes language
EMOTIONAL CONTEXT, not just semantic content.

APPROACH
--------
Use the dimensional emotion model from psychology (Russell 1980):
every emotion lives in a 2D space:
  - VALENCE: pleasant (+) vs unpleasant (-)
  - AROUSAL: high-activation vs low-activation

Each emotion word gets an HV bundled from:
  - bind(R_VALENCE,  valence_level_hv)    (positive / negative / neutral)
  - bind(R_AROUSAL,  arousal_level_hv)    (high / mid / low)
  - bind(R_CATEGORY, emotion_category_hv) (joy / fear / anger / etc.)

Then words that AREN'T emotions but ASSOCIATE with feelings (spider
-> often scared, sun -> often warm/happy, fall -> often pain) can
get an `associated_emotion` binding too — via the existing
reading_lattice.bind_semantic() hook.

Result: Telp's emotion subspace is searchable.  He can answer
"what do you feel?" with words that occupy a similar region of
emotion-HV space.
"""
from __future__ import annotations
import sys
from typing import Optional

import numpy as np

from train.v5_hdc_prototype import D, bind, bundle, hamming_distance
from lattice.phoneme_hdc import _det_hv, similarity


# --- Emotion role HVs --------------------------------------------


R_VALENCE  = _det_hv("emotion_role::valence")
R_AROUSAL  = _det_hv("emotion_role::arousal")
R_CATEGORY = _det_hv("emotion_role::category")
R_TRIGGER  = _det_hv("emotion_role::trigger")
R_RESPONSE = _det_hv("emotion_role::response")


def _level_hv(level: str) -> np.ndarray:
    return _det_hv(f"emotion_level::{level}")


def _category_hv(category: str) -> np.ndarray:
    return _det_hv(f"emotion_category::{category}")


# --- Toddler-grade emotion lexicon -------------------------------
#
# Each entry: word -> {valence, arousal, category}
# Valence: positive | neutral | negative
# Arousal: high | mid | low
# Category: Ekman-style basic emotions + toddler-relevant additions

EMOTION_WORDS = {
    # Positive / High arousal
    "excited":   {"valence": "positive", "arousal": "high", "category": "joy"},
    "happy":     {"valence": "positive", "arousal": "mid",  "category": "joy"},
    "joyful":    {"valence": "positive", "arousal": "high", "category": "joy"},
    "glad":      {"valence": "positive", "arousal": "mid",  "category": "joy"},
    "proud":     {"valence": "positive", "arousal": "mid",  "category": "pride"},
    "loved":     {"valence": "positive", "arousal": "mid",  "category": "affection"},
    "safe":      {"valence": "positive", "arousal": "low",  "category": "comfort"},
    "calm":      {"valence": "positive", "arousal": "low",  "category": "comfort"},
    "sleepy":    {"valence": "neutral",  "arousal": "low",  "category": "tired"},
    "tired":     {"valence": "negative", "arousal": "low",  "category": "tired"},
    "bored":     {"valence": "negative", "arousal": "low",  "category": "bored"},
    "sad":       {"valence": "negative", "arousal": "low",  "category": "sadness"},
    "lonely":    {"valence": "negative", "arousal": "low",  "category": "sadness"},
    "scared":    {"valence": "negative", "arousal": "high", "category": "fear"},
    "afraid":    {"valence": "negative", "arousal": "high", "category": "fear"},
    "worried":   {"valence": "negative", "arousal": "mid",  "category": "fear"},
    "shy":       {"valence": "negative", "arousal": "mid",  "category": "fear"},
    "angry":     {"valence": "negative", "arousal": "high", "category": "anger"},
    "mad":       {"valence": "negative", "arousal": "high", "category": "anger"},
    "frustrated": {"valence": "negative", "arousal": "high", "category": "anger"},
    "surprised": {"valence": "neutral",  "arousal": "high", "category": "surprise"},
    "curious":   {"valence": "positive", "arousal": "mid",  "category": "interest"},
    "confused":  {"valence": "negative", "arousal": "mid",  "category": "confusion"},
    "hungry":    {"valence": "negative", "arousal": "mid",  "category": "need"},
    "thirsty":   {"valence": "negative", "arousal": "mid",  "category": "need"},
    "hurt":      {"valence": "negative", "arousal": "high", "category": "pain"},
}


# --- Build emotion HV --------------------------------------------


def emotion_hv(emotion_word: str) -> Optional[np.ndarray]:
    """Build the HV for an emotion word from valence + arousal + category.

    Returns None if not in EMOTION_WORDS lexicon.
    """
    spec = EMOTION_WORDS.get(emotion_word.lower())
    if spec is None:
        return None
    components = [
        bind(R_VALENCE,  _level_hv(spec["valence"])),
        bind(R_AROUSAL,  _level_hv(spec["arousal"])),
        bind(R_CATEGORY, _category_hv(spec["category"])),
    ]
    return bundle(components)


def emotion_axis_hv(valence: str = None,
                            arousal: str = None,
                            category: str = None) -> np.ndarray:
    """Build an emotion HV from explicit axis values.  Useful for
    query: "find words near positive/high-arousal" -> looks for words
    whose emotion HV is similar to emotion_axis_hv("positive", "high").
    """
    components = []
    if valence:  components.append(bind(R_VALENCE,  _level_hv(valence)))
    if arousal:  components.append(bind(R_AROUSAL,  _level_hv(arousal)))
    if category: components.append(bind(R_CATEGORY, _category_hv(category)))
    if not components:
        return np.zeros(D, dtype=np.int8)
    return bundle(components)


# --- Object/event -> default associated emotion ------------------
# For non-emotion words that have a strong default affective load
# (e.g. "spider" often co-occurs with "scared"; "rainbow" with "happy").
# This is the toddler's gut-feel layer.

OBJECT_DEFAULT_EMOTION = {
    "spider":   "scared",
    "monster":  "scared",
    "snake":    "scared",
    "fall":     "hurt",
    "broken":   "sad",
    "rainbow":  "happy",
    "sun":      "happy",
    "cookie":   "happy",
    "hug":      "loved",
    "mom":      "loved",
    "mother":   "loved",
    "dad":      "loved",
    "father":   "loved",
    "baby":     "loved",
    "puppy":    "happy",
    "kitten":   "happy",
    "rain":     "calm",
    "storm":    "scared",
    "dark":     "scared",
    "night":    "sleepy",
    "morning":  "happy",
}


# --- Grounding API -----------------------------------------------


def ground_emotions_in_lattice(lattice, verbose: bool = True) -> dict:
    """Attach emotion HVs to two sets of words in the lattice:

      1. Direct emotion words (happy, sad, scared, ...) get their
         own emotion_hv as the "emotion" semantic binding.
      2. Associated-emotion words (spider->scared, sun->happy) get
         a "default_emotion" binding pointing to the corresponding
         emotion's HV.

    Returns stats dict.
    """
    n_direct = 0
    n_assoc  = 0
    n_skip   = 0

    for word in list(lattice.word_mem.labels()):
        # Direct emotion word
        e_hv = emotion_hv(word)
        if e_hv is not None:
            lattice.bind_semantic(word, "emotion", e_hv)
            n_direct += 1
            continue
        # Associated emotion
        assoc = OBJECT_DEFAULT_EMOTION.get(word)
        if assoc:
            a_hv = emotion_hv(assoc)
            if a_hv is not None:
                lattice.bind_semantic(word, "default_emotion", a_hv)
                n_assoc += 1
                continue
        n_skip += 1

    stats = {
        "n_direct_emotions":     n_direct,
        "n_associated_emotions": n_assoc,
        "n_no_emotion":          n_skip,
        "total_words":           len(lattice.word_mem),
    }
    if verbose:
        print(f"[emotion] grounded {n_direct} direct + {n_assoc} "
              f"associated of {stats['total_words']} words",
              file=sys.stderr)
    return stats


def words_with_emotion(lattice,
                                  valence: str = None,
                                  arousal: str = None,
                                  category: str = None,
                                  top_k: int = 10) -> list[tuple[str, float]]:
    """Find words in the lattice whose emotion HV is most similar to
    the requested axis.  e.g. words_with_emotion(valence="positive",
    arousal="high") -> excited, joyful, surprised, ...
    """
    target = emotion_axis_hv(valence=valence, arousal=arousal,
                                          category=category)
    if not np.any(target):
        return []
    scored = []
    for word in lattice.word_mem.labels():
        sem = lattice.word_semantics.get(word, {})
        # `or` on numpy arrays raises; check None explicitly
        e_hv = sem.get("emotion")
        if e_hv is None:
            e_hv = sem.get("default_emotion")
        if e_hv is None:
            continue
        scored.append((word, similarity(target, e_hv)))
    scored.sort(key=lambda kv: -kv[1])
    return scored[:top_k]


def emotion_of(lattice, word: str) -> Optional[dict]:
    """Return {valence, arousal, category, kind} for a word's emotion."""
    if word.lower() in EMOTION_WORDS:
        spec = dict(EMOTION_WORDS[word.lower()])
        spec["kind"] = "direct"
        return spec
    if word.lower() in OBJECT_DEFAULT_EMOTION:
        emo_word = OBJECT_DEFAULT_EMOTION[word.lower()]
        spec = dict(EMOTION_WORDS[emo_word])
        spec["kind"]      = "associated"
        spec["via"]       = emo_word
        return spec
    return None


# --- CLI smoke test ----------------------------------------------


if __name__ == "__main__":
    import json
    print(f"=== Emotion lexicon ({len(EMOTION_WORDS)} direct emotions) ===\n")

    # Show emotion HV similarities within categories
    print("Same-category emotion sim (should be highest):")
    for c in ["joy", "fear", "anger", "sadness"]:
        words = [w for w, spec in EMOTION_WORDS.items()
                       if spec["category"] == c]
        if len(words) >= 2:
            sims = []
            for i, w1 in enumerate(words):
                for w2 in words[i+1:]:
                    sims.append(similarity(emotion_hv(w1), emotion_hv(w2)))
            print(f"  {c:10}: words={words}  avg_sim={sum(sims)/len(sims):.3f}")

    print("\nCross-category emotion sim (should be lower):")
    cross_pairs = [
        ("happy", "sad"),
        ("happy", "scared"),
        ("excited", "calm"),
        ("scared", "angry"),
        ("scared", "afraid"),  # same category
        ("happy", "joyful"),   # same category
    ]
    for w1, w2 in cross_pairs:
        h1 = emotion_hv(w1)
        h2 = emotion_hv(w2)
        if h1 is not None and h2 is not None:
            s = similarity(h1, h2)
            same_cat = EMOTION_WORDS[w1]["category"] == EMOTION_WORDS[w2]["category"]
            tag = "(same cat)" if same_cat else "(cross cat)"
            print(f"  {w1:10} vs {w2:10}  sim={s:.3f}  {tag}")

    print(f"\n=== Object -> default emotion lookups ===\n")
    for obj in ["spider", "rainbow", "mom", "fall", "storm", "cookie", "dark"]:
        spec = emotion_of(None, obj) if False else None  # we'll use the static lookup
        if obj in OBJECT_DEFAULT_EMOTION:
            emo = OBJECT_DEFAULT_EMOTION[obj]
            full = EMOTION_WORDS[emo]
            print(f"  {obj:10} -> feels {emo:8}  (valence={full['valence']}, "
                  f"arousal={full['arousal']}, category={full['category']})")
