"""autopilot/emotion.py — neurosymbolic emotion detection on user input.

Approach: HDC encode each emotion category as the BUNDLE of sample
sentences exhibiting that emotion.  At inference, encode the user's
message and compute Hamming similarity to each anchor; the highest-
matching anchor wins (unless all are too close to random).

This is a low-cost emotion classifier — no transformer, no API call,
no model file.  Lives entirely in the HDC substrate.

Five emotion categories:
  * frustrated  — user is annoyed, frustrated, stuck
  * excited     — user is enthusiastic, energized, celebrating
  * confused    — user is uncertain, asking for clarification
  * urgent      — user needs fast/concise help
  * casual      — neutral / conversational

Returns the dominant emotion (or None if confidence is too low).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))


# ─── Anchor sample sentences ─────────────────────────────────────


_EMOTION_ANCHORS = {
    "frustrated": [
        "why isn't this working", "this is so annoying",
        "i keep trying and it keeps failing",
        "i don't understand why this won't work",
        "ugh this is broken", "this is frustrating",
        "i've tried everything", "nothing i do works",
        "why does this keep happening", "this is ridiculous",
        "fix this", "this is the worst",
        "i'm so done with this", "what the hell",
        "this is making me crazy", "come on already",
        "stop doing that", "knock it off",
    ],
    "excited": [
        "let's go", "yes yes yes", "this is amazing",
        "i love it", "perfect", "awesome",
        "so cool", "great work", "incredible",
        "wow", "that's exactly what i wanted",
        "yes that's perfect", "let's do it",
        "this is going to be huge", "i'm pumped",
        "this is great news", "fantastic",
        "we're doing it", "this is the moment",
    ],
    "confused": [
        "i don't understand", "what does that mean",
        "wait what", "can you explain",
        "i'm not following", "huh",
        "what's going on", "i'm lost",
        "i don't get it", "can you clarify",
        "what are you talking about", "i need more context",
        "say that again", "rephrase that",
        "i'm confused", "make this clearer",
        "explain like i'm five", "walk me through it",
    ],
    "urgent": [
        "quick", "fast", "hurry", "asap", "right now",
        "no time", "make it quick",
        "give me the short version", "tldr",
        "in one sentence", "be brief",
        "just the facts", "skip the details",
        "we need this now", "no fluff",
        "what's the bottom line",
    ],
    "casual": [
        "hey there", "what's up", "how's it going",
        "just curious", "by the way", "side question",
        "random question", "tell me about",
        "what do you think", "i was wondering",
        "kind of curious about", "no big deal",
        "whenever you have a sec",
    ],
}


# ─── Build anchor HVs ────────────────────────────────────────────


_ANCHOR_HVS = None


def _build_anchors(encoder) -> dict[str, np.ndarray]:
    """Encode each sample sentence and bundle them per category."""
    from train.v5_hdc_prototype import bundle
    out: dict[str, np.ndarray] = {}
    for emotion, samples in _EMOTION_ANCHORS.items():
        if not samples:
            continue
        hvs = [encoder.encode(s) for s in samples]
        out[emotion] = bundle(hvs)
    return out


def get_anchors(encoder) -> dict[str, np.ndarray]:
    """Lazy-load + cache the anchor HVs.  First call encodes ~80
    sample sentences; subsequent calls reuse the result."""
    global _ANCHOR_HVS
    if _ANCHOR_HVS is None:
        _ANCHOR_HVS = _build_anchors(encoder)
    return _ANCHOR_HVS


# ─── Classifier ──────────────────────────────────────────────────


# Lightweight keyword priors — boost an emotion when a strong signal
# word appears, in case the HDC encoder isn't catching it.
_KEYWORD_BOOSTS = {
    "frustrated": {"ugh", "wtf", "damn", "fuck", "broken",
                       "stuck", "fail", "failing", "frustrat",
                       "annoyed", "annoying", "stop", "ridiculous"},
    "excited":    {"awesome", "amazing", "incredible", "perfect",
                       "yes!", "let's go", "fantastic", "love it",
                       "great", "!!", "wow"},
    "confused":   {"huh", "wait", "what?", "what.", "what,",
                       "don't understand", "confused", "lost",
                       "clarify", "explain"},
    "urgent":     {"asap", "hurry", "quick", "fast", "now",
                       "tldr", "brief", "short version"},
}


def classify_emotion(msg: str, encoder=None,
                          min_similarity: float = 0.52) -> Optional[str]:
    """Classify the user's message into one of the emotion categories,
    or None when confidence is too low.

    Strategy:
      1. Apply keyword priors — strong signal words shortcut the
         classifier (avoid encoder noise for very clear cases).
      2. If no keyword fires, use HDC anchor matching.
      3. Reject the prediction if the top similarity is too close to
         the second-best (i.e., the classifier isn't confident).
    """
    if not msg:
        return None
    low = msg.lower()

    # ── Keyword fast-path ──
    keyword_hits: dict[str, int] = {}
    for emotion, words in _KEYWORD_BOOSTS.items():
        hits = sum(1 for w in words if w in low)
        if hits:
            keyword_hits[emotion] = hits
    if keyword_hits:
        # Pick the emotion with the most keyword hits
        return max(keyword_hits, key=keyword_hits.get)

    # ── HDC anchor matching ──
    if encoder is None:
        return None
    try:
        anchors = get_anchors(encoder)
    except Exception:
        return None
    if not anchors:
        return None

    from train.v5_hdc_prototype import hamming_distance, D
    msg_hv = encoder.encode(msg)
    sims: dict[str, float] = {}
    for emotion, anchor_hv in anchors.items():
        d = hamming_distance(msg_hv, anchor_hv)
        sims[emotion] = 1.0 - d / D

    ranked = sorted(sims.items(), key=lambda kv: -kv[1])
    top_emotion, top_sim = ranked[0]
    second_sim = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_sim - second_sim

    # Confidence threshold: top sim must be above min_similarity AND
    # have a small margin over second place (otherwise it's ambiguous).
    if top_sim < min_similarity or margin < 0.005:
        return None
    return top_emotion


# ─── Smoke test ──────────────────────────────────────────────────


def _self_test():
    """Test keyword classifier (the HDC path needs an encoder)."""
    cases = [
        ("ugh this is broken",                       "frustrated"),
        ("yes! this is amazing",                     "excited"),
        ("wait what does that mean",                 "confused"),
        ("quick give me the answer",                 "urgent"),
        ("hey what's up",                            "casual"),
        ("when was einstein born",                   None),
        ("what is 23 times 47",                      None),
    ]
    for msg, expected in cases:
        result = classify_emotion(msg)
        mark = "OK " if (result == expected) or (
            expected is None and result not in {"frustrated", "excited",
            "confused", "urgent"}
        ) else "BAD"
        print(f"  [{mark}] {msg!r:<45}  → {result!r}  (expected {expected!r})")


if __name__ == "__main__":
    _self_test()
