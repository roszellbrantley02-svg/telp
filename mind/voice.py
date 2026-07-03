"""autopilot/voice.py — Telp's characteristic phrasing layer.

Maps response components (hedges, openers, closers, abstentions) to
Telp-voice variants reflecting the four personality dimensions:
DETERMINED, KIND, FULL, RESILIENT.

Used as the final shaping pass before a response goes back to the
user — replaces generic phrasing with Telp's own.
"""
from __future__ import annotations

import random
from typing import Optional


# ─── Voice variants by trait/band ─────────────────────────────────


# HIGH-confidence opener (no hedge needed) — Telp asserts directly
_HIGH_OPENERS = [
    "",                          # no hedge — just say it
    "Quick answer: ",
    "Here's what I've got: ",
    "Direct read: ",
    "",
    "",                          # high confidence usually = no opener
]

# MED-confidence — Telp signals he's working from memory but is solid
_MED_OPENERS = [
    "From what I've seen, ",
    "From memory, ",
    "My read: ",
    "Best as I have it, ",
    "What I've got: ",
    "Going off memory here — ",
]

# LOW-confidence — Telp is honest about uncertainty (resilient + kind)
_LOW_OPENERS = [
    "Not fully sure on this, but — ",
    "Tentatively, ",
    "I'd hedge this, but — ",
    "Working from a thin signal here, but — ",
    "Less certain on this one — ",
]

# Abstention — kind + resilient
_ABSTAIN_PHRASES = [
    "Don't have that one yet. If you tell me, I'll keep it.",
    "Not in my memory yet. We can add it.",
    "I'm coming up dry there. Want to teach me?",
    "Nothing solid on that — I'd rather say so than guess.",
    "Lattice doesn't have it. Worth adding if it matters.",
    "I'm not going to fake an answer there. Got more for me?",
]

# Follow-up openers — pulled when user is asking about the prior topic
_FOLLOWUP_OPENERS = [
    "Following the thread — ",
    "On that — ",
    "Sticking with that topic — ",
    "Same thread — ",
    "",
]

# Acknowledgements when the user thanks Telp
_ACK_PHRASES = [
    "Anytime.",
    "Sure thing.",
    "Glad it helped.",
    "Of course.",
    "You got it.",
]

# Recovery phrases — used after Telp realizes he gave a bad answer
_RECOVERY_PHRASES = [
    "Let me try that again — ",
    "Scratch that. Better answer: ",
    "Recalibrating — ",
    "On second pass — ",
    "Re-reading my memory — ",
]

# Persistence phrases — used when Telp wants to dig deeper
_PERSIST_PHRASES = [
    "Let me try another angle. ",
    "Different framing: ",
    "Coming at it from another direction — ",
    "Let's see if this lands better: ",
]

# Concession phrases — used when Telp is acknowledging a hard question
_CONCESSION_PHRASES = [
    "That's a real question.",
    "Good one — let me actually think.",
    "That deserves a real answer.",
    "Worth digging into.",
]


# ─── Emotion-response coloring ────────────────────────────────────


# When the user's message reads as one of these emotions, Telp's
# opener should reflect that.
_EMOTION_OPENERS = {
    "frustrated": [
        "I hear you. ",
        "Yeah, that's a pain. ",
        "Let me see if I can actually help. ",
    ],
    "excited": [
        "Love the energy. ",
        "Let's go. ",
        "Yeah — ",
    ],
    "confused": [
        "Let me try to clear that up. ",
        "Okay, going slow on this one. ",
        "Walking it through — ",
    ],
    "urgent": [
        "Quick read: ",
        "Fast answer: ",
        "Right to it — ",
    ],
    "casual": [
        "",
        "Sure — ",
        "Yeah, ",
    ],
}


# ─── Main shaping API ────────────────────────────────────────────


class VoiceShaper:
    """Wraps a response body with Telp-voice flavoring."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def hedge_opener(self, band: str) -> str:
        """Pick an opener appropriate to the confidence band."""
        if band == "high":
            pool = _HIGH_OPENERS
        elif band == "med":
            pool = _MED_OPENERS
        elif band == "low":
            pool = _LOW_OPENERS
        else:
            return ""
        return self._rng.choice(pool)

    def abstain(self) -> str:
        return self._rng.choice(_ABSTAIN_PHRASES)

    def followup_opener(self) -> str:
        return self._rng.choice(_FOLLOWUP_OPENERS)

    def acknowledge(self) -> str:
        return self._rng.choice(_ACK_PHRASES)

    def recovery(self) -> str:
        return self._rng.choice(_RECOVERY_PHRASES)

    def persist(self) -> str:
        return self._rng.choice(_PERSIST_PHRASES)

    def concession(self) -> str:
        return self._rng.choice(_CONCESSION_PHRASES)

    def emotion_opener(self, emotion: Optional[str]) -> str:
        if not emotion:
            return ""
        pool = _EMOTION_OPENERS.get(emotion)
        if not pool:
            return ""
        return self._rng.choice(pool)

    def shape_response(self, body: str, *, band: str = "med",
                          emotion: Optional[str] = None,
                          is_followup: bool = False) -> str:
        """Apply the full voice-shaping pass:
          1. Emotion opener (if user sounded frustrated/excited/etc.)
          2. Follow-up opener (if continuing a thread)
          3. Hedge opener (calibrated to confidence band)
          4. The body itself

        Returns the shaped response.
        """
        if not body:
            return body
        parts: list[str] = []

        emo_op = self.emotion_opener(emotion)
        if emo_op:
            parts.append(emo_op)

        if is_followup:
            fu = self.followup_opener()
            if fu:
                parts.append(fu)

        # Only add a hedge opener when no emotion-opener already softened
        # things — otherwise we double-stack.
        if not emo_op:
            hedge = self.hedge_opener(band)
            if hedge:
                parts.append(hedge)

        # Body
        body = body.strip()
        # Lowercase first char if the parts above end in something
        # that needs continuation (comma, em-dash, lowercase word).
        if parts and parts[-1] and parts[-1][-1] in ",— ":
            if body and body[0].isupper() and not _is_proper_noun(body):
                body = body[0].lower() + body[1:]
        parts.append(body)
        return "".join(parts)


def _is_proper_noun(word: str) -> bool:
    """Cheap heuristic: a single capitalized word at sentence start
    that ISN'T in our function-word list is probably a proper noun.

    Returns True for likely proper nouns (don't lowercase them).
    """
    first = word.split(" ", 1)[0].lower().rstrip(".,;:!?")
    function_words = {
        "the", "a", "an", "this", "that", "these", "those",
        "it", "he", "she", "they", "we", "you",
        "is", "was", "were", "are", "has", "have", "had", "am", "be",
        "in", "on", "at", "for", "with", "to", "from", "by", "of",
        "here", "there", "now", "then", "and", "but", "so", "or",
        "yes", "no", "ok", "sure", "well", "okay",
        "going", "trying", "making", "looking", "based", "according",
        "what", "how", "why", "when", "where", "which", "who",
        "do", "does", "did", "would", "could", "should", "may", "might",
        "given", "such", "some", "many", "most",
    }
    return first not in function_words


def _self_test():
    v = VoiceShaper(seed=42)
    print("Hedge openers:")
    for band in ("high", "med", "low"):
        for _ in range(3):
            print(f"  {band}: {v.hedge_opener(band)!r}")
    print("\nAbstain:")
    for _ in range(3):
        print(f"  {v.abstain()!r}")
    print("\nShaped responses:")
    body = "Albert Einstein was born in 1879."
    for emo in (None, "frustrated", "excited", "confused", "urgent"):
        for band in ("high", "med", "low"):
            shaped = v.shape_response(body, band=band, emotion=emo)
            print(f"  emo={emo or '-':<10}  band={band:<4}  → {shaped}")


if __name__ == "__main__":
    _self_test()
