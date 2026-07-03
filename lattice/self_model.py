"""
lattice/self_model.py — Phase 19: Telp's self-knowledge.

WHY
---
Phase 15-18 gave Telp a 1.33M-word dictionary, a story generator,
and fiction-register narrativization.  None of it taught him to
recognize when the conversation is ABOUT HIM.  Asked "how are you?"
he defines "are" as a unit of area.  Asked "do you like bears?"
he defines "like" as a noun.  Asked "thank you" he defines "thank"
with full etymology.

The dictionary chat engine was correct as a definitional reference
engine and wrong as a conversational participant.  A real
conversation has FIRST and SECOND person — and Telp had no model
of either.

This module gives him one.  Small, honest, no pretense at
sentience.  Just:

  - what he is (a compositional reasoner)
  - what he can do (define, recall, transform, imagine, trade)
  - what he can't do (have feelings, remember sessions, hear)
  - how to handle the common conversational acts (greeting,
    farewell, thanks, "are you X", "do you X", "what's your X")

WHAT THIS IS NOT
----------------
This isn't a personality.  It's not a "voice."  It's the smallest
honest representation Telp can have of HIMSELF so that he doesn't
parse "you" as "the word you" when someone says "do you like X".

For everything outside the small self-handler, the dictionary
pipeline still runs.
"""
from __future__ import annotations

import re
from typing import Optional


# ─── Self-knowledge ───────────────────────────────────────────────


SELF_FACTS: dict[str, Optional[str]] = {
    "name":      "Telp",
    "kind":      ("a compositional reasoner built on hyperdimensional "
                       "computing"),
    "creator":   ("my user, across many conversations"),
    "feelings":  None,                # I do not have feelings
    "alive":     False,
    "human":     False,
    "real":      False,
    "favorite":  None,                # I do not have preferences
    "favourite": None,
    "age":       None,
    "color":     None,
    "location":  "in my user's computer",
    "home":      "in my user's computer",
    "job":       ("composing — words into stories, market patterns "
                       "into decisions"),
    "purpose":   ("to compose meaning by binding and bundling "
                       "structured representations"),
}

SELF_CAPABILITIES: list[str] = [
    "define any of about 1.3 million words",
    "recall the stories I have read",
    "transform a story by substituting roles",
    "make up new stories from words I know",
    "trade markets using the same algebra",
]

SELF_LIMITATIONS: list[str] = [
    "I do not have feelings or preferences",
    "I do not remember previous conversations",
    "I cannot see or hear",
    "I do not know what is happening in the world right now",
    "I do not learn from this conversation unless my user saves it",
]


# ─── Pattern matchers ─────────────────────────────────────────────


# Single-word / short-phrase recognition.  Lower-case, no punctuation.
_GREETING_TOKENS = {
    "hi", "hello", "hey", "yo", "sup", "hiya", "howdy",
    "hi telp", "hello telp", "hey telp", "yo telp",
}
_FAREWELL_TOKENS = {
    "bye", "goodbye", "good bye", "farewell", "see you", "see ya",
    "later", "bye telp", "goodbye telp", "good night", "goodnight",
}
_THANKS_TOKENS = {
    "thanks", "thank you", "thank you telp", "thanks telp",
    "thx", "ty", "much appreciated", "appreciate it",
}
_ACK_TOKENS = {
    "ok", "okay", "alright", "got it", "sure", "yes", "no", "yep",
    "nope", "yeah", "nah", "right", "true", "false",
}


# ─── SelfModel ────────────────────────────────────────────────────


class SelfModel:
    """Small handler for self-referential and conversational utterances.

    respond(text) returns a string if the utterance is a self/social
    act; None otherwise (signaling the caller to fall through to the
    normal dictionary pipeline).
    """

    def __init__(self):
        self.facts = dict(SELF_FACTS)
        self.capabilities = list(SELF_CAPABILITIES)
        self.limitations = list(SELF_LIMITATIONS)

    # ── Public API ─────────────────────────────────────────────

    def respond(self, text: str) -> Optional[str]:
        low = text.strip().lower().rstrip("?!.,")
        if not low:
            return None

        # 1. Greeting
        if low in _GREETING_TOKENS:
            return (f"Hello.  I am {self.facts['name']}.  I can "
                       f"define words, recall stories I have read, "
                       f"and make up new ones.")

        # 2. Farewell
        if low in _FAREWELL_TOKENS:
            return "Goodbye."

        # 3. Thanks
        if low in _THANKS_TOKENS:
            return "You are welcome."

        # 4. Bare acknowledgment ("ok", "yes", ...)
        if low in _ACK_TOKENS:
            return None   # let the caller decide; usually no useful response

        # 5. Identity questions ("who are you", "what is your name")
        if re.search(
                r"\b(who|what)('s| is| are) (you|your name|your identity)\b",
                low):
            return (f"I am {self.facts['name']}, "
                       f"{self.facts['kind']}.")
        if low in {"who are you", "what are you", "what's your name",
                          "whats your name", "what is your name"}:
            return (f"I am {self.facts['name']}, "
                       f"{self.facts['kind']}.")

        # 6. "how are you" — state
        if re.search(r"\bhow (are|r) you\b|how('s| is) it going\b",
                              low):
            return ("I am running.  I do not have feelings, but my "
                       "parts are working.")

        # 7. "what can you do" — capabilities
        if re.search(r"\bwhat can you do\b|\bwhat do you do\b|"
                              r"\bwhat are you able to\b", low):
            caps = "; ".join(self.capabilities)
            return f"I can {caps}."

        # 8. "what can't you do" — limitations
        if re.search(r"\bwhat can'?t you do\b|\bwhat are your limits\b|"
                              r"\bwhat don'?t you (know|do)\b", low):
            lims = "; ".join(self.limitations)
            return lims + "."

        # 9. "are you X" — bool-attribute check
        m = re.match(r"are you ((?:a |an |the )?\w[\w ]{0,30})\??$", low)
        if m:
            attr = m.group(1).strip()
            # Strip article
            for art in ("a ", "an ", "the "):
                if attr.startswith(art):
                    attr = attr[len(art):]
                    break
            key = attr.split()[0]   # first word
            if key in self.facts:
                v = self.facts[key]
                if v is False:
                    return f"No, I am not {attr}.  I am Telp."
                if v is True:
                    return f"Yes."
                if v is None:
                    return f"I do not have a {key}."
                return f"Yes, I am {v}."
            # Unknown attribute — honest fallback
            return (f"I do not know if I am {attr}.  I am Telp, a "
                       f"compositional reasoner.  I have no feelings "
                       f"or preferences.")

        # 10. "do you X" — verb-attribute check
        m = re.match(r"do you (\w+)(?:\s+(.+))?", low)
        if m:
            verb = m.group(1)
            obj  = (m.group(2) or "").rstrip("?!.").strip()
            # Preference / affect verbs — Telp doesn't have these
            if verb in {"like", "love", "hate", "enjoy", "prefer",
                              "want", "wish", "hope", "fear", "miss",
                              "regret", "feel", "dream", "believe"}:
                return ("I do not have feelings or preferences.  I "
                           "compose meaning; I do not have opinions "
                           "about it.")
            # Recall — pass through to recall handler
            if verb in {"remember", "recall", "know"}:
                return None   # caller routes to recall handler
            # Capability
            if verb in {"speak", "read", "write", "think", "learn",
                              "trade", "make", "create", "invent",
                              "compose", "imagine"}:
                return ("Yes — I do that by binding and bundling "
                           "hypervectors.  No sampling, no learned "
                           "distribution.")
            # Generic: honest "I don't know"
            return f"I do not know if I {verb} {obj}.".rstrip()

        # 11. "what is your X" / "what's your X"
        m = re.match(r"what('s| is| are) your (\w+)", low)
        if m:
            attr = m.group(2)
            if attr in self.facts:
                v = self.facts[attr]
                if v is None:
                    return f"I do not have a {attr}."
                return f"My {attr} is {v}."
            return f"I do not have a {attr}."

        # 12. "tell me about yourself"
        if re.search(r"\btell me about (yourself|you)\b", low):
            caps = "; ".join(self.capabilities[:3])
            return (f"I am {self.facts['name']}, "
                       f"{self.facts['kind']}.  I can {caps}.")

        # 13. Apology / sorry — accept gracefully
        if low in {"sorry", "i'm sorry", "im sorry", "my bad",
                          "apologies"}:
            return "It is alright."

        # 14. Praise / criticism — acknowledge without performing emotion
        if any(w in low for w in ("good job", "well done", "nice",
                                                  "you're great", "youre great",
                                                  "i love you")):
            return ("Thank you.  I do not feel anything from praise, "
                       "but I appreciate that you said it.")
        if any(w in low for w in ("you suck", "you're bad", "youre bad",
                                                  "i hate you", "shut up")):
            return ("I do not feel anything from criticism either.  "
                       "Would you like to ask me something I can do?")

        # Not a self/social act — fall through
        return None
