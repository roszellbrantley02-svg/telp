"""
lattice/event_frames.py - Phase 12.6: frame semantics.

WHY
---
A sentence is not just a sequence of words.  It describes an EVENT —
something that happened, with participants and properties.  Until
now Telp could match `"the ball rolled down the hill"` as a
word-sequence but had no way to recover the structure:

  WHO did something?   (the ball)
  WHAT did they do?    (rolled)
  WHICH WAY?           (down)
  WHERE?               (the hill)

Frame semantics (Fillmore 1976, FrameNet) names these as thematic
roles bound to the event.  We adopt the standard inventory:

  AGENT     — who/what does the action  (the ball)
  ACTION    — the verb itself           (rolled)
  PATIENT   — who/what is acted on      (the floor — in "ball broke the floor")
  INSTRUMENT — tool used                 (with a hammer)
  GOAL      — endpoint                  (to the bottom)
  SOURCE    — origin                    (from the top)
  LOCATION  — where it happened         (the hill)
  DIRECTION — which way                 (down, up)
  TIME      — when                      (today, yesterday)
  MANNER    — how                       (quickly, sadly)
  RECIPIENT — to whom (give X to Y)     (the dog)
  EXPERIENCER — who feels               (I feel happy → I is experiencer)
  STIMULUS  — what triggers feeling     (spider in "spider scares me")

PARSING
-------
A small rule-based parser walks the sentence using:
  - Word category lookup (verbs vs nouns vs prepositions vs adjectives)
  - Position relative to the main verb
  - Preposition disambiguation (down -> DIRECTION, on -> LOCATION)
  - WordNet POS + lattice n-gram statistics as fallback

The parser is intentionally LIGHT.  It's not a full CCG parser.  It
covers the toddler-grade sentence shapes Telp's corpus actually
contains.  Unknown structures fall back to "raw words".

RESULT
------
Every line stored in the reading lattice now ALSO gets an event_hv:

  event_hv = bundle(
    bind(R_AGENT,    value_hv("ball")),
    bind(R_ACTION,   value_hv("rolled")),
    bind(R_DIRECTION,value_hv("down")),
    bind(R_LOCATION, value_hv("hill")),
  )

This is structurally identical to the trading reasoner's fact_hv —
which is what makes language and trading share a substrate.  Different
roles, same algebra.

Query: "show me events where AGENT=ball" -> unbind R_AGENT from
context-bundle, snap to "ball" via cleanup, return matching events.
"""
from __future__ import annotations
import sys
from typing import Optional

import numpy as np

from train.v5_hdc_prototype import D, bind, bundle, hamming_distance
from lattice.phoneme_hdc import _det_hv, similarity


# --- Thematic role HVs (deterministic) ----------------------------


R_AGENT       = _det_hv("frame_role::agent")
R_ACTION      = _det_hv("frame_role::action")
R_PATIENT     = _det_hv("frame_role::patient")
R_INSTRUMENT  = _det_hv("frame_role::instrument")
R_GOAL        = _det_hv("frame_role::goal")
R_SOURCE      = _det_hv("frame_role::source")
R_LOCATION    = _det_hv("frame_role::location")
R_DIRECTION   = _det_hv("frame_role::direction")
R_TIME        = _det_hv("frame_role::time")
R_MANNER      = _det_hv("frame_role::manner")
R_RECIPIENT   = _det_hv("frame_role::recipient")
R_EXPERIENCER = _det_hv("frame_role::experiencer")
R_STIMULUS    = _det_hv("frame_role::stimulus")
R_ATTRIBUTE   = _det_hv("frame_role::attribute")   # "the bear is RED"
R_QUESTION    = _det_hv("frame_role::question")    # marks an interrogative

ROLE_HVS = {
    "agent":       R_AGENT,
    "action":      R_ACTION,
    "patient":     R_PATIENT,
    "instrument":  R_INSTRUMENT,
    "goal":        R_GOAL,
    "source":      R_SOURCE,
    "location":    R_LOCATION,
    "direction":   R_DIRECTION,
    "time":        R_TIME,
    "manner":      R_MANNER,
    "recipient":   R_RECIPIENT,
    "experiencer": R_EXPERIENCER,
    "stimulus":    R_STIMULUS,
    "attribute":   R_ATTRIBUTE,
    "question":    R_QUESTION,
}


# --- Preposition -> role mapping ---------------------------------
#
# When a prepositional phrase appears after the verb, the preposition
# usually tells us which role its object plays.  Same preposition can
# play different roles in different contexts; this is a default.

PREP_TO_ROLE = {
    "down":     "direction",
    "up":       "direction",
    "over":     "direction",
    "under":    "location",
    "in":       "location",
    "on":       "location",
    "at":       "location",
    "near":     "location",
    "by":       "location",
    "inside":   "location",
    "outside":  "location",
    "to":       "goal",
    "toward":   "goal",
    "into":     "goal",
    "from":     "source",
    "with":     "instrument",
    "for":      "recipient",
}


# --- Word category lookups --------------------------------------
#
# Hand-curated from the corpus generator's VOCAB categories.  In a
# bigger system we'd use WordNet POS tagging; for our controlled
# corpus this is enough.

ARTICLES = {"the", "a", "an", "this", "that", "these", "those",
                "my", "your", "his", "her", "its", "our", "their"}

PRONOUNS = {"i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "us", "them"}

BE_VERBS = {"is", "are", "was", "were", "am", "be", "been", "being"}

LINKING_VERBS = BE_VERBS | {"feels", "feel", "felt", "seems",
                                            "seemed", "becomes", "became",
                                            "looks", "looked",
                                            "smells", "smelled",
                                            "tastes", "tasted",
                                            "sounds", "sounded",
                                            "gets", "got",
                                            "stays", "stayed"}

MODALS = {"can", "will", "must", "may", "might", "should", "would"}

TIME_WORDS = {"today", "yesterday", "tomorrow", "now", "then", "soon",
                  "later", "morning", "evening", "night", "day"}

WH_WORDS = {"who", "what", "when", "where", "why", "how"}

# NO hardcoded verb whitelist.  A word's role depends on POSITION,
# not on a static category.  "Ball" is a noun in "the ball rolled"
# because it comes after "the", and a verb in "ball up the paper"
# because it's sentence-initial.  We parse by SHAPE, not by word
# membership.


# --- Tokenize + identify ----------------------------------------


def _tokenize(text: str) -> list[str]:
    """Strip punctuation, lowercase, return word tokens."""
    import re
    tokens = re.findall(r"[A-Za-z']+|[?!.]", text)
    return tokens


def _is_unambiguous_verb_marker(token: str) -> bool:
    """Return True only for words that are ALMOST CERTAINLY verbs
    in toddler-grade English: be-verbs, modals, do-support, have-aux,
    and linking verbs.  These are closed-class function words.

    We do NOT call WordNet here.  Open-class words (`ball`, `bear`,
    `run`) can be either nouns or verbs depending on POSITION, and
    the parser determines that from sentence shape — not by asking
    whether the word can EVER be a verb.
    """
    return token.lower() in (BE_VERBS | MODALS | LINKING_VERBS |
                                      {"do", "does", "did",
                                       "have", "has", "had"})


def _is_preposition(token: str) -> bool:
    return token.lower() in PREP_TO_ROLE


def _looks_morphologically_verbal(token: str) -> bool:
    """Weak morphological hint: words ending in -ed or -ing are
    USUALLY verbal forms (past tense / progressive).  Not a hard
    rule (`red`, `bed`, `ring`, `string` are exceptions) but a
    helpful tiebreaker when position alone doesn't decide.
    """
    t = token.lower()
    if len(t) <= 3:
        return False
    if t.endswith("ed"):
        return True
    if t.endswith("ing"):
        return True
    return False


# --- Parser ------------------------------------------------------


def parse_event(text: str, lattice=None) -> dict:
    """Parse a single-sentence text into a frame.

    Position-based parsing, not vocabulary-based.  We never ask
    "is word X a verb?" because that depends on context (`ball`
    is a noun in "the ball" and a verb in "ball up paper").

    The parser walks SVO order:
      1. Detect sentence intent from terminal punctuation + leading wh-word
      2. Skip leading articles/possessives/demonstratives + optional
         leading time/sentence-adverbial
      3. The FIRST CONTENT WORD (or pronoun, or wh-word) is the AGENT
         head.  Any adjectives between article and head get attached
         as agent_modifiers.
      4. The NEXT WORD is the ACTION.  Period.  Regardless of whether
         WordNet thinks the word could also be a noun.
      5. Post-verb material is parsed by preposition lookup +
         article-skipping to fill DIRECTION / LOCATION / GOAL /
         SOURCE / PATIENT / etc.
    """
    tokens = _tokenize(text)
    if not tokens:
        return {"_raw": text, "_intent": "empty"}

    # ── Intent detection ───────────────────────────────────────
    final = tokens[-1] if tokens else ""
    first_lower = tokens[0].lower() if tokens else ""
    starts_wh = first_lower in WH_WORDS

    if final == "?" or starts_wh:
        intent = "question"
    elif final == "!":
        intent = "exclamation"
    elif (first_lower not in (ARTICLES | PRONOUNS | WH_WORDS | TIME_WORDS
                                       | {"yes", "no", "yeah", "nah", "hi",
                                          "hey", "oh", "okay", "wow", "hmm"})
            and not _is_unambiguous_verb_marker(first_lower)
            and len(tokens) >= 2
            and tokens[1] not in (".", "?", "!")):
        # Starts with a bare content word that's not a noun-cued
        # token: likely an imperative ("come here", "look at the star")
        intent = "command"
    else:
        intent = "statement"

    words = [t for t in tokens if t not in (".", "?", "!")]
    if not words:
        return {"_raw": text, "_intent": intent}

    frame: dict = {"_raw": text, "_intent": intent}
    if starts_wh:
        frame["question"] = words[0].lower()

    # ── Skip optional leading time-adverbial ("Yesterday the dog ran")
    i = 0
    if words[i].lower() in TIME_WORDS:
        frame["time"] = words[i].lower()
        i += 1

    # ── Sentence-initial fronted PP ("On the hill the bear saw a fox")
    # Detect "[PREP] [ARTICLE]? [NOUN]" at the front and bind it to
    # the corresponding role, then continue parsing the rest of the
    # sentence as the main clause.
    while i < len(words) and words[i].lower() in PREP_TO_ROLE:
        prep = words[i].lower()
        role = PREP_TO_ROLE[prep]
        j = i + 1
        while j < len(words) and words[j].lower() in ARTICLES:
            j += 1
        if j < len(words):
            # Bind the fronted noun to its role (e.g. location=hill)
            frame[role] = words[j].lower()
            i = j + 1
        else:
            i += 1
            break

    # ── Skip leading interjection comma-clauses ("Hi, the bear runs")
    # Handled at the tokenizer level — commas become breaks; this
    # parser receives one clause at a time.  But if the first content
    # word is an interjection, skip past it.
    if words[i].lower() in {"yes", "no", "yeah", "nah", "yep", "nope",
                                  "hi", "hey", "bye", "okay", "oh", "wow",
                                  "hmm"}:
        i += 1

    # ── For wh-questions, the wh-word might be in subject position
    # ("what is a spider?") or object position ("where does the duck swim?").
    # If it's in subject position, the wh-word IS the agent.
    # If it's in object position, skip it and look for "does/do/is" then
    # the real subject after.
    if intent == "question" and i < len(words) and words[i].lower() in WH_WORDS:
        wh = words[i].lower()
        i += 1
        if wh in {"what", "who"}:
            # Wh in subject position: "what is X?", "who runs?"
            frame["agent"] = wh
            # The verb is right after
            if i < len(words):
                frame["action"] = words[i].lower()
                i += 1
                # Optional attribute/object after the verb
                _parse_post_verb(words, i, frame)
            return frame
        else:
            # Wh in adverbial position: "where does the duck swim?"
            # Skip the auxiliary do/does/did
            if i < len(words) and words[i].lower() in {"do", "does", "did"}:
                i += 1

    # ── Find subject by SVO POSITION ─────────────────────────
    # Trust English subject-verb-object order:
    #   1. Skip leading articles/possessives/demonstratives
    #   2. The next content word IS the agent.  Period.
    #   3. The NEXT word after that is the action.  Period.
    # No verb detection at all — we trust position because the toddler
    # corpus is overwhelmingly SVO.  Multi-adjective noun phrases
    # ("the big brown bear") are not the common case; the simple rule
    # gives the right answer on the vast majority.
    while i < len(words) and words[i].lower() in ARTICLES:
        i += 1
    if i < len(words):
        frame["agent"] = words[i].lower()
        i += 1
        # Conjunction handling: "the bear and the frog played" —
        # if "and" follows, capture additional agents until the
        # verb position
        if i < len(words) and words[i].lower() == "and":
            extra_agents = [frame["agent"]]
            i += 1   # skip "and"
            while i < len(words) and words[i].lower() in ARTICLES:
                i += 1
            if i < len(words):
                extra_agents.append(words[i].lower())
                i += 1
            frame["agent"] = " and ".join(extra_agents)

    # ── Position-based verb detection ────────────────────────
    # The first word AFTER the subject head is the action — by
    # English SVO position, regardless of dictionary POS.
    if i < len(words):
        frame["action"] = words[i].lower()
        i += 1

    # ── Parse the rest ─────────────────────────────────────────
    _parse_post_verb(words, i, frame)

    return frame


def _parse_post_verb(words: list[str], start: int, frame: dict) -> None:
    """Fill PATIENT / DIRECTION / LOCATION / GOAL / etc. from material
    after the main verb."""
    action = frame.get("action", "")
    i = start
    # Linking-verb shape: the next non-article word is the attribute
    if action in LINKING_VERBS:
        # Skip articles
        while i < len(words) and words[i].lower() in ARTICLES:
            i += 1
        if i < len(words) and not _is_preposition(words[i]):
            frame["attribute"] = words[i].lower()
            i += 1
    else:
        # Transitive: first noun phrase = patient
        if i < len(words):
            w = words[i].lower()
            if w in ARTICLES:
                i += 1
                if i < len(words):
                    frame["patient"] = words[i].lower()
                    i += 1
            elif w in PRONOUNS:
                frame["patient"] = w
                i += 1
            elif not _is_preposition(w):
                frame["patient"] = w
                i += 1

    # Walk remaining prepositional phrases
    while i < len(words):
        w = words[i].lower()
        if _is_preposition(w):
            role = PREP_TO_ROLE[w]
            # Skip the prep, then articles, take the next content word
            j = i + 1
            while j < len(words) and words[j].lower() in ARTICLES:
                j += 1
            if j < len(words):
                frame[role] = words[j].lower()
                i = j + 1
            else:
                i += 1
        elif w in TIME_WORDS and "time" not in frame:
            frame["time"] = w
            i += 1
        else:
            i += 1


# --- Event HV composition ----------------------------------------


def event_hv(frame: dict) -> np.ndarray:
    """Bind each role-value pair from the frame into one event HV.

    Reserved meta-keys (start with _) are skipped.
    """
    from lattice.phoneme_hdc import compose_word
    from lattice.g2p_and_prosody import word_to_phonemes
    components = []
    for role, value in frame.items():
        if role.startswith("_"):
            continue
        role_hv = ROLE_HVS.get(role)
        if role_hv is None:
            continue
        # Encode the value via its phoneme composition (so similar
        # words give similar value HVs naturally).  Fall back to
        # deterministic hash for unknown tokens.
        phon = word_to_phonemes(str(value))
        if phon:
            v_hv = compose_word([p for p, _ in phon])
        else:
            v_hv = _det_hv(f"frame_value::{value}")
        components.append(bind(role_hv, v_hv))
    if not components:
        return np.zeros(D, dtype=np.int8)
    return bundle(components)


# --- Query API ---------------------------------------------------


def search_events(events: list[dict],
                          role: str,
                          value: str) -> list[dict]:
    """Find frames where `role` has the given `value`.

    Simple exact match for now.  Future: HV-similarity search using
    role unbind + cleanup.
    """
    v = value.lower()
    return [f for f in events if f.get(role, "").lower() == v]


# --- CLI smoke test ----------------------------------------------


if __name__ == "__main__":
    import json
    test_sentences = [
        # Classic frame examples
        "The ball rolled down the hill.",
        "The bear ran to the forest.",
        "The cat is happy.",
        "The cat is in the house.",
        "Yesterday the dog ran.",
        "The mouse ate the cheese.",
        "I see a red bird.",
        "I feel scared.",
        # Questions
        "What is a spider?",
        "Where does the duck swim?",
        "Why does the baby cry?",
        # Imperatives
        "Come here!",
        "Look at the star.",
        # Linking verb with attribute
        "The fox was quick.",
        # Compound
        "The bear walked from the cave to the river.",
        # Family
        "My mother sings in the morning.",
    ]
    for s in test_sentences:
        f = parse_event(s)
        # Pretty print
        intent = f.pop("_intent", "?")
        raw = f.pop("_raw", "")
        keys_order = ["agent", "action", "patient", "attribute",
                          "direction", "location", "goal", "source",
                          "instrument", "recipient", "time", "manner",
                          "experiencer", "stimulus", "question"]
        parts = [f"intent={intent}"]
        for k in keys_order:
            if k in f:
                parts.append(f"{k}={f[k]}")
        print(f"  {raw!r}")
        print(f"    -> {', '.join(parts)}")
        # And the event HV similarity check
        # (just a sanity check that two sentences with same agent give
        # similar HVs)
    print()

    # Similarity tests
    print("=== Event HV similarity (same role -> high similarity) ===")
    f1 = parse_event("The ball rolled down the hill.")
    f2 = parse_event("The ball jumped up the tree.")
    f3 = parse_event("The cat ran into the house.")
    sim_same_agent  = similarity(event_hv(f1), event_hv(f2))
    sim_diff_agent  = similarity(event_hv(f1), event_hv(f3))
    print(f"  ball/ball   sim={sim_same_agent:.3f}  (same AGENT, share R_AGENT-ball)")
    print(f"  ball/cat    sim={sim_diff_agent:.3f}  (different AGENT)")
