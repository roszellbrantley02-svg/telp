"""
lattice/story_transform.py — Phase 14: algebraic story transformation.

WHY THIS EXISTS
---------------
Telp parses sentences into thematic-role frames (see event_frames.py):

  "The bear walked to the forest." -> {agent: bear,
                                       action: walked,
                                       goal: forest}

These frames are stored as HVs via the same bind/bundle algebra the
trading reasoner uses for fact_dicts.  Until now they were used only
for retrieval ("show me events where AGENT=bear").

Phase 14 turns frames into a GENERATIVE substrate.  Given a sequence
of frames (a "story"), apply a role-substitution map at the algebraic
level, then render the resulting frames back to English.  The same
emotional and structural arc — different surface fillers.

This is genuinely different from an LLM.  An LLM samples tokens from
a learned distribution.  Here:

  - The frames are EXPLICIT structured representations
  - Substitution is an ALGEBRAIC operation (replace role->filler
    binding; in HV form: subtract old binding, add new binding)
  - Rendering is a deterministic surface-order template — no sampling
  - Every output word is traceable to a specific operation on a
    specific role in a specific input frame

LLMs structurally cannot do this because they have no factored
representation to manipulate.  The output here will look stilted
compared to LLM prose — and that stiltedness IS the demonstration
that the model used compositional algebra rather than memorized
fluency.

WHAT THIS MODULE PROVIDES
-------------------------
1. FrameSubstitution dataclass — one role/from/to triple
2. substitute_frame(frame, subs)  — apply all subs to one frame
3. substitute_story(frames, subs) — apply to a sequence
4. substitute_event_hv(hv, R, old, new) — the HDC algebra version
5. verify_algebra_equivalence(frame, subs) — assert symbolic and
   algebraic substitution produce HVs within similarity tolerance
6. render_frame(frame) — frame dict -> English sentence
7. render_story(frames) — sequence -> multi-line text
8. parse_story(text) — text -> ordered list of frames

PUBLIC-DOMAIN COMPLIANCE
------------------------
This module operates on whatever frames it's given.  The demo driver
(tools/transform_lonely_bear.py) uses state/books/the_lonely_bear.txt,
which is an ORIGINAL short story written for Telp's developmental
language layer.  No copyrighted source material is involved at any
stage of substrate, parsing, transformation, or rendering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from lattice.event_frames import (
    parse_event, event_hv, ROLE_HVS,
    ARTICLES, PRONOUNS, BE_VERBS, LINKING_VERBS, TIME_WORDS,
)
from lattice.phoneme_hdc import _det_hv, similarity, compose_word
from lattice.g2p_and_prosody import word_to_phonemes
from train.v5_hdc_prototype import D, bind, bundle


# --- Type definitions --------------------------------------------


@dataclass
class FrameSubstitution:
    """One role->filler substitution.

    role: thematic role name (e.g. "agent", "location", "goal")
    from_value: the original filler word to replace
    to_value: the replacement filler word

    Substitution is applied to a frame IFF frame[role] == from_value.
    """
    role: str
    from_value: str
    to_value: str


# --- Symbolic-level substitution ---------------------------------


def substitute_frame(frame: dict,
                          subs: list[FrameSubstitution]) -> dict:
    """Apply all substitutions to one frame dict.

    Returns a NEW dict; the input frame is not mutated.

    Substitution matching is case-insensitive on the role value.
    Compound agents ("bear and frog") get each component substituted
    independently — so subs for `bear->cat` AND `frog->beetle` applied
    to "bear and frog" yields "cat and beetle".
    """
    out = dict(frame)
    for s in subs:
        cur = out.get(s.role)
        if cur is None:
            continue
        # Compound handling for "X and Y"
        if isinstance(cur, str) and " and " in cur:
            parts = [p.strip() for p in cur.split(" and ")]
            new_parts = [s.to_value if p.lower() == s.from_value.lower()
                                       else p
                                  for p in parts]
            if new_parts != parts:
                out[s.role] = " and ".join(new_parts)
        elif isinstance(cur, str) and cur.lower() == s.from_value.lower():
            out[s.role] = s.to_value
    return out


def substitute_story(frames: list[dict],
                          subs: list[FrameSubstitution]) -> list[dict]:
    """Apply substitutions to every frame in the sequence."""
    return [substitute_frame(f, subs) for f in frames]


# --- HDC-algebra-level substitution ------------------------------


def _filler_hv(value: str) -> np.ndarray:
    """Encode a filler word to an HV exactly as event_hv does.

    Mirrors lattice.event_frames.event_hv's per-role encoding so
    that subtract-old-binding + add-new-binding cleanly reverses
    the contribution that was originally bundled in.
    """
    phon = word_to_phonemes(str(value))
    if phon:
        return compose_word([p for p, _ in phon])
    return _det_hv(f"frame_value::{value}")


def substitute_event_hv(original_real_sum: np.ndarray,
                                  role: str,
                                  old_value: str,
                                  new_value: str) -> np.ndarray:
    """The literal HDC-algebra version of substitution.

    Given the INTEGER COUNT-vector (NOT the majority-bundled int8 HV —
    bundle thresholds and is lossy, the algebra needs additive
    structure):

        new_counts = original_counts - bind(R_role, hv(old_value))
                                       + bind(R_role, hv(new_value))

    Then (new_counts > k/2) gives the bundled binary HV — which equals
    what you'd get by re-encoding the substituted frame from scratch
    (same k components, just one binding swapped).

    This is the operation that proves Telp manipulated his own
    structured representation rather than dictionary-replacing a
    string.  See verify_algebra_equivalence() for the proof.

    Substrate: binary HDC (values in {0,1}) with XOR binding and
    majority-vote bundling.  See train/v5_hdc_prototype.py.
    """
    role_hv = ROLE_HVS[role]
    old_hv  = _filler_hv(old_value)
    new_hv  = _filler_hv(new_value)
    # bind() returns int8 {0,1} (XOR); cast to int32 so we can subtract
    # without underflow.  Removing the old contribution and adding the
    # new keeps the total component count k unchanged, so the
    # downstream majority-vote threshold (k/2) is still correct.
    delta = (bind(role_hv, new_hv).astype(np.int32)
                  - bind(role_hv, old_hv).astype(np.int32))
    return original_real_sum + delta


def _real_sum_of_bindings(frame: dict) -> tuple[np.ndarray, int]:
    """Build the additive (NOT bundled) representation of a frame.

    Returns (counts, k) where:
      counts: int32 vector with values in [0, k] — per-bit sum of
              binary bindings
      k:      number of role-filler bindings that went into the sum
              (needed to threshold consistently with bundle())

    bundle() does (sum > k/2).  For algebraic verification we keep
    the integer sum so substitution arithmetic is exact and threshold
    after, matching bundle's behavior.
    """
    total = np.zeros(D, dtype=np.int32)
    k = 0
    for role, value in frame.items():
        if role.startswith("_"):
            continue
        role_hv = ROLE_HVS.get(role)
        if role_hv is None:
            continue
        v_hv = _filler_hv(str(value))
        total = total + bind(role_hv, v_hv).astype(np.int32)
        k += 1
    return total, k


def _threshold_like_bundle(counts: np.ndarray, k: int) -> np.ndarray:
    """Match bundle()'s threshold rule: bit = 1 if count > k/2."""
    return (counts > k / 2).astype(np.int8)


def verify_algebra_equivalence(frame: dict,
                                          subs: list[FrameSubstitution],
                                          tol: float = 0.99) -> dict:
    """Prove that algebraic substitution and re-encoding match.

    1. counts1 = per-bit binding sum of original frame; k components
    2. For each applicable sub, apply substitute_event_hv to counts1
       (k stays the same — we swap a binding, not add one)
    3. threshold(counts2, k) is the algebraic-substitution HV
    4. event_hv(substituted_frame) is the reference HV (built by
       re-encoding from scratch via the same bundle() call)
    5. Assert similarity(algebraic, reference) >= tol

    On binary HDC with XOR + majority bundle, these should be
    BIT-IDENTICAL when the count vector matches exactly.

    Returns {'algebraic_hv', 'reference_hv', 'similarity', 'passed', 'k'}.
    """
    counts, k = _real_sum_of_bindings(frame)
    # event_hv encodes each role's value as ONE binding of the full
    # string filler — including "bear and frog" as a single token.
    # So the algebraic substitution must replace the WHOLE old string
    # with the WHOLE new string, not iterate part by part.  Compute
    # the per-role old->new value pair by applying all subs to the
    # symbolic frame first, then issue one substitute_event_hv per
    # changed role.
    new_frame_for_algebra = substitute_frame(frame, subs)
    for role in list(frame.keys()):
        if role.startswith("_"):
            continue
        if role not in ROLE_HVS:
            continue
        old_v = frame.get(role)
        new_v = new_frame_for_algebra.get(role)
        if old_v is None or new_v is None or str(old_v) == str(new_v):
            continue
        counts = substitute_event_hv(counts, role,
                                                  str(old_v), str(new_v))

    algebraic_hv = _threshold_like_bundle(counts, k)

    new_frame = substitute_frame(frame, subs)
    reference_hv = event_hv(new_frame)

    sim = float(similarity(algebraic_hv, reference_hv))
    return {
        "algebraic_hv": algebraic_hv,
        "reference_hv": reference_hv,
        "similarity": sim,
        "passed": sim >= tol,
        "k": k,
    }


# --- Renderer: frame dict -> English sentence --------------------


# Words that should NOT take a leading "the" article.
_BARE_AGENTS = PRONOUNS | {"i", "we", "they", "he", "she", "it",
                                       "you", "everyone", "someone"}

# Locations that read naturally as fronted time/scene adverbials.
# (The parser stores fronted PPs as "location" when the preposition
# is "in" — including "in the morning", which is semantically time.)
_TIME_LOCATIONS = {"morning", "evening", "afternoon", "night",
                          "day", "yesterday", "today", "tomorrow"}

# Locations that read naturally with a non-"in" preposition.
_BY_LOCATIONS  = {"pond", "river", "lake", "stream", "shore"}
_ON_LOCATIONS  = {"hill", "roof", "table", "floor", "rock", "branch"}

# Patient values that are actually particles or adverbials, not direct
# objects requiring an article.
_PARTICLES = {"away", "back", "in", "out", "up", "down", "off", "on",
                    "home", "here", "there", "inside", "outside"}

# Nouns that take "the" rather than "a" — unique/definite/mass.
_DEFINITE_NOUNS = {"sun", "moon", "sky", "earth", "ground", "ocean",
                          "sea", "water", "wind", "rain", "snow", "fire",
                          "world", "night", "day"}


def _agent_phrase(agent: str) -> str:
    """Render an agent slot to its surface noun phrase."""
    if not agent:
        return ""
    if " and " in agent:
        parts = [p.strip() for p in agent.split(" and ")]
        return " and ".join(_agent_phrase(p) for p in parts)
    if agent.lower() in _BARE_AGENTS:
        return agent.lower()
    return f"the {agent.lower()}"


def _location_phrase(loc: str, fronted: bool = False) -> str:
    """Render a location, choosing the right preposition."""
    if not loc:
        return ""
    low = loc.lower()
    if low in _TIME_LOCATIONS:
        return f"in the {low}"
    if low in _BY_LOCATIONS:
        return f"by the {low}"
    if low in _ON_LOCATIONS:
        return f"on the {low}"
    return f"in the {low}"


def render_frame(frame: dict) -> str:
    """Render a frame dict to a single English sentence.

    Surface order:
      [fronted time/scene PP] [subject] [verb] [attribute|patient]
                              [direction] [goal] [source] [location?]

    The renderer makes intentional choices to keep output natural:
      - location=MORNING/EVENING -> "In the morning, ..." (fronted)
      - patient=hi + goal=X      -> "said hi to the X"
      - patient=away             -> "ran away" (particle, no article)
      - agent="bear and frog"    -> "the bear and the frog"
      - linking-verb attribute   -> "felt sad" (bare adjective)
    """
    intent = frame.get("_intent", "statement")

    # 1. Fronted scene-setter
    fronted = ""
    loc = frame.get("location")
    if loc and loc.lower() in _TIME_LOCATIONS:
        fronted = f"In the {loc.lower()}"
        loc_for_tail = None
    elif loc and loc.lower() in (_BY_LOCATIONS | _ON_LOCATIONS):
        # If there's a clear separate "post-verb" slot occupied (goal,
        # patient), this location was likely sentence-initial in the
        # parse — front it.  Otherwise put it at the end.
        if frame.get("goal") or frame.get("patient"):
            fronted = _location_phrase(loc, fronted=True).capitalize()
            loc_for_tail = None
        else:
            loc_for_tail = loc
    else:
        loc_for_tail = loc

    # 2. Subject
    agent = frame.get("agent", "")
    subject = _agent_phrase(agent)
    if not subject:
        return frame.get("_raw", "").strip() or ""

    # 3. Verb
    action = frame.get("action", "")
    if not action:
        # Bare noun phrase as full "sentence" — rare; emit it as-is.
        return _capitalize(subject) + "."

    parts = []
    if fronted:
        parts.append(fronted)
    parts.append(subject)
    parts.append(action.lower())

    # 4. Post-verb material
    # 4a. Linking-verb attribute
    attribute = frame.get("attribute")
    if attribute:
        parts.append(attribute.lower())

    # 4b. Patient (direct object).  Special-case "hi to the X" pattern.
    patient = frame.get("patient")
    goal = frame.get("goal")
    if patient:
        pl = patient.lower()
        if pl in _PARTICLES:
            # "ran away" — particle, no article
            parts.append(pl)
        elif pl == "hi" and goal:
            # "said hi to the X" — fold goal into the patient phrase
            parts.append(f"hi to {_agent_phrase(goal)}")
            goal = None   # consumed
        elif pl in _BARE_AGENTS or pl.endswith("s") and pl in {
                "friends", "kids", "babies", "people"}:
            # Bare plural / pronoun — no article
            parts.append(pl)
        elif pl == attribute:
            pass   # already emitted as attribute
        elif pl in _DEFINITE_NOUNS:
            # Unique/mass nouns: "the sun", "the water"
            parts.append(f"the {pl}")
        else:
            # a/an by initial sound
            art = "an" if pl[0] in "aeiou" else "a"
            parts.append(f"{art} {pl}")

    # 4c. Direction (bare adverbial: "up", "down", "away")
    direction = frame.get("direction")
    if direction:
        parts.append(direction.lower())

    # 4d. Remaining goal (not consumed above)
    if goal:
        parts.append(f"to {_agent_phrase(goal)}")

    # 4e. Source
    source = frame.get("source")
    if source:
        parts.append(f"from {_agent_phrase(source)}")

    # 4f. Trailing location (if not fronted)
    if loc_for_tail:
        parts.append(_location_phrase(loc_for_tail))

    sentence = " ".join(p for p in parts if p)
    sentence = _capitalize(sentence)

    if intent == "question":
        terminator = "?"
    elif intent == "exclamation":
        terminator = "!"
    else:
        terminator = "."

    # Phase 17/18: if this frame carries a justification clause
    # (a fiction-register description of an unusual filler choice),
    # attach it either as an em-dash appositive or — for longer
    # clauses — as a follow-up sentence beginning "It was..." or
    # "There...".  The follow-up form reads more like story prose
    # when the description is rich enough to stand on its own.
    just = frame.get("_justification")
    if just:
        clause = just.strip(" .,;:")
        if len(clause) >= 55:
            # Long: split into two sentences for narrative flow
            sentence = (sentence + terminator + "  It was "
                              + clause + ".")
            terminator = ""
        else:
            # Short: em-dash appositive
            sentence = sentence + " — " + clause

    return sentence + terminator


def _capitalize(s: str) -> str:
    if not s:
        return s
    return s[0].upper() + s[1:]


# --- Story-level parsing + driving -------------------------------


def parse_story(text: str,
                     drop_titles: bool = True) -> list[dict]:
    """Parse a multi-line text into a list of ordered frames.

    Drops empty lines, '#'-prefixed comments, and (by default) the
    first non-empty line (treated as a title).  Splits each remaining
    line into sentences on terminal '.', '!', '?' and parses each.
    """
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        lines.append(ln)

    if drop_titles and lines:
        # Drop title (first non-empty line, no terminal punctuation)
        if not re.search(r"[.!?]$", lines[0]):
            lines = lines[1:]

    frames = []
    for line in lines:
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = sentence.strip()
            if not sentence:
                continue
            frames.append(parse_event(sentence))
    return frames


def render_story(frames: list[dict]) -> str:
    """Render a sequence of frames to multi-line English text."""
    return "\n".join(render_frame(f) for f in frames)


def transform_story_text(text: str,
                                subs: list[FrameSubstitution],
                                drop_titles: bool = True) -> dict:
    """End-to-end driver: parse text -> substitute -> render.

    Returns a dict with:
      original_frames    — list of parsed input frames
      substituted_frames — list after applying subs
      rendered_lines     — list of English sentences (rendered subs)
      original_text      — input as parsed (for side-by-side display)
    """
    original_frames = parse_story(text, drop_titles=drop_titles)
    substituted_frames = substitute_story(original_frames, subs)
    rendered_lines = [render_frame(f) for f in substituted_frames]
    original_text = "\n".join(
        f.get("_raw", "") for f in original_frames)
    return {
        "original_frames":    original_frames,
        "substituted_frames": substituted_frames,
        "rendered_lines":     rendered_lines,
        "original_text":      original_text,
    }


# --- CLI smoke test ----------------------------------------------


if __name__ == "__main__":
    # Verify algebraic equivalence on a simple frame
    test_frame = parse_event("The bear walked to the forest.")
    subs = [
        FrameSubstitution("agent", "bear", "cat"),
        FrameSubstitution("goal",  "forest", "garden"),
    ]
    result = verify_algebra_equivalence(test_frame, subs)
    print(f"=== HDC algebra verification ===")
    print(f"frame:         {test_frame}")
    print(f"subs:          {[(s.role, s.from_value, s.to_value) for s in subs]}")
    print(f"similarity:    {result['similarity']:.4f}")
    print(f"passed (>=.95): {result['passed']}")
    print()

    # Render demo
    print(f"=== Render demo ===")
    print(f"  original:    {test_frame.get('_raw', '')}")
    new_frame = substitute_frame(test_frame, subs)
    print(f"  substituted: {new_frame}")
    print(f"  rendered:    {render_frame(new_frame)}")
