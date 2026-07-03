"""
lattice/multi_hop_qa.py - chain two structured-QA queries via HDC substitution.

A single-hop question fills one unknown slot.  A multi-hop question
embeds a referring expression that itself needs to be resolved before
the outer question can be answered:

    "Where was the COMPOSER of the Brandenburg Concertos born?"
      inner: (?, composed, Brandenburg Concertos) -> Bach
      outer: (Bach, born, ?)                       -> Germany

    "What is the capital of the country where Einstein was born?"
      inner: (Einstein, born, ?)                   -> Germany
      outer: (Germany, capital, ?)                 -> Berlin

The substitution itself is the HDC trick: the inner-query's answer
becomes the SUBJ slot of the outer query, and the outer is then
resolved by the same role-bound slot-wise matcher as before.

This file does:
  1. Pattern-match the multi-hop shape from the surface text.
  2. Build the inner StructuredQA query, run it.
  3. Substitute the resolved entity into the outer query, run it.
  4. Return both hops and the final answer (with confidences).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from lattice.structured_qa import StructuredQA


# noun -> the canonical verb whose OBJECT is X, used to find the
# entity standing in for "the [noun] of X".
_NOUN_TO_VERB = {
    "composer":   "composed",
    "composers":  "composed",
    "inventor":   "invented",
    "inventors":  "invented",
    "writer":     "wrote",
    "writers":    "wrote",
    "author":     "wrote",
    "authors":    "wrote",
    "founder":    "founded",
    "founders":   "founded",
    "discoverer": "discovered",
    "creator":    "developed",
    "creators":   "developed",
    "developer":  "developed",
    "developers": "developed",
    "painter":    "painted",
    "painters":   "painted",
    "designer":   "designed",
    "designers":  "designed",
    "builder":    "built",
    "builders":   "built",
}


# Multi-hop pattern definitions.  Each entry knows how to extract the
# inner sub-query and the outer sub-query from a regex match.
@dataclass
class MultiHopPattern:
    name:    str
    pattern: re.Pattern
    # Functions that take a re.Match and return a (verb, subj, obj, unknown)
    # quadruple for the inner and the outer sub-queries.  The outer uses
    # a placeholder string "<ANSWER>" wherever the inner's answer should
    # be substituted.
    inner: callable
    outer: callable


_PATTERNS: list[MultiHopPattern] = []


def _register(name: str, pattern: str, inner_fn, outer_fn):
    _PATTERNS.append(MultiHopPattern(
        name=name,
        pattern=re.compile(pattern, re.I),
        inner=inner_fn,
        outer=outer_fn,
    ))


# Pattern 1: "What is the capital of the country where X was [verb]?"
#   inner: (X, verb, ?) -> country
#   outer: (country, capital, ?) -> capital
_register(
    "capital_of_country_where_X_verbed",
    r"\bwhat\s+is\s+(?:the\s+)?capital\s+of\s+the\s+country\s+"
    r"(?:where|in\s+which)\s+(?P<X>.+?)\s+(?:was|is)\s+"
    r"(?P<verb>born|located|founded|from)\??$",
    inner_fn=lambda m: {
        "subj": m.group("X").strip(),
        "verb": "born",   # canonicalised for these location verbs
        "obj":  None, "unknown": "OBJ",
    },
    outer_fn=lambda m: {
        "subj": "<ANSWER>", "verb": "capital",
        "obj":  None, "unknown": "OBJ",
    },
)


# Pattern 2: "Where was the [composer/inventor/writer/...] of X [verb]?"
#   inner: (?, role_verb, X)        -> person
#   outer: (person, born/etc, ?)    -> place
_register(
    "where_was_NOUN_of_X_verbed",
    r"\bwhere\s+(?:was|is)\s+(?:the\s+)?(?P<noun>"
    + "|".join(_NOUN_TO_VERB.keys()) +
    r")\s+of\s+(?P<X>.+?)\s+(?P<verb>born|from)\??$",
    inner_fn=lambda m: {
        "subj": None,
        "verb": _NOUN_TO_VERB[m.group("noun").lower()],
        "obj":  m.group("X").strip(),
        "unknown": "SUBJ",
    },
    outer_fn=lambda m: {
        "subj": "<ANSWER>", "verb": "born",
        "obj":  None, "unknown": "OBJ",
    },
)


# Pattern 3: "What did the [NOUN] of X do?"  / "What is the profession of the [NOUN] of X?"
#   inner: (?, role_verb, X) -> person
#   outer: (person, is, ?)   -> profession
_register(
    "what_did_NOUN_of_X_do",
    r"\bwhat\s+(?:did|does|is)\s+(?:the\s+)?(?:profession\s+of\s+)?(?:the\s+)?(?P<noun>"
    + "|".join(_NOUN_TO_VERB.keys()) +
    r")\s+of\s+(?P<X>.+?)\s+do\??$",
    inner_fn=lambda m: {
        "subj": None,
        "verb": _NOUN_TO_VERB[m.group("noun").lower()],
        "obj":  m.group("X").strip(),
        "unknown": "SUBJ",
    },
    outer_fn=lambda m: {
        "subj": "<ANSWER>", "verb": "is",
        "obj":  None, "unknown": "OBJ",
    },
)


# Pattern 4: "Where was the person who Verbed X born?"
#   inner: (?, verb, X) -> person
#   outer: (person, born, ?) -> place
_register(
    "where_was_person_who_verbed_X_born",
    r"\bwhere\s+(?:was|is)\s+(?:the\s+)?person\s+who\s+"
    r"(?P<verb>wrote|composed|invented|discovered|founded|developed|"
    r"created|came\s+up\s+with|painted|made|designed|built|formulated)\s+"
    r"(?:the\s+)?(?P<X>.+?)\s+born\??$",
    inner_fn=lambda m: {
        "subj": None,
        "verb": m.group("verb").lower().split()[0],
        "obj":  m.group("X").strip(),
        "unknown": "SUBJ",
    },
    outer_fn=lambda m: {
        "subj": "<ANSWER>", "verb": "born",
        "obj":  None, "unknown": "OBJ",
    },
)


@dataclass
class MultiHopAnswer:
    final_answer: str
    pattern_name: str
    inner_sim:    float
    inner_answer: str
    outer_sim:    float
    outer_answer: str
    inner_text:   str
    outer_text:   str

    def explain(self) -> str:
        return (
            f"Hop 1 ({self.pattern_name}): {self.inner_answer} "
            f"(sim={self.inner_sim:.2f})\n"
            f"Hop 2: {self.outer_answer} (sim={self.outer_sim:.2f})\n"
            f"Final: {self.final_answer}"
        )


class MultiHopQA:
    """Chains two StructuredQA queries via HDC substitution."""

    def __init__(self, structured: StructuredQA,
                  abstain_threshold: float = 0.45):
        self.structured = structured
        self.abstain_threshold = abstain_threshold

    def answer(self, query: str) -> MultiHopAnswer | None:
        """Try to answer `query` by detecting a multi-hop shape,
        resolving inner then outer.  Returns None if either no
        pattern matches or either hop abstains.
        """
        for shape in _PATTERNS:
            m = shape.pattern.search(query)
            if not m:
                continue

            inner_slots = shape.inner(m)
            outer_slots = shape.outer(m)

            # Resolve the inner query.
            inner_q = self._slots_to_query(inner_slots)
            if inner_q is None:
                return None
            inner_result = self._answer_with_slots(inner_slots)
            if inner_result is None:
                return None
            if inner_result["similarity"] < self.abstain_threshold:
                return None

            answer_word = inner_result["answer_word"]

            # Substitute into the outer query.
            substituted = {}
            for k, v in outer_slots.items():
                substituted[k] = answer_word if v == "<ANSWER>" else v
            outer_result = self._answer_with_slots(substituted)
            if outer_result is None:
                return None
            if outer_result["similarity"] < self.abstain_threshold:
                return None

            return MultiHopAnswer(
                final_answer = outer_result["answer_word"],
                pattern_name = shape.name,
                inner_sim    = inner_result["similarity"],
                inner_answer = inner_result["answer_word"],
                outer_sim    = outer_result["similarity"],
                outer_answer = outer_result["answer_word"],
                inner_text   = inner_result["sentence"],
                outer_text   = outer_result["sentence"],
            )
        return None

    def _slots_to_query(self, slots: dict) -> str | None:
        """Convert a slot-dict into a fake question string just so we
        can reuse the structured QA's `answer` path.  Not strictly
        necessary — we bypass to `_answer_with_slots` instead."""
        return "synthetic"

    def _answer_with_slots(self, slots: dict) -> dict | None:
        """Run structured QA's slot-wise search directly, given a
        (subj, verb, obj, unknown) dict — bypassing the surface-form
        parser.  This lets us re-issue an outer query with the inner's
        resolved entity as the subject.
        """
        sqa = self.structured
        sqa._ensure_stacks()
        if sqa._subj_stack is None:
            return None

        unknown = slots["unknown"]
        q_subj_hv = (sqa._key_token_hv(slots["subj"])
                       if unknown != "SUBJ" and slots.get("subj") else None)
        q_verb_hv = (sqa._word_hv(slots["verb"])
                       if unknown != "VERB" and slots.get("verb") else None)
        q_obj_hv  = (sqa._encode_phrase(slots["obj"])
                       if unknown != "OBJ" and slots.get("obj") else None)

        import numpy as np
        per_slot: list[np.ndarray] = []
        if q_subj_hv is not None:
            d = np.bitwise_xor(sqa._subj_stack, q_subj_hv[None, :]).sum(axis=1)
            per_slot.append(1.0 - 2.0 * d / sqa.dim)
        if q_verb_hv is not None:
            d = np.bitwise_xor(sqa._verb_stack, q_verb_hv[None, :]).sum(axis=1)
            per_slot.append(1.0 - 2.0 * d / sqa.dim)
        if q_obj_hv is not None:
            d = np.bitwise_xor(sqa._obj_stack, q_obj_hv[None, :]).sum(axis=1)
            per_slot.append(1.0 - 2.0 * d / sqa.dim)
        if not per_slot:
            return None
        mat = np.stack(per_slot, axis=0)
        mean = mat.mean(axis=0)
        mn   = mat.min(axis=0)
        scores = mean - 0.5 * (mean - mn)
        best = int(np.argmax(scores))
        sim  = float(scores[best])
        min_sim = float(mn[best])
        if sim < self.abstain_threshold or min_sim < self.abstain_threshold:
            return None
        subj, verb, obj = sqa.claim_triple[best]
        answer = {"SUBJ": subj, "VERB": verb, "OBJ": obj}[unknown]
        return {
            "sentence":   sqa.claim_text[best],
            "source":     sqa.claim_source[best],
            "subj":       subj, "verb": verb, "obj": obj,
            "answer_word":answer,
            "similarity": sim,
            "unknown":    unknown,
        }
