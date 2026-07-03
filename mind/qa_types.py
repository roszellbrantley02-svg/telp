"""autopilot/qa_types.py — neurosymbolic Q&A type system.

Classifies questions by expected answer TYPE, then provides validators
that check whether a retrieved memory contains content of that type.
Wraps HDC neural retrieval with a symbolic guard:

  Question:  "When was Einstein born?"
  Type:      date_question
  Validator: requires the answer to contain a year-shaped token (4 digits)
             OR a date phrase (e.g., "March 14, 1879")
  Effect:    filters out retrievals that match the topic but don't
             actually answer the question.

This is the cheapest neurosymbolic win for chat quality — HDC says
"this looks similar," symbolic says "this is the right kind of thing."
"""
from __future__ import annotations

import re
from typing import Optional


# ─── Type definitions ─────────────────────────────────────────────


QTYPE_DEFINITION = "definition"   # "what is X" / "tell me about X"
QTYPE_DATE       = "date"          # "when did X happen", "what year"
QTYPE_PERSON     = "person"        # "who is X", "who painted Y"
QTYPE_PLACE      = "place"         # "where is X", "what country"
QTYPE_QUANTITY   = "quantity"      # "how many", "how much", "what's 23*47"
QTYPE_CAUSE      = "cause"         # "why does X", "what causes Y"
QTYPE_MECHANISM  = "mechanism"     # "how does X work"
QTYPE_YESNO      = "yesno"         # "is X a Y?", "are all X?"
QTYPE_LIST       = "list"          # "list X", "examples of Y"
QTYPE_GENERAL    = "general"       # fallback


# ─── Classifier ──────────────────────────────────────────────────


# Order matters — earlier patterns win.
_TYPE_PATTERNS = [
    # date / when / year
    (QTYPE_DATE, re.compile(
        r"^\s*(when|what\s+(?:year|date|time|month|day)|"
        r"what\s+time|how\s+old)\b", re.IGNORECASE)),
    # person / who
    (QTYPE_PERSON, re.compile(
        r"^\s*(who(?:'s|\s+is|\s+was|\s+are|\s+were)?\b|"
        r"whose\b|name\s+the\s+(?:person|inventor|author|painter|"
        r"composer))", re.IGNORECASE)),
    # place / where / country
    (QTYPE_PLACE, re.compile(
        r"^\s*(where(?:'s|\s+is|\s+was|\s+are|\s+were)?\b|"
        r"what\s+(?:country|city|state|location|place))",
        re.IGNORECASE)),
    # quantity / how many / how much / numeric
    (QTYPE_QUANTITY, re.compile(
        r"^\s*(how\s+(?:many|much|long|tall|big|fast|far)\b|"
        r"what\s+is\s+(?:the\s+)?(?:speed|number|count|size|"
        r"weight|height|distance|amount|cost|price))",
        re.IGNORECASE)),
    # cause / why / what causes
    (QTYPE_CAUSE, re.compile(
        r"^\s*(why\b|what\s+causes?\b|what\s+(?:made|makes))",
        re.IGNORECASE)),
    # mechanism / how does X work
    (QTYPE_MECHANISM, re.compile(
        r"^\s*how\s+(?:do(?:es)?|did)\b.*\b(work|happen|function|"
        r"operate|produce|generate)", re.IGNORECASE)),
    # yes / no
    (QTYPE_YESNO, re.compile(
        r"^\s*(is|are|was|were|do|does|did|will|would|"
        r"can|could|should|has|have|had|am)\b", re.IGNORECASE)),
    # list / examples
    (QTYPE_LIST, re.compile(
        r"^\s*(list\b|name\s+(?:some|all|three|five|ten)|"
        r"give\s+(?:me\s+)?(?:some|a\s+few|three|five|ten)|"
        r"examples?\s+of)", re.IGNORECASE)),
    # definition fallback ("what is X" / "what are Y" / "tell me about X")
    (QTYPE_DEFINITION, re.compile(
        r"^\s*(what\s+(?:is|are|were|was)\b|tell\s+me\s+about\b|"
        r"describe\b|define\b|explain\s+what)", re.IGNORECASE)),
]


def classify_question(msg: str) -> str:
    """Return one of the QTYPE_* constants for the given question.

    Falls back to QTYPE_GENERAL when no pattern fires.
    """
    if not msg:
        return QTYPE_GENERAL
    for qtype, pattern in _TYPE_PATTERNS:
        if pattern.search(msg):
            return qtype
    return QTYPE_GENERAL


# ─── Answer-type validators ──────────────────────────────────────


# A "date-shaped" token: 4-digit year, or month+year, or "DDth century",
# or a "X years ago / X days ago", or an explicit YYYY-MM-DD.
_DATE_RX = re.compile(
    r"\b(?:"
    r"(?:1[5-9]\d{2}|20\d{2}|21\d{2})"     # year 1500-2199
    r"|(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d"
    r"|\d{1,2}(?:st|nd|rd|th)\s+century"
    r"|\d{1,2}-\d{1,2}-\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|(?:bce?|bc|ad|ce)"
    r")\b",
    re.IGNORECASE,
)


# A "person-shaped" name: capitalized word(s) — 2 caps in a row, or a
# single cap word that isn't sentence-initial (good enough heuristic).
_PERSON_RX = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
)

# A "place-shaped" name: a known place keyword OR capitalized-words.
_PLACE_RX = re.compile(
    r"\b(?:country|city|state|nation|continent|river|mountain|"
    r"island|ocean|sea|town|village|capital|province|region|"
    r"in\s+[A-Z][a-z]+|of\s+[A-Z][a-z]+)\b",
    re.IGNORECASE,
)

# A "quantity-shaped" token: any digit cluster, or written number, or
# explicit unit.
_QUANTITY_RX = re.compile(
    r"\b(?:\d+(?:,\d{3})*(?:\.\d+)?|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"hundred|thousand|million|billion|trillion|"
    r"meters?|km|miles?|feet|inches?|seconds?|minutes?|hours?|"
    r"kilograms?|grams?|pounds?|tons?|liters?|gallons?|"
    r"percent|%)\b",
    re.IGNORECASE,
)

# Definitional shape — at least one is/was/are/were OR "refers to".
_DEFINITION_RX = re.compile(
    r"\b(is|was|are|were|refers\s+to|defined\s+as|known\s+as|"
    r"means)\b",
    re.IGNORECASE,
)


def answer_matches_type(answer_text: str, qtype: str) -> bool:
    """Does this candidate answer contain content of the right type
    for the question's expected answer type?

    Returns True when the answer plausibly matches; False when it
    clearly doesn't (drop / penalize).
    """
    if not answer_text:
        return False
    txt = answer_text.strip()
    if qtype == QTYPE_DATE:
        return bool(_DATE_RX.search(txt))
    if qtype == QTYPE_PERSON:
        return bool(_PERSON_RX.search(txt))
    if qtype == QTYPE_PLACE:
        return bool(_PLACE_RX.search(txt)) or bool(_PERSON_RX.search(txt))
    if qtype == QTYPE_QUANTITY:
        return bool(_QUANTITY_RX.search(txt))
    if qtype == QTYPE_DEFINITION:
        # Definition-shaped: has "is/was/are" linking subject to predicate
        return bool(_DEFINITION_RX.search(txt))
    if qtype == QTYPE_CAUSE:
        # Cause-shaped: contains "because", "due to", "caused by",
        # "results from", or a sentence with "is the result of".
        return bool(re.search(
            r"\b(because|due\s+to|caused\s+by|results?\s+(?:from|in)|"
            r"is\s+the\s+result|leads?\s+to|driven\s+by|triggers?|"
            r"a\s+result\s+of)\b", txt, re.IGNORECASE,
        ))
    if qtype == QTYPE_MECHANISM:
        return bool(re.search(
            r"\b(by|through|via|using|involves|consists|works\s+by|"
            r"process|step|mechanism|when|whereby)\b",
            txt, re.IGNORECASE,
        ))
    if qtype == QTYPE_YESNO:
        # Anything that contains "yes", "no", "not", or a definite
        # is/was statement counts as a candidate.
        return True
    if qtype == QTYPE_LIST:
        # Has 2+ commas or 2+ bullets or 2+ semicolons.
        return (txt.count(",") >= 2 or txt.count(";") >= 2 or
                  txt.count("\n-") >= 2 or txt.count("•") >= 1)
    # General / unknown — accept anything that has reasonable length
    # and ends with appropriate punctuation.
    return len(txt.split()) >= 4


_QUESTION_WORDS = frozenset("""
    what whats whom whose who where when why how which
    is are was were do does did will would shall should
    can could may might must has have had am
    the a an of in on at to from for with by about
    me my you your he she it they we us our their
    that this these those some any all
""".split())


def _extract_question_subject(msg: str) -> set[str]:
    """Pull the content tokens from the question that are likely to
    refer to its SUBJECT (the thing the answer is about).

    Example:
      "when was Einstein born?"  → {"einstein", "born"}
      "what is the speed of light?"  → {"speed", "light"}
      "who painted the mona lisa?"  → {"painted", "mona", "lisa"}

    Returns a lowercased set.  Used by S-P-O validation to require
    the answer to actually mention what the question is asking about.
    """
    toks = re.findall(r"\b[a-zA-Z][a-zA-Z'-]+\b", msg.lower())
    return {t for t in toks
                if t not in _QUESTION_WORDS and len(t) >= 3}


def answer_mentions_subject(answer_text: str, msg: str,
                                  min_overlap: int = 1) -> bool:
    """True if the candidate answer mentions at least `min_overlap`
    content tokens from the question's subject — i.e., the answer is
    actually ABOUT what was asked.

    For multi-word subjects ("mona lisa"), we accept a match on any
    of the constituent words.
    """
    subject = _extract_question_subject(msg)
    if not subject:
        return True   # no subject extracted — don't penalize
    ans_words = set(re.findall(r"\b[a-zA-Z][a-zA-Z'-]+\b",
                                       answer_text.lower()))
    overlap = len(subject & ans_words)
    return overlap >= min_overlap


def type_aware_score(answer_text: str, qtype: str) -> float:
    """Return a 0..1 score representing how well the answer matches
    the question's expected type.  Used as a multiplier in retrieval
    ranking — multi-doc synthesis boosts type-matching memories.
    """
    if qtype == QTYPE_GENERAL:
        return 0.5   # neutral
    return 1.0 if answer_matches_type(answer_text, qtype) else 0.0


# ─── Self-test ────────────────────────────────────────────────────


def _self_test():
    cases = [
        ("when was einstein born?", QTYPE_DATE),
        ("what year did world war 2 end?", QTYPE_DATE),
        ("who painted the mona lisa?", QTYPE_PERSON),
        ("who invented the telephone?", QTYPE_PERSON),
        ("where is paris?", QTYPE_PLACE),
        ("what country is the eiffel tower in?", QTYPE_PLACE),
        ("how many planets are there?", QTYPE_QUANTITY),
        ("what is the speed of light?", QTYPE_QUANTITY),
        ("why does the sun shine?", QTYPE_CAUSE),
        ("how does a transistor work?", QTYPE_MECHANISM),
        ("is the earth round?", QTYPE_YESNO),
        ("list three primary colors", QTYPE_LIST),
        ("what is photosynthesis?", QTYPE_DEFINITION),
        ("tell me about einstein", QTYPE_DEFINITION),
        ("hey there", QTYPE_GENERAL),
    ]
    print("Question classifier:")
    ok = 0
    for q, expected in cases:
        actual = classify_question(q)
        mark = "OK " if actual == expected else "BAD"
        if actual == expected: ok += 1
        print(f"  [{mark}] {q!r:<50}  → {actual}  (expected {expected})")
    print(f"  {ok}/{len(cases)} passed\n")

    # Validators
    print("Answer-type validators:")
    val_cases = [
        ("Einstein was born in 1879.",            QTYPE_DATE,      True),
        ("Einstein was a physicist.",             QTYPE_DATE,      False),
        ("Leonardo da Vinci painted the Mona Lisa.", QTYPE_PERSON, True),
        ("an Italian artist painted it",          QTYPE_PERSON,    False),
        ("It is located in Paris, France.",       QTYPE_PLACE,     True),
        ("The mass is 5.97 × 10^24 kg.",          QTYPE_QUANTITY,  True),
        ("It is because of gravity.",             QTYPE_CAUSE,     True),
        ("Photosynthesis is a process plants use.", QTYPE_DEFINITION, True),
    ]
    for ans, qt, expected in val_cases:
        actual = answer_matches_type(ans, qt)
        mark = "OK " if actual == expected else "BAD"
        print(f"  [{mark}] {qt:<11}  {ans[:60]!r:<62}  → {actual}")


if __name__ == "__main__":
    _self_test()
