"""
lattice/structured_qa.py - role-bound HDC Q&A with native abstention.

The architectural idea
----------------------
Most retrieval treats a question as a bag of words.  But a question
has STRUCTURE: who, what, where, when, why.  The user is naming some
slots ("created", "relativity") and asking us to fill another ("Who?").

In Vector Symbolic Architecture (VSA) terms, every claim is a
role-filler bundle:

    claim_HV = bundle(
        ROLE_SUBJ  ⊗ subject_HV,
        ROLE_VERB  ⊗ verb_HV,
        ROLE_OBJ   ⊗ object_HV,
    )

A question fills the same shape but binds an UNKNOWN hypervector to
the slot the user wants filled:

    question_HV = bundle(
        ROLE_SUBJ  ⊗ UNKNOWN_HV,      # "who"
        ROLE_VERB  ⊗ developed_HV,
        ROLE_OBJ   ⊗ relativity_HV,
    )

The nearest stored claim by Hamming distance is the best answer.  Its
value at the UNKNOWN slot is the specific entity to return.

Why this beats the heuristic stack
-----------------------------------
1. Synonym robustness comes free.  "developed" and "created" share
   context in the user's corpus, so their RI vectors overlap.  Bound
   to the same ROLE_VERB, they project to similar bits.  Active and
   passive forms of the same fact ("Einstein developed relativity"
   vs "Relativity was developed by Einstein") map to the same triple
   and the same HV.

2. Paraphrase robustness comes free.  Once the question parses to
   (?, verb, object), the rest is algebra.  No regex over the
   question's surface form.

3. Native abstention.  The Hamming similarity of the best match IS
   the confidence.  Below threshold -> "I don't know."  Compare
   this to bag-of-words retrieval which always returns SOMETHING.

References:
    Plate (1995) - Holographic Reduced Representations
    Kanerva (2009) - Hyperdimensional computing: An introduction
    Rachkovskij (2001) - Representation and processing of structures
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.standalone_encoder import CorpusRIEncoder


# ─── Verb dictionaries ───────────────────────────────────────────


# Verbs that we know how to look for.  Mapped to a *canonical* form
# so synonyms get the same lookup key.  The RI encoder also handles
# similarity across these via context co-occurrence — this map just
# helps when the user types a verb that didn't appear in the corpus
# but a synonym did.
_VERB_CANONICAL = {
    "developed":    "developed",  "developing": "developed",
    "created":      "developed",  "creating":   "developed",
    "came":         "developed",  "formulated": "developed",
    "established":  "founded",    "founded":    "founded",
    "wrote":        "wrote",      "written":    "wrote",  "writing": "wrote",
    "composed":     "composed",   "composing":  "composed",
    "invented":     "invented",   "inventing":  "invented",
    "discovered":   "discovered", "discovering": "discovered",
    "founded":      "founded",
    "painted":      "painted",    "painting":   "painted",
    "made":         "made",       "making":     "made",
    "designed":     "designed",   "designing":  "designed",
    "built":        "built",      "building":   "built",
    "born":         "born",
    "born_year":    "born_year",
    "died":         "died",
    "died_year":    "died_year",
    "ruled":        "ruled",      "ruling":     "ruled",
    "led":          "led",        "leading":    "led",
    "is":           "is",   "was": "is", "were": "is", "are": "is", "be": "is",
}


# Common stop words to exclude when sniffing for subject/object.
_STOP = {
    "a","an","the","of","in","on","at","by","to","from","with","for",
    "and","or","but","is","was","were","are","be","been","being",
    "this","that","these","those","it","its","they","their","them",
    "he","she","his","her","i","you","we","us","our","my","your",
    "as","than","then","also","both","such","not","no","yes",
}


_TOKEN_RE = re.compile(r"\b[\w-]+\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-z][\w-]*(?:\s+[A-Z][a-z][\w-]*)*\b")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _canon_verb(word: str) -> str | None:
    """Return canonical verb form, or None if the word isn't a known verb."""
    return _VERB_CANONICAL.get(word.lower())


# ─── Triple extraction ──────────────────────────────────────────


# ─── Pattern-based extractors ───────────────────────────────────


# Each extractor takes the raw sentence and a list of capitalised
# phrases, and yields zero or more (subj, verb, obj) claims.  Verbs
# are stored in their canonical form so questions and claims share
# the same VERB key.


_CAPITAL_OF = re.compile(
    r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+is\s+(?:the\s+)?(?:nation'?s\s+)?capital"
    r"(?:\s+(?:city|and\s+\w+\s+city|of\s+([A-Z][\w-]+)))?",
    re.I)

_X_IS_CAPITAL_OF_Y = re.compile(
    r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+is\s+(?:the\s+)?capital\s+(?:and\s+\w+\s+\w+\s+)?(?:of\s+)?([A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)",
    re.I)

_Y_CAPITAL_IS_X = re.compile(
    r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)'s\s+capital\s+(?:and\s+\w+\s+\w+\s+)?(?:city\s+)?is\s+([A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)",
    re.I)

_X_BORN = re.compile(
    r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,3})\s+was\s+born\s+in\s+"
    r"(\d{3,4}|[A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,3})",
    re.I)


# Parenthetical date pattern - Wikipedia standard for biographical
# articles: "Name (DD Month YYYY - DD Month YYYY)" or just "(YYYY - YYYY)".
# We extract both born_year and died_year when both are present, or
# just born_year when there's a single year.
_PAREN_DATES = re.compile(
    r"((?-i:[A-Z])[\w'-]+(?:\s+(?-i:[A-Z])[\w'-]+){0,4})"
    r"\s*\(\s*c?\.?\s*"
    r"(?:\d{1,2}\s+\w+\s+)?"
    r"(?P<born>\d{3,4})"
    r"(?:\s*[-–—]\s*(?:\d{1,2}\s+\w+\s+)?(?P<died>\d{3,4}))?"
    r"\s*\)",
)


# Sentence-level "X was born in <year>" / "X (1564-1616) was..."
_BORN_YEAR_INLINE = re.compile(
    r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,3})\s+was\s+born\s+(?:on\s+)?"
    r"(?:\d{1,2}\s+\w+\s+)?(\d{3,4})\b",
    re.I)


# Build the nationality-list regex once for reuse.
_NATIONALITY_LIST = (
    r"German|French|British|English|Scottish|Irish|Welsh|Italian|Spanish|"
    r"Portuguese|Dutch|Belgian|Swiss|Austrian|Polish|Russian|Greek|Turkish|"
    r"Egyptian|Chinese|Japanese|Korean|Indian|Pakistani|Australian|"
    r"American|Canadian|Mexican|Brazilian|Argentine|Argentinian|Czech|"
    r"Danish|Norwegian|Swedish|Finnish|Hungarian|Romanian|Bulgarian|"
    r"Croatian|Serbian|Slovak|Slovenian|Ukrainian|Israeli|Iranian|Iraqi|"
    r"Saudi|Lebanese|Syrian|Moroccan|Algerian|Tunisian|Libyan|Kenyan|"
    r"Nigerian|Ghanaian|Ethiopian|Somali|Vietnamese|Thai|Indonesian|"
    r"Filipino|Malaysian|Singaporean|Cuban|Colombian|Venezuelan|Peruvian|"
    r"Chilean|Bolivian|Ecuadorian|Uruguayan|Paraguayan"
)

# Nationality-born: "X was a German-born ..." -> (X, born, Germany).
_NATIONALITY_BORN = re.compile(
    r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,3})\s+(?:was|is)\s+(?:a|an|the)\s+"
    r"(?:[\w-]+\s+){0,2}"
    r"(" + _NATIONALITY_LIST + r")"
    r"[-\s]born\b",
    re.I)
# _NATIONALITY_PROFESSION is built lazily after _PROFESSION_LIST is
# defined (further down).  See the assignment below the profession list.
_NATIONALITY_PROFESSION = None


# Map nationality adjective -> country name.  Used to convert
# nationality-born matches into (X, born, Country) claims.
_NATIONALITY_TO_COUNTRY = {
    "german":     "Germany",
    "french":     "France",
    "british":    "United Kingdom",
    "english":    "England",
    "scottish":   "Scotland",
    "irish":      "Ireland",
    "welsh":      "Wales",
    "italian":    "Italy",
    "spanish":    "Spain",
    "portuguese": "Portugal",
    "dutch":      "Netherlands",
    "belgian":    "Belgium",
    "swiss":      "Switzerland",
    "austrian":   "Austria",
    "polish":     "Poland",
    "russian":    "Russia",
    "greek":      "Greece",
    "turkish":    "Turkey",
    "egyptian":   "Egypt",
    "chinese":    "China",
    "japanese":   "Japan",
    "korean":     "Korea",
    "indian":     "India",
    "pakistani":  "Pakistan",
    "australian": "Australia",
    "american":   "United States",
    "canadian":   "Canada",
    "mexican":    "Mexico",
    "brazilian":  "Brazil",
    "argentine":  "Argentina",
    "argentinian":"Argentina",
    "czech":      "Czech Republic",
    "danish":     "Denmark",
    "norwegian":  "Norway",
    "swedish":    "Sweden",
    "finnish":    "Finland",
    "hungarian":  "Hungary",
    "romanian":   "Romania",
    "bulgarian":  "Bulgaria",
    "croatian":   "Croatia",
    "serbian":    "Serbia",
    "slovak":     "Slovakia",
    "slovenian":  "Slovenia",
    "ukrainian":  "Ukraine",
    "israeli":    "Israel",
    "iranian":    "Iran",
    "iraqi":      "Iraq",
    "saudi":      "Saudi Arabia",
    "lebanese":   "Lebanon",
    "syrian":     "Syria",
    "moroccan":   "Morocco",
    "algerian":   "Algeria",
    "tunisian":   "Tunisia",
    "libyan":     "Libya",
    "kenyan":     "Kenya",
    "nigerian":   "Nigeria",
    "ghanaian":   "Ghana",
    "ethiopian":  "Ethiopia",
    "somali":     "Somalia",
    "vietnamese": "Vietnam",
    "thai":       "Thailand",
    "indonesian": "Indonesia",
    "filipino":   "Philippines",
    "malaysian":  "Malaysia",
    "singaporean":"Singapore",
    "cuban":      "Cuba",
    "colombian":  "Colombia",
    "venezuelan": "Venezuela",
    "peruvian":   "Peru",
    "chilean":    "Chile",
    "bolivian":   "Bolivia",
    "ecuadorian": "Ecuador",
    "uruguayan":  "Uruguay",
    "paraguayan": "Paraguay",
}

_X_VERB_Y = re.compile(
    r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,3})\s+"
    r"(wrote|composed|invented|discovered|founded|established|"
    r"developed|created|painted|made|designed|built|formulated)\s+"
    r"(?:the\s+|a\s+|an\s+)?([A-Za-z][\w-]+(?:\s+[A-Z][\w-]+){0,4})",
    re.I)

_X_WAS_VERBED_BY_Y = re.compile(
    r"\b(?:the\s+)?([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,3})\s+"
    r"(?:was|were)\s+"
    r"(written|composed|invented|discovered|founded|established|"
    r"developed|created|painted|made|designed|built)\s+by\s+"
    r"([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,3})",
    re.I)

_X_KNOWN_FOR_VERBING_Y = re.compile(
    r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,3})\s+(?:was|is)\s+"
    r"(?:[\w\-,]+\s+){0,8}"
    r"(?:known|famous)\s+for\s+"
    r"(?:developing|creating|writing|composing|inventing|discovering|"
    r"founding|painting|making|designing|building|formulating)\s+"
    r"(?:the\s+)?([A-Za-z][\w-]+(?:\s+[A-Z][\w-]+){0,4})",
    re.I)

# Subject pattern that explicitly requires the *first* token to be
# capitalised AND not a stop word (so "an English polymath who" is
# rejected as a subject candidate).
_PROFESSION_LIST = (
    r"physicist|chemist|mathematician|astronomer|scientist|biologist|"
    r"naturalist|geologist|philosopher|writer|novelist|poet|playwright|"
    r"author|journalist|composer|musician|singer|guitarist|pianist|"
    r"artist|painter|sculptor|architect|engineer|inventor|explorer|"
    r"general|president|emperor|king|queen|leader|politician|cosmologist|"
    r"astrophysicist|crystallographer|astrobiologist|naturalist|conductor"
)

_X_IS_PROFESSION = re.compile(
    # Subject must START with a capital, and we DON'T allow {who, that,
    # which, where, an, the, a, English, French, ...} to leak in.
    # Each subj-token must start with capital and not be a function
    # word like "Who"/"That".
    r"(?<!\w)"
    r"(?P<subj>[A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,4})"
    r"\s+(?:was|is)\s+(?:a|an|the)\s+"
    r"(?:[\w-]+\s+){0,3}"
    r"(?P<prof>" + _PROFESSION_LIST + r")",
    re.I)


# Nationality + profession: "X was a German composer" -> (X, born, Germany).
# Also tolerates an intervening comma-clause: "X, better known as Y, was a
# Polish physicist" - regex allows `, ... ,` between subject and "was".
_NATIONALITY_PROFESSION = re.compile(
    r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,4})"
    r"(?:,\s+[^,.]{0,80}?,)?"        # optional ", parenthetical clause,"
    r"\s+(?:was|is)\s+(?:a|an|the)\s+"
    r"(?:[\w-]+\s+){0,2}"
    r"(" + _NATIONALITY_LIST + r")\s+"
    r"(?:and\s+[\w-]+\s+)?"
    r"(?:" + _PROFESSION_LIST + r")",
    re.I)


_TOKYO_CAPITAL = re.compile(
    r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,2})\s+is\s+the\s+"
    r"(?:nation'?s\s+|country'?s\s+)?capital\b",
    re.I)


# "the capital X" / "the capital, X" / "Its capital is X" — captures
# the capital as a noun-phrase head followed by the city.  Combined
# with article_topic context, this yields (article_topic, capital, X).
# Note: the inner [A-Z] is wrapped in (?-i:...) so case-insensitivity
# doesn't make it match lowercase tokens like "and largest city".
_CAPITAL_NOUN_X = re.compile(
    r"\bthe\s+capital\s+(?:,\s+|is\s+)?"
    r"((?-i:[A-Z])[\w-]+(?:\s+(?-i:[A-Z])[\w-]+){0,2})",
    re.I)
_ITS_CAPITAL_IS = re.compile(
    r"\bits\s+capital\s+(?:and\s+\w+\s+(?:and\s+\w+\s+)?\w+\s+)?"
    r"(?:city\s+)?(?:is\s+)"
    r"((?-i:[A-Z])[\w-]+(?:\s+(?-i:[A-Z])[\w-]+){0,2})",
    re.I)
# Capital after "and largest city is" pattern (Wikipedia bio standard).
_CAPITAL_AND_LARGEST = re.compile(
    r"\b(?:the\s+|its\s+)?capital\s+and\s+(?:largest|most\s+populous)\s+city\s+is\s+"
    r"((?-i:[A-Z])[\w-]+(?:\s+(?-i:[A-Z])[\w-]+){0,2})",
    re.I)


_BAD_SUBJ_PREFIX = {
    "who", "that", "which", "where", "an", "the", "a", "his", "her",
    "their", "this", "those", "these", "what", "such", "some",
}


def _good_subj(s: str) -> bool:
    """Filter clearly-bad subject candidates."""
    if not s:
        return False
    first = s.split()[0].lower()
    if first in _BAD_SUBJ_PREFIX:
        return False
    if not s[0].isupper():
        return False
    return True


# Negation markers — when one appears in a clause we DON'T extract
# a positive claim from that clause.  Better to miss a fact than to
# store its inverse (which would actively poison the QA store).
_NEGATION_PATTERN = re.compile(
    r"\b(?:not|never|no\s+longer|wasn'?t|weren'?t|isn'?t|aren'?t|"
    r"didn'?t|doesn'?t|don'?t|hadn'?t|hasn'?t|haven'?t|"
    r"won'?t|wouldn'?t|cannot|can'?t|couldn'?t|shouldn'?t|"
    r"none\s+of|neither|nor)\b",
    re.I,
)


def _is_negated(sentence: str) -> bool:
    """True if the sentence contains a negation marker.

    We're deliberately strict: ANY negation in the sentence skips
    extraction.  Sub-clause-level negation analysis would be more
    accurate ("X was an Italian composer who never visited France"
    contains "never" but the core claim about Italian is still
    true) — but the false-positive cost is low (we miss a fact),
    while the false-negative cost of NOT detecting negation is
    high (we store an inverted fact that poisons answers).
    """
    return bool(_NEGATION_PATTERN.search(sentence))


def extract_claims(sentence: str,
                     article_topic: str | None = None
                     ) -> list[tuple[str, str, str]]:
    """Extract (subject, verb, object) claims using pattern-specific
    matchers.  Each matcher targets one common biographical/factual
    construction.

    `article_topic` is an optional context entity (e.g. "Japan" when
    parsing a sentence from the Japan article).  It helps resolve
    "the country's capital is X" claims that lack an explicit
    country name.
    """
    claims: list[tuple[str, str, str]] = []
    text = sentence.strip()
    if not text:
        return claims
    # Skip extraction on negated sentences — better no claim than
    # an inverted one.
    if _is_negated(text):
        return claims

    def add(c):
        s, v, o = c
        if _good_subj(s) and o and len(o) >= 2:
            claims.append((s.strip(), v, o.strip()))

    # X is the capital of Y -> (Y, capital, X)
    for m in _X_IS_CAPITAL_OF_Y.finditer(text):
        add((m.group(2), "capital", m.group(1)))

    # Y's capital is X -> (Y, capital, X)
    for m in _Y_CAPITAL_IS_X.finditer(text):
        add((m.group(1), "capital", m.group(2)))

    # "X is the (nation's|country's) capital" with an article topic
    # context -> (article_topic, capital, X).  Catches "Tokyo is the
    # country's capital" inside the Japan article.
    if article_topic:
        for m in _TOKYO_CAPITAL.finditer(text):
            add((article_topic, "capital", m.group(1)))
        # "the capital Vienna" / "the capital, Vienna" / "Its capital is Vienna"
        # All assume the article topic is the country being described.
        for m in _CAPITAL_NOUN_X.finditer(text):
            add((article_topic, "capital", m.group(1)))
        for m in _ITS_CAPITAL_IS.finditer(text):
            add((article_topic, "capital", m.group(1)))
        for m in _CAPITAL_AND_LARGEST.finditer(text):
            add((article_topic, "capital", m.group(1)))

    # X was born in Y -> (X, born, Y)
    for m in _X_BORN.finditer(text):
        # If Y is purely a year (3-4 digits), tag it as born_year
        # instead so date questions can find it.
        y = m.group(2).strip()
        verb = "born_year" if y.isdigit() else "born"
        add((m.group(1), verb, y))

    # Parenthetical date: "X (1879 - 1955)" -> (X, born_year, 1879)
    # and (X, died_year, 1955)
    for m in _PAREN_DATES.finditer(text):
        name = m.group(1).strip()
        if m.group("born"):
            add((name, "born_year", m.group("born")))
        if m.group("died"):
            add((name, "died_year", m.group("died")))

    # "X was born on D Month YYYY" / "X was born YYYY"
    for m in _BORN_YEAR_INLINE.finditer(text):
        add((m.group(1), "born_year", m.group(2)))

    # X was a German-born ... -> (X, born, Germany)
    for m in _NATIONALITY_BORN.finditer(text):
        nationality = m.group(2).lower()
        country = _NATIONALITY_TO_COUNTRY.get(nationality)
        if country:
            add((m.group(1), "born", country))

    # X was a German composer -> (X, born, Germany).  Plain nationality
    # before profession is the dominant Wikipedia opener and indicates
    # country of origin for multi-hop chaining purposes.
    for m in _NATIONALITY_PROFESSION.finditer(text):
        nationality = m.group(2).lower()
        country = _NATIONALITY_TO_COUNTRY.get(nationality)
        if country:
            add((m.group(1), "born", country))

    # X Verbed Y -> (X, verb, Y)
    for m in _X_VERB_Y.finditer(text):
        verb = _canon_verb(m.group(2)) or m.group(2).lower()
        add((m.group(1), verb, m.group(3)))

    # X was Verbed by Y -> (Y, verb, X) (passive normalized)
    for m in _X_WAS_VERBED_BY_Y.finditer(text):
        verb = _canon_verb(m.group(2)) or m.group(2).lower()
        add((m.group(3), verb, m.group(1)))

    # X is known for Verbing Y -> (X, verb, Y)
    for m in _X_KNOWN_FOR_VERBING_Y.finditer(text):
        gerund = re.search(
            r"\b(developing|creating|writing|composing|inventing|"
            r"discovering|founding|painting|making|designing|building|"
            r"formulating)\b", m.group(0), re.I)
        if gerund:
            verb = _canon_verb(gerund.group(1)) or gerund.group(1).lower()
        else:
            verb = "developed"
        add((m.group(1), verb, m.group(2)))

    # X is/was a (modifiers) profession -> (X, is, profession)
    for m in _X_IS_PROFESSION.finditer(text):
        add((m.group("subj"), "is", m.group("prof").lower()))

    # Dedup while preserving order.
    seen = set()
    deduped = []
    for c in claims:
        if c in seen:
            continue
        seen.add(c)
        deduped.append(c)
    return deduped


# ─── Question parsing ───────────────────────────────────────────


# Question shapes.  Each pattern declares which slot is unknown and
# how to extract the other two slots from the regex groups.
_QUESTION_SHAPES = [
    # "Who Verbed X?"  / "Who has Verbed X?"  -> subject unknown
    {
        "pattern": re.compile(
            r"\bwho\s+(?:has\s+|had\s+)?"
            r"(?P<verb>wrote|written|composed|invented|discovered|"
            r"founded|established|developed|created|came\s+up\s+with|"
            r"formulated|painted|made|designed|built)\s+"
            r"(?:the\s+|a\s+|an\s+)?(?P<obj>.+?)\??$",
            re.I),
        "unknown": "SUBJ",
        "verb_from": "verb",
        "obj_from":  "obj",
    },
    # "X was Verbed by whom?"  -> subject unknown (passive)
    {
        "pattern": re.compile(
            r"\b(?:the\s+|a\s+|an\s+)?(?P<obj>.+?)\s+"
            r"(?:was|were)\s+"
            r"(?P<verb>wrote|written|composed|invented|discovered|"
            r"founded|established|developed|created|formulated|painted|"
            r"made|designed|built)\s+by\s+whom\b",
            re.I),
        "unknown": "SUBJ",
        "verb_from": "verb",
        "obj_from":  "obj",
    },
    # "X was Verbed by ..."  -> subject unknown (passive with trailing entity)
    # we still treat the by-clause as the answer to query
    # "What/Where is X's Y?" / "What is the Y of X?"
    {
        "pattern": re.compile(
            r"\bwhat\s+is\s+(?:the\s+)?(?P<verb>capital|founder|inventor|"
            r"creator|composer|author|leader|king|queen|president)"
            r"\s+of\s+(?:the\s+)?(?P<subj>.+?)\??$",
            re.I),
        "unknown": "OBJ",
        "subj_from": "subj",
        "verb_from": "verb",
    },
    # "X's capital?" / "What's X's capital?"
    {
        "pattern": re.compile(
            r"(?:what'?s?\s+|whats\s+)?"
            r"(?P<subj>[\w-]+)'s\s+"
            r"(?P<verb>capital|founder|inventor|leader|king|queen)\b",
            re.I),
        "unknown": "OBJ",
        "subj_from": "subj",
        "verb_from": "verb",
    },
    # "Where was X born?"
    {
        "pattern": re.compile(
            r"\bwhere\s+(?:was|is)\s+(?P<subj>.+?)\s+"
            r"(?P<verb>born|located|from|founded)\??$",
            re.I),
        "unknown": "OBJ",
        "subj_from": "subj",
        "verb_from": "verb",
    },
    # "When was X born?" -> we want a YEAR, so use the *_year verb
    {
        "pattern": re.compile(
            r"\bwhen\s+(?:was|were)\s+(?P<subj>.+?)\s+born\??$",
            re.I),
        "unknown": "OBJ",
        "subj_from": "subj",
        "verb_from": "_constant:born_year",
    },
    # "When did X die?" -> we want a YEAR
    {
        "pattern": re.compile(
            r"\bwhen\s+did\s+(?P<subj>.+?)\s+die\??$",
            re.I),
        "unknown": "OBJ",
        "subj_from": "subj",
        "verb_from": "_constant:died_year",
    },
    # Legacy "When was X founded/etc?" - keeps existing behaviour
    {
        "pattern": re.compile(
            r"\bwhen\s+(?:was|did)\s+(?P<subj>.+?)\s+"
            r"(?P<verb>founded|created|invented|discovered)\b",
            re.I),
        "unknown": "OBJ",
        "subj_from": "subj",
        "verb_from": "verb",
    },
]


def parse_question(query: str) -> dict | None:
    """Parse a question into (subj, verb, obj) plus the unknown role.

    Returns None if no pattern matches — the agent should fall back
    to bag-of-words retrieval for such queries.
    """
    for shape in _QUESTION_SHAPES:
        m = shape["pattern"].search(query)
        if not m:
            continue
        out = {"unknown": shape["unknown"], "subj": None,
                  "verb": None, "obj": None}
        for slot, group_key in [("subj", "subj_from"),
                                 ("verb", "verb_from"),
                                 ("obj",  "obj_from")]:
            if group_key in shape:
                ref = shape[group_key]
                # "_constant:foo" means use the literal "foo" for this
                # slot rather than reading a regex group.  Used when a
                # question doesn't textually contain the verb (e.g.
                # "When was X born?" -> verb="born_year").
                if ref.startswith("_constant:"):
                    out[slot] = ref.split(":", 1)[1]
                    continue
                value = m.group(ref)
                # Strip trailing punctuation / articles.
                value = re.sub(r"[?\.]+\s*$", "", value).strip()
                value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)
                out[slot] = value
        # Canonicalise verb.
        if out["verb"]:
            cv = _canon_verb(out["verb"].split()[0])
            if cv:
                out["verb"] = cv
        return out
    return None


# ─── Role-bound HDC Q&A ─────────────────────────────────────────


class StructuredQA:
    """Q&A over role-bound HDC claims with slot-wise matching and
    native abstention.

    Unlike a single-bundle approach, this keeps each slot's HV
    separate per claim.  At query time, we score each claim by
    summing the cosine similarity of every KNOWN slot — the
    unknown slot doesn't contribute, so it can't drag the wrong
    claim to the top.

    The advantage of HDC here: slot HVs come from the CorpusRIEncoder
    which means "composed" / "wrote" / "developed" are close in the
    binding space because they share context in the user's corpus.
    Synonym robustness is a side-effect of the geometry, not a
    hand-curated synonym table.
    """

    def __init__(self, encoder: CorpusRIEncoder, seed: int = 7,
                  abstain_threshold: float = 0.10):
        self.encoder = encoder
        self.dim = encoder.dim
        rng = np.random.default_rng(seed)
        # Role hypervectors retained for diagnostic / future use;
        # the slot-wise matcher below doesn't bind them.
        self.ROLE_SUBJ    = rng.integers(0, 2, size=self.dim, dtype=np.int8)
        self.ROLE_VERB    = rng.integers(0, 2, size=self.dim, dtype=np.int8)
        self.ROLE_OBJ     = rng.integers(0, 2, size=self.dim, dtype=np.int8)
        self.ROLE_UNKNOWN = rng.integers(0, 2, size=self.dim, dtype=np.int8)

        # Storage: parallel arrays per slot for fast slot-wise search.
        self.claim_text:   list[str] = []
        self.claim_source: list[str] = []
        self.claim_triple: list[tuple[str, str, str]] = []
        self._subj_stack: np.ndarray | None = None
        self._verb_stack: np.ndarray | None = None
        self._obj_stack:  np.ndarray | None = None
        self._dirty = True
        self.abstain_threshold = abstain_threshold

    # ─── Binding primitives ─────────────────────────────────

    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.bitwise_xor(a, b)

    def _bundle_balanced(self, hvs: list[np.ndarray]) -> np.ndarray:
        """Majority bundle with top-D/2 partition (same scheme as
        the corpus encoder) so all hypervectors have D/2 bits set."""
        if not hvs:
            return np.zeros(self.dim, dtype=np.int8)
        # Convert {0,1} -> {-1,+1} for summation, then sign-threshold.
        acc = np.zeros(self.dim, dtype=np.int64)
        for h in hvs:
            acc += h.astype(np.int64) * 2 - 1
        # top-D/2 partition with deterministic jitter to break ties.
        jitter = self.encoder.rng.integers(0, 7, size=self.dim,
                                                dtype=np.int64)
        scored = acc * 7 + jitter
        top = np.argpartition(scored, self.dim // 2)[self.dim // 2:]
        out = np.zeros(self.dim, dtype=np.int8)
        out[top] = 1
        return out

    _word_hv_cache: dict[str, np.ndarray] = {}

    def _word_hv(self, word: str) -> np.ndarray:
        """Deterministic balanced word hypervector for slot encoding.

        Each lowercased token maps to a unique balanced binary vector
        seeded by a strong hash of the FULL token.  We use SHA-1
        (truncated) rather than `hash()` (which is salted per Python
        run) or naive int.from_bytes(...)[:8] (which collides when
        words share a 4-byte prefix — e.g. "austria" / "australia").

        Synonyms are handled UPSTREAM by _canon_verb mapping
        ("created" -> "developed"), so by the time we hit this
        function, the verb is already in canonical form.
        """
        import hashlib
        w = word.lower().strip()
        if not w:
            return np.zeros(self.dim, dtype=np.int8)
        if w in self._word_hv_cache:
            return self._word_hv_cache[w]
        # SHA-1 of the FULL utf-8 token, truncated to 8 bytes -> 64-bit
        # seed.  Different tokens differ in at least one bit with
        # extremely high probability.
        digest = hashlib.sha1(w.encode("utf-8")).digest()[:8]
        seed = int.from_bytes(digest, "little")
        rng = np.random.default_rng(seed)
        positions = rng.permutation(self.dim)[: self.dim // 2]
        hv = np.zeros(self.dim, dtype=np.int8)
        hv[positions] = 1
        self._word_hv_cache[w] = hv
        return hv

    # ─── Claim ingestion ────────────────────────────────────

    def add_sentence(self, sentence: str, source: str = "") -> int:
        """Extract claims from `sentence` and store per-slot HVs.
        Returns the number of claims added.

        If `source` is a wikipedia: source, the article topic is
        passed to the extractor so context-dependent claims like
        "Tokyo is the country's capital" inside the Japan article
        resolve to (Japan, capital, Tokyo).
        """
        topic = None
        if source.startswith("wikipedia:"):
            topic = source.split(":", 1)[1].replace("_", " ")
        added = 0
        for subj, verb, obj in extract_claims(sentence, article_topic=topic):
            self.claim_text.append(sentence)
            self.claim_source.append(source)
            self.claim_triple.append((subj, verb, obj))
            added += 1
        if added:
            self._dirty = True
        return added

    def _rebuild_stacks(self):
        """Encode per-slot stacks once for batch search."""
        if not self.claim_triple:
            self._subj_stack = None
            self._verb_stack = None
            self._obj_stack  = None
            self._dirty = False
            return
        # Subject: encode by KEY TOKEN (usually the surname / proper-noun
        # head).  This lets "Marie Curie" match "Maria Salomea Skłodowska
        # Curie" because both reduce to "curie".  See _key_token().
        subj_hvs = [self._key_token_hv(s) for (s, v, o) in self.claim_triple]
        verb_hvs = [self._word_hv(v)        for (s, v, o) in self.claim_triple]
        obj_hvs  = [self._encode_phrase(o) for (s, v, o) in self.claim_triple]
        self._subj_stack = np.stack(subj_hvs)
        self._verb_stack = np.stack(verb_hvs)
        self._obj_stack  = np.stack(obj_hvs)
        self._dirty = False

    def _key_token(self, phrase: str) -> str:
        """Return the "key" token of a multi-word phrase.

        Heuristic: the LAST capitalised token in the phrase is usually
        the surname or proper-noun head ("Maria Salomea Skłodowska
        Curie" -> "Curie", "Sir Isaac Newton" -> "Newton").  If no
        token is capitalised (e.g. an object like "theory of relativity"),
        fall back to the last content word.
        """
        if not phrase:
            return ""
        toks = _tokens(phrase)
        if not toks:
            return ""
        # Prefer the LAST capitalised token.
        for w in reversed(toks):
            if w and w[0].isupper():
                return w.lower()
        # Fall back to last content (non-stop) token.
        for w in reversed(toks):
            if w.lower() not in _STOP and len(w) >= 2:
                return w.lower()
        return toks[-1].lower()

    def _key_token_hv(self, phrase: str) -> np.ndarray:
        """HV for the phrase's key token (see _key_token)."""
        return self._word_hv(self._key_token(phrase))

    def add_corpus(self, sentences: list[str],
                     sources: list[str] | None = None) -> int:
        sources = sources or [""] * len(sentences)
        n = 0
        for s, src in zip(sentences, sources):
            n += self.add_sentence(s, src)
        return n

    # ─── Search ─────────────────────────────────────────────

    def _ensure_stacks(self):
        if self._dirty:
            self._rebuild_stacks()

    def answer(self, query: str) -> dict | None:
        """Try to answer the query via slot-wise role-bound search.
        Returns None when the question can't be parsed OR when no
        stored claim is close enough (abstention).
        """
        parsed = parse_question(query)
        if parsed is None:
            return None
        self._ensure_stacks()
        if self._subj_stack is None:
            return None

        unknown = parsed["unknown"]
        # Encode each KNOWN slot.  Unknown slot doesn't contribute.
        # SUBJ uses key-token encoding so partial names match
        # (e.g. "Marie Curie" matches stored "Maria ... Curie").
        q_subj_hv = (self._key_token_hv(parsed["subj"])
                       if unknown != "SUBJ" and parsed.get("subj") else None)
        q_verb_hv = (self._word_hv(parsed["verb"])
                       if unknown != "VERB" and parsed.get("verb") else None)
        q_obj_hv  = (self._encode_phrase(parsed["obj"])
                       if unknown != "OBJ" and parsed.get("obj") else None)

        n_claims = self._subj_stack.shape[0]
        # Track each known slot's per-claim similarity separately.  We
        # combine via geometric-style aggregation (sum + min check)
        # so a claim with a single high-similarity slot can't win
        # if its other known slots are weak.
        per_slot_sims: list[np.ndarray] = []
        if q_subj_hv is not None:
            dists = np.bitwise_xor(self._subj_stack,
                                     q_subj_hv[None, :]).sum(axis=1)
            per_slot_sims.append(1.0 - 2.0 * dists / self.dim)
        if q_verb_hv is not None:
            dists = np.bitwise_xor(self._verb_stack,
                                     q_verb_hv[None, :]).sum(axis=1)
            per_slot_sims.append(1.0 - 2.0 * dists / self.dim)
        if q_obj_hv is not None:
            dists = np.bitwise_xor(self._obj_stack,
                                     q_obj_hv[None, :]).sum(axis=1)
            per_slot_sims.append(1.0 - 2.0 * dists / self.dim)
        if not per_slot_sims:
            return None
        slot_matrix = np.stack(per_slot_sims, axis=0)    # (n_slots, n_claims)
        mean_sim = slot_matrix.mean(axis=0)
        min_sim  = slot_matrix.min(axis=0)
        # Combined score: mean - penalty for a weak slot.  A claim
        # with two strong slots and one weak slot scores below a
        # claim with two strong slots and one absent slot.
        scores = mean_sim - 0.5 * (mean_sim - min_sim)
        order = np.argsort(-scores)[:5]
        best = int(order[0])
        sim = float(scores[best])
        min_at_best = float(min_sim[best])
        # Abstain unless BOTH the combined score AND the weakest
        # slot score clear the threshold.  This prevents a claim
        # that matches only one slot from winning.
        if (sim < self.abstain_threshold
                or min_at_best < self.abstain_threshold):
            return None

        subj, verb, obj = self.claim_triple[best]
        answer_word = {"SUBJ": subj, "VERB": verb, "OBJ": obj}[unknown]
        return {
            "sentence":     self.claim_text[best],
            "source":       self.claim_source[best],
            "subj":         subj,
            "verb":         verb,
            "obj":          obj,
            "unknown_role": unknown,
            "answer_word":  answer_word,
            "similarity":   sim,
            "rank_explain": [(float(scores[int(i)]),
                                self.claim_triple[int(i)],
                                self.claim_source[int(i)])
                                for i in order],
        }

    def _encode_phrase(self, phrase: str) -> np.ndarray:
        """Encode a multi-word phrase as the bundle of its content
        words' RI vectors.  Falls back to a single word's HV when
        the phrase is one token."""
        words = [w for w in _tokens(phrase.lower())
                  if w not in _STOP and len(w) >= 2]
        if not words:
            return self._word_hv(phrase)
        if len(words) == 1:
            return self._word_hv(words[0])
        # Bundle word-level HVs using the same balanced scheme.
        return self._bundle_balanced([self._word_hv(w) for w in words])

    # ─── Inspection ─────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "n_claims": len(self.claim_triple),
            "dim":      self.dim,
            "abstain_threshold": self.abstain_threshold,
        }
