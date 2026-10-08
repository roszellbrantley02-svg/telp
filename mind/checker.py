"""
mind/checker.py - Telp checks the model's answer against its own sources.

In harness mode a language model writes the reply and cites the numbered
evidence Telp gave it as [n]. Models still slip: a wrong year, a name that
was never in the sources, a citation to a source that doesn't exist. Telp
promises never to state what it can't back up, so before the user sees the
answer every sentence is checked here - by plain rules, no language model:

  * the answer is split into sentences, carefully: "Dr." / "e.g." / "c."
    don't end a sentence, "13.8" is a number, citations written after the
    full stop ("...Iceland. [1]") stay with their sentence, and bullet or
    numbered list items are sentences of their own
  * greetings, "I don't know" / "the sources don't say", questions back to
    the user and lead-ins ("Here's what I found:") make no claim
  * every claim must be backed by the sources it cites - or, when it cites
    nothing, by the single best-matching source:
      - each NUMBER, YEAR and DATE in it must be in that source
        ("January 8, 1642" == "8 January 1642", "390,000" == "390000",
        "13.8 billion" is exactly 13.8 billion - not 14 billion)
      - each NAME in it must be in that source (Reykjavík == Reykjavik,
        "Italian" finds "Italy")
      - its other meaningful words must be there too, with light stemming
        ("discovered" finds "discovery"), or close in meaning when an
        encoder is given (encoder.focus_alignment)
      - a "not" the source doesn't have - or a source's "not" the
        sentence dropped - fails it
    A citation to a source Telp never supplied fails the sentence.
  * annotate() gives what the user sees: unverified sentences marked, then
    the list of sources actually used. summarize() gives the counts.

This is a precision tool. A true sentence worded far from its source may be
marked [unverified] - honest, if over-cautious - but a sentence carrying a
number or a name that no source contains is never passed as verified.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from bisect import bisect_left
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from mind.harness_types import Evidence, SentenceCheck  # noqa: E402


# ─── tuning ─────────────────────────────────────────────────────────

MIN_OVERLAP = 0.6       # weighted share of a claim's words its source must have
MISS_EVERY = 8          # one unmatched informative word forgiven per this many
GENERIC_WEIGHT = 0.25   # "known", "located", "famous"... count for little
ALIGN_MIN = 0.72        # encoder word-meaning similarity that counts as a match
WINDOW = 6              # words either side of a number that say what it is
UNVERIFIED = " [unverified]"


# ─── vocabulary ─────────────────────────────────────────────────────

_STOP = frozenset("""
a about above across after afterwards again against ago ai all almost along
already also although always am among amongst an and another any anyone
anything anyway anywhere are around as at be became because been before
being below beside besides between beyond both but by ca can cannot could
did do does doing done down during each either else elsewhere enough etc
even ever every everyone everything everywhere few for from further had has
have having he hence her here hers herself him himself his how however i if
in indeed into is it its itself just let lets like may me might mine more
moreover much must my myself neither no nobody none nor not nothing now of
off often on once one only onto or other others otherwise our ours
ourselves out over own per perhaps quite rather really same shall she
should since so some somehow someone something sometimes somewhat still
such than that the their theirs them themselves then there thereby
therefore these they this those though through throughout thus to together
too toward towards under until up upon us very via was we well were what
whatever when whenever where whereas wherever whether which while who
whoever whom whose why will with within without wo would yet you your
yours yourself yourselves yes yeah ok okay s t
""".split())

_NEGATIONS = frozenset(
    "not no never none nobody nothing neither nor cannot nowhere".split())
_CONTRAST = frozenset("but rather instead although though however".split())
_CLITICS = frozenset({"s", "re", "ve", "d", "ll", "m"})

# Words that carry little of a claim: hedges, fillers, light verbs. A
# sentence can add or drop these and still say what its source says.
_GENERIC = frozenset("""
known called named name names located situated lie lies lying include
includes included including contain contains contained containing lot lots
major main important significant famous notable noted renowned prominent
popular great key primary leading considered regarded seen described
referred often usually generally typically commonly widely mainly mostly
primarily especially particularly approximately roughly nearly almost
estimated estimate total amount part parts kind type sort way thing things
place area home serve serves served serving become becomes became make
makes made making get gets got take takes took give gives gave go goes went
come comes came use used uses using find finds found show shows showed
shown say says said according fact actually simply currently today recently
time times year years day days various certain specific particular several
many numerous multiple whole entire full real true remains remain remained
anymore longer
""".split())

# Words that talk about the answer itself rather than the world: a
# sentence made only of these ("Here's what I found") claims nothing.
_META = frozenset("""
here there found find finding findings summary summarize summarized short
overall answer answers answered question questions source sources memory
memories note notes according following follows below above detail details
information info fact facts key point points main quick brief briefly
explain explanation look looked looking check checked search searched say
says said tell told mention mentioned know knew known remember recall think
believe guess sure certain hope help helps thing things way something
anything everything result results based ask asked telp assistant word
words provided provide given gave give clear clearly short long story
""".split())

_DISCOURSE = frozenset("""
also however overall unfortunately fortunately interestingly notably
additionally furthermore moreover finally firstly secondly thirdly lastly
first second third next then today currently historically generally
typically basically actually honestly indeed instead meanwhile later
earlier now here there according based given note summary hello hi hey
thanks thank sorry please great good sure yes no well so in short today
born died known
""".split())

_NAME_SKIP = frozenset("i telp ad bc bce ce am pm ok okay dr mr mrs ms mx "
                       "prof".split())
_CONNECTORS = frozenset(
    "of the de da di du del der von van la le al bin ibn".split())

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7,
    "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12,
    "dec": 12}
_MONTH_WORDS = frozenset(m for m in _MONTHS if len(m) > 3) | {"may"}
_WEEKDAYS = frozenset("monday tuesday wednesday thursday friday saturday "
                      "sunday".split())

_UNITS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
# "first" and "second" are left out on purpose: "First, ..." and "per
# second" are not numbers. They stay ordinary words that must match.
_ORDINALS = {"third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
             "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
             "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
             "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
             "eighteenth": 18, "nineteenth": 19, "twentieth": 20}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6,
           "billion": 10 ** 9, "trillion": 10 ** 12}
_SCALE_ABBR = {**_SCALES, "bn": 10 ** 9, "mn": 10 ** 6}

# the same thing spelled two ways (units, British/American)
_ALIASES = {"km": "kilometer", "kilometre": "kilometer", "kms": "kilometer",
            "kilometres": "kilometer", "metre": "meter", "metres": "meter",
            "cm": "centimeter", "centimetre": "centimeter", "mm": "millimeter",
            "kg": "kilogram", "kilogramme": "kilogram", "g": "gram",
            "lb": "pound", "lbs": "pound", "mi": "mile", "ft": "foot",
            "feet": "foot", "mph": "mile", "colour": "color",
            "colours": "color", "centre": "center", "theatre": "theater",
            "favourite": "favorite", "grey": "gray", "metres'": "meter"}

_COMMON_START = _STOP | _GENERIC | _META | _DISCOURSE


# ─── text helpers ───────────────────────────────────────────────────

_TRANSLIT = str.maketrans({
    "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "æ": "ae", "Æ": "Ae",
    "ø": "o", "Ø": "O", "ß": "ss", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
    "œ": "oe", "Œ": "Oe", "ı": "i", "’": "'", "‘": "'", "ʼ": "'",
    "“": '"', "”": '"', "−": "-", "–": "-", "—": "-", " ": " ",
    " ": " ", " ": " "})


def _fold(text: str) -> str:
    """Accents off, case kept: 'Reykjavík' -> 'Reykjavik', 'Þing' ->
    'Thing'. Both sides of every comparison are folded the same way."""
    text = text.translate(_TRANSLIT)
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


_TOK = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


@dataclass(slots=True)
class _Tok:
    start: int
    end: int
    raw: str
    base: str       # lower case, clitics off: "Iceland's" -> "iceland"
    neg: bool       # not / never / "wasn't"...


def _base(raw: str) -> tuple[str, bool]:
    w = raw.lower()
    if w.endswith("n't"):
        return w[:-3], True
    if "'" in w:
        head, _, tail = w.partition("'")
        w = head if tail in _CLITICS else w.replace("'", "")
    return w, w in _NEGATIONS


def _tokens(text: str) -> list[_Tok]:
    out = []
    for m in _TOK.finditer(text):
        base, neg = _base(m.group(0))
        out.append(_Tok(m.start(), m.end(), m.group(0), base, neg))
    return out


# Suffixes stripped once after the plural: enough for 'discovered' to find
# 'discovery', 'astronomer' 'astronomy', 'Italian' 'Italy'.
_SUFFIXES = ("ational", "ations", "ation", "ingly", "ically", "ness",
             "ment", "ments", "ical", "ally", "edly", "ity", "ing", "ian",
             "est", "ous", "ful", "ive", "ed", "er", "ly", "al", "ic", "y",
             "e")
_SUFFIXES = tuple(sorted(_SUFFIXES, key=len, reverse=True))
_IRREGULAR = {"died": "die", "dies": "die", "dying": "die", "children": "child",
              "men": "man", "women": "woman", "people": "person",
              "lives": "life", "wives": "wife", "mice": "mouse",
              "geese": "goose", "teeth": "tooth", "big": "large",
              "bigger": "larger", "biggest": "largest"}


@lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    """Light stem: plural off, then one common suffix."""
    w = _ALIASES.get(word, word)
    w = _IRREGULAR.get(w, w)
    if len(w) > 5 and w.endswith("our"):
        w = w[:-3] + "or"
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith(("ses", "xes", "zes", "ches", "shes")):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


_GENERIC_STEMS = frozenset(_stem(w) for w in _GENERIC)
_META_STEMS = frozenset(_stem(w) for w in _META)


@lru_cache(maxsize=65536)
def _name_forms(base: str) -> frozenset[str]:
    """The spellings a name may take in a source: plural or possessive off,
    and the country behind a demonym ('italian' -> 'ital' like 'italy',
    'icelandic' -> 'iceland', 'russian' -> 'russia', 'japanese' ->
    'japan'). Deliberately narrow: 'Mars' must never find 'Maria'."""
    forms = {base}
    if len(base) > 3 and base.endswith("s") and not base.endswith("ss"):
        forms.add(base[:-1])
    if len(base) > 5 and base.endswith("ian"):
        forms |= {base[:-1], base[:-3]}
    elif len(base) > 4 and base.endswith("an"):
        forms.add(base[:-1])
    if len(base) > 5 and base.endswith(("ese", "ish")):
        forms.add(base[:-3])
    if len(base) > 4 and base.endswith("ic"):
        forms.add(base[:-2])
    if len(base) > 3 and base.endswith("y"):
        forms.add(base[:-1])
    return frozenset(f for f in forms if len(f) >= 3)


def _join(labels: list[str], word: str = "or") -> str:
    labels = list(dict.fromkeys(labels))
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + f" {word} " + labels[-1]


def _refs(ns: list[int]) -> str:
    return _join([f"[{n}]" for n in ns], "and")


# ─── citations ──────────────────────────────────────────────────────

_CITE_ONE = (
    r"\[\^?\s*(?:(?:sources?|src|refs?|evidence)\s*#?\s*)?\d{1,3}"
    r"(?:\s*(?:,|;|&|and|-|–|—)\s*(?:(?:sources?|refs?)\s*#?\s*)?\d{1,3})*"
    r"\s*\]"
    r"|【\s*\d{1,3}(?:\s*[,;]\s*\d{1,3})*\s*】"
    r"|\((?:sources?|refs?)\s*#?\s*\d{1,3}(?:\s*(?:,|&|and)\s*\d{1,3})*\)")
_CITE_RE = re.compile(_CITE_ONE, re.I)
_CITE_RUN = re.compile(r"(?:\s*(?:" + _CITE_ONE + r"))+", re.I)
_CITE_TAIL = re.compile(r"(?:" + _CITE_ONE + r")\s*$", re.I)
_CITE_NUMS = re.compile(r"(\d{1,3})\s*[-–—]\s*(\d{1,3})|(\d{1,3})")


def _parse_cites(text: str) -> list[int]:
    """[1][2], [1, 3], [2-4], 【1】, (source 2) -> the numbers, in order."""
    out: list[int] = []
    for m in _CITE_RE.finditer(text):
        for r in _CITE_NUMS.finditer(m.group(0)):
            if r.group(3):
                nums = [int(r.group(3))]
            else:
                a, b = int(r.group(1)), int(r.group(2))
                nums = list(range(a, b + 1)) if a <= b <= a + 20 else [a, b]
            out.extend(n for n in nums if n not in out)
    return out


# ─── the answer's layout: paragraphs, lists, headings, tables ───────

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
_BOLD_LINE = re.compile(r"^\s*(?:\*\*|__)([^*_]{1,80})(?:\*\*|__)\s*:?\s*$")
_HR = re.compile(r"^\s*(?:[-*_]\s*){3,}$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")
_LIST_MARK = re.compile(
    r"^\s*(?:>\s*)?(?:[-*+•▪◦‣–]|\(?\d{1,3}[.)]|\(?[a-z][.)]|\([ivx]{1,4}\))"
    r"\s+")
_QUOTE_MARK = re.compile(r"^\s*>\s?")
_REFS_HEAD = re.compile(
    r"^\W*(?:sources?|references?|citations?|cited sources)\W*:?\W*$", re.I)
_REFS_INLINE = re.compile(
    r"^\W*(?:sources?|references?|citations?)\W*:\s*(.*)$", re.I)
_REF_LINE = re.compile(r"^(?:[-*•]\s*)?(?:\[\^?\d{1,3}\]|【\d{1,3}】|\d{1,3}[.)])")
_ENDS_CLOSED = re.compile(r"(?:[.!?:…]|\])[\"'”’)\]*_]*\s*$")


def _clean(answer: str) -> str:
    """The answer without the model's thinking. llm_client already splits
    thinking off; this is the safety net for anything that slipped by."""
    text = (answer or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _THINK_BLOCK.sub("", text)
    low = text.lower()
    if "</think>" in low:       # thinking opened by the prompt template
        text = text[low.rfind("</think>") + len("</think>"):]
        low = text.lower()
    if "<think>" in low:        # cut off mid-thought: the rest isn't an answer
        text = text[:low.find("<think>")]
    return text.strip()


@dataclass
class _Unit:
    start: int
    end: int
    kind: str      # text | heading | table_header | table_row | refs | skip


def _refs_inline(s: str) -> bool:
    """'Sources: [1], [2]' or 'Sources: [1] wikipedia:Iceland' - the
    model's own reference list, which Telp replaces with its own."""
    m = _REFS_INLINE.match(s)
    if not m:
        return False
    rest = m.group(1).strip()
    return (not re.search(r"[^\W_]", _CITE_RE.sub("", rest))
            or bool(_CITE_RE.match(rest)))


def _layout(text: str) -> list[_Unit]:
    """Cut the answer into blocks: paragraphs (a line soft-wrapped into the
    next is joined back), list items, headings, table rows, code (skipped)
    and the model's own source list (replaced by Telp's)."""
    units: list[_Unit] = []
    pos = 0
    in_fence = in_refs = False
    can_join = False            # may this line continue the last paragraph?
    last_line = ""
    for line in text.split("\n"):
        ls, le = pos, pos + len(line)
        pos = le + 1
        s = line.strip()
        if _FENCE.match(line):
            in_fence = not in_fence
            units.append(_Unit(ls, le, "skip"))
            can_join = False
            continue
        if in_fence:
            units.append(_Unit(ls, le, "skip"))
            continue
        if not s:
            can_join = False
            continue
        if _REFS_HEAD.match(s) or _refs_inline(s):
            in_refs = bool(_REFS_HEAD.match(s))
            units.append(_Unit(ls, le, "refs"))
            can_join = False
            continue
        if in_refs:
            if _REF_LINE.match(s):
                units.append(_Unit(ls, le, "refs"))
                continue
            in_refs = False
        if _CITE_RUN.fullmatch(s) and units and units[-1].kind == "text":
            units[-1].end = le          # a citation alone on its line
            continue
        if _HR.match(s):
            units.append(_Unit(ls, le, "skip"))
            can_join = False
            continue
        lead = len(line) - len(line.lstrip())
        bold = _BOLD_LINE.match(s)
        if _HEADING.match(s) or (
                bold and not re.search(r"[.!?]\s*$", bold.group(1))
                and not _CITE_RE.search(s) and len(bold.group(1).split()) <= 10):
            units.append(_Unit(ls + lead, le, "heading"))
            can_join = False
            continue
        if s.startswith("|"):
            if _TABLE_SEP.match(s):
                if units and units[-1].kind == "table_row":
                    units[-1].kind = "table_header"
                units.append(_Unit(ls, le, "skip"))
            else:
                units.append(_Unit(ls + lead, le, "table_row"))
            can_join = False
            continue
        m = _LIST_MARK.match(line)
        if m:
            units.append(_Unit(ls + m.end(), le, "text"))
            can_join, last_line = True, s
            continue
        if (can_join and units and units[-1].kind == "text"
                and not _ENDS_CLOSED.search(last_line)):
            units[-1].end = le          # soft-wrapped sentence goes on
            last_line = s
            continue
        q = _QUOTE_MARK.match(line)
        start = ls + (q.end() if q else lead)
        units.append(_Unit(start, le, "text"))
        can_join, last_line = True, s
    return units


# ─── sentences ──────────────────────────────────────────────────────

# a full stop after these never ends a sentence ("Dr. Smith", "e.g. Paris")
_NEVER_END = frozenset("""dr mr mrs ms mx prof st mt ft gen col lt maj sgt
capt cmdr adm rev fr sen rep gov pres hon messrs mme mlle e.g i.e cf viz vs
v approx""".split())
# ...nor after these when a number follows ("c. 1500", "No. 5", "Jan. 8")
_DIGIT_ABBR = frozenset("""c ca circa no nos nr p pp vol vols fig figs ch
sec est b d fl r jan feb mar apr jun jul aug sep sept oct nov dec""".split())
# ...and these end one only before a capital ("...pears, etc. The")
_UPPER_END = frozenset("etc inc ltd co corp jr sr al bros llc plc a.m p.m"
                       .split())
# after an initial ("J.") or "U.S." only these words start a new sentence
_STARTERS = frozenset("""the a an it its this that these those he she they
we i in on at his her their there however also but and so as after before
when while during since for from by with today both each some many most
one our my your if although though yet then thus still""".split())
_CLOSERS = "\"')]»*_”’"
_OPENERS = "\"'“‘(*_["


def _is_boundary(text: str, s0: int, i: int, punct: str, j: int,
                 end: int) -> bool:
    """Does the punctuation at text[i:j] end the sentence begun at s0?"""
    if j < end and not text[j].isspace():
        return False                    # "13.8", "e.g.x", "example.com"
    k = j
    while k < end and text[k].isspace():
        k += 1
    if k >= end:
        return True
    k2 = k
    while k2 < end and text[k2] in _OPENERS:
        k2 += 1
    if k2 < end and text[k2].islower():
        return False                    # "...approx. five", "vs. the"
    if punct != ".":
        return True
    w0 = i
    while w0 > s0 and (text[w0 - 1].isalnum() or text[w0 - 1] == "."):
        w0 -= 1
    prev = text[w0:i]
    low = prev.lower()
    nxt_digit = k2 < end and text[k2].isdigit()
    if not prev:
        return True
    if low in _NEVER_END or (low in _DIGIT_ABBR and nxt_digit):
        return False
    if prev.isdigit() and len(prev) <= 2:
        before = text[s0:w0].rstrip()
        if not before or before.endswith(":"):
            return False                # an enumerator: "Steps: 1. Boil..."
    if low in _UPPER_END:
        return True
    letters = prev.replace(".", "")
    if (len(prev) == 1 and prev.isupper()) or (
            "." in prev and letters.isalpha() and letters.isupper()):
        word = re.match(r"[^\W\d_]+", text[k2:end])
        return bool(word) and word.group(0).lower() in _STARTERS
    return True


def _split_unit(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentence spans inside one block, citations kept with the sentence
    they follow."""
    spans: list[tuple[int, int]] = []
    s0, i = start, start
    while i < end:
        if text[i] not in ".!?…":
            i += 1
            continue
        j = i + 1
        while j < end and text[j] in ".!?…":
            j += 1
        punct = text[i:j]
        while j < end and text[j] in _CLOSERS:
            j += 1
        m = _CITE_RUN.match(text, j, end)
        if m:
            j = m.end()
        if _is_boundary(text, s0, i, punct, j, end):
            spans.append((s0, j))
            k = j
            while k < end and text[k].isspace():
                k += 1
            s0 = i = k
        else:
            i = j
    if s0 < end:
        spans.append((s0, end))
    out: list[tuple[int, int]] = []
    for a, b in spans:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a >= b:
            continue
        bare = _CITE_RE.sub("", text[a:b])
        if out and not re.search(r"[^\W_]", bare):
            out[-1] = (out[-1][0], b)   # a stray "[2]." joins the sentence before
        else:
            out.append((a, b))
    return out


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """(kind, start, end) of every sentence, in order."""
    out = []
    for u in _layout(text):
        if u.kind in ("skip", "refs"):
            continue
        if u.kind == "text":
            out.extend(("text", a, b) for a, b in _split_unit(text, u.start,
                                                              u.end))
            continue
        a, b = u.start, u.end
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            out.append((u.kind, a, b))
    return out


def split_sentences(answer: str) -> list[str]:
    """The answer's sentences as check_answer sees them (thinking removed,
    the model's own source list left out)."""
    text = _clean(answer)
    return [text[a:b] for _, a, b in _sentence_spans(text)]


# ─── what a sentence says: keys and words ───────────────────────────

_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_LEAD_ENUM = re.compile(r"^\s*(?:\(?\d{1,2}[.)]|\(?[a-z][.)])\s+")
_INLINE_ENUM = re.compile(r"(?<=:)\s*\d{1,2}[.)]\s+")


def _strip_markup(sentence: str, table: bool = False) -> str:
    """The sentence as words: no citations, markdown, enumerators."""
    s = _CITE_RE.sub(" ", sentence)
    s = _MD_LINK.sub(r"\1", s)
    s = _HTML_TAG.sub(" ", s)
    for mark in ("**", "__", "`", "~~"):
        s = s.replace(mark, "")
    s = re.sub(r"(?<![\w*])\*(?=\S)|(?<=\S)\*(?![\w*])", "", s)
    if table:
        s = s.strip().strip("|").replace("|", " ; ")
    s = _QUOTE_MARK.sub("", s)
    s = _LEAD_ENUM.sub("", s)
    s = _INLINE_ENUM.sub(" ", s)
    s = re.sub(r"\(\s*\)", " ", s)
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    return " ".join(s.split())


@dataclass(slots=True)
class _Key:
    """Something a source must contain verbatim (give or take formatting)."""
    kind: str               # number | date | name | id
    value: object           # Decimal | [(y, m, d)...] | base word
    label: str              # how a note names it
    pos: int                # token position in the sentence
    group: int = 0          # names: which name ("Galileo Galilei" = one)
    forms: frozenset = frozenset()


@dataclass
class _Claim:
    keys: list[_Key]
    words: list[tuple[str, float, str, int]]    # stem, weight, word, pos
    stems: set[str]
    names: set[str]
    neg_follow: list[str]       # stems of what the sentence negates


_MON = (r"(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|"
        r"oct|nov|dec)\.?")
_ORD = r"(?:st|nd|rd|th)?"
_DATE_RX = [
    ("iso", re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")),
    ("dmy", re.compile(rf"\b(\d{{1,2}}){_ORD}(?:\s+of)?\s+{_MON},?\s+"
                       rf"(\d{{1,4}})\b")),
    ("mdy", re.compile(rf"\b{_MON}\s+(\d{{1,2}}){_ORD},?\s+(\d{{1,4}})\b")),
    ("slash", re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")),
    ("my", re.compile(rf"\b{_MON},?\s+(\d{{3,4}})\b")),
    ("dm", re.compile(rf"\b(\d{{1,2}}){_ORD}(?:\s+of)?\s+{_MON}(?![a-z])")),
    ("md", re.compile(rf"\b{_MON}\s+(\d{{1,2}}){_ORD}\b(?![.,]?\d)")),
]
_NUM_RX = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?:st|nd|rd|th|s)?"
    r"(?!\w)(?:\s*(?:%|percent\b|per\s+cent\b))?"
    r"(?:\s+(thousand|million|billion|trillion|bn|mn)\b)?")
_ORDINAL_DIGITS = re.compile(r"\d+(?:st|nd|rd|th|s)")


def _valid(y, m, d) -> bool:
    return 1 <= m <= 12 and (d is None or 1 <= d <= 31)


def _date_alts(kind: str, g: tuple) -> list[tuple]:
    """A date match as (year, month, day) readings; None = not given."""
    if kind == "iso":
        cands = [(int(g[0]), int(g[1]), int(g[2]))]
    elif kind == "dmy":
        cands = [(int(g[2]), _MONTHS[g[1]], int(g[0]))]
    elif kind == "mdy":
        cands = [(int(g[2]), _MONTHS[g[0]], int(g[1]))]
    elif kind == "slash":           # 8/1/1642: day-first or month-first
        a, b, y = int(g[0]), int(g[1]), int(g[2])
        cands = list(dict.fromkeys([(y, b, a), (y, a, b)]))
    elif kind == "my":
        cands = [(int(g[1]), _MONTHS[g[0]], None)]
    elif kind == "dm":
        cands = [(None, _MONTHS[g[1]], int(g[0]))]
    else:
        cands = [(None, _MONTHS[g[0]], int(g[1]))]
    return [c for c in cands if _valid(*c)]


def _find_dates(low: str) -> list[tuple[list, int, int]]:
    found, taken = [], []
    for kind, rx in _DATE_RX:
        for m in rx.finditer(low):
            s, e = m.span()
            if any(s < b and e > a for a, b in taken):
                continue
            alts = _date_alts(kind, m.groups())
            if alts:
                taken.append((s, e))
                found.append((alts, s, e))
    found.sort(key=lambda f: f[1])
    return found


def _find_numbers(masked: str) -> list[tuple[Decimal, Decimal | None, int, int]]:
    """(value, value without its scale word, start, end): '13.8 billion'
    -> 13800000000 and 13.8; '390,000' -> 390000; '-40' -> -40."""
    out = []
    for m in _NUM_RX.finditer(masked):
        s, e = m.span()
        raw = m.group(1).replace(",", "")
        v = Decimal(raw + ("." + m.group(2) if m.group(2) else ""))
        if s >= 1 and masked[s - 1] == "-" and (s == 1 or masked[s - 2] in " (\t\n"):
            v, s = -v, s - 1
        scale = _SCALE_ABBR.get(m.group(3) or "")
        out.append((v * scale if scale else v, v if scale else None, s, e))
    return out


def _word_numbers(toks: list[_Tok], used: list[bool],
                  text: str) -> list[tuple[Decimal, int, int]]:
    """Numbers written as words: 'four', 'twenty-one', 'a million',
    'two hundred', 'fifth'. Returns (value, first token, after last)."""
    out = []
    n, i = len(toks), 0
    while i < n:
        b = toks[i].base
        nb = toks[i + 1].base if i + 1 < n else ""
        starts = (b in _UNITS or b in _TENS or b in _ORDINALS
                  or (b in ("one", "a", "an") and nb in _SCALES))
        if used[i] or not starts:
            i += 1
            continue
        if (i == 0 and b in _ORDINALS and i + 1 < n
                and "," in text[toks[i].end:toks[i + 1].start]):
            i += 1                      # "Third, ..." is not a number
            continue
        total = cur = 0
        j = i
        while j < n and not used[j]:
            w = toks[j].base
            if w in _UNITS:
                cur += _UNITS[w]
            elif w in _TENS:
                cur += _TENS[w]
            elif w == "one" and (cur >= 20 and cur % 10 == 0
                                 or (j + 1 < n and toks[j + 1].base in _SCALES)):
                cur += 1
            elif w in ("a", "an") and j == i:
                cur = 1
            elif w in _SCALES and (cur or total):
                if w == "hundred":
                    cur *= 100
                else:
                    total, cur = total + cur * _SCALES[w], 0
            elif w in _ORDINALS:
                cur += _ORDINALS[w]
                j += 1
                break
            elif (w == "and" and cur and j + 1 < n
                  and (toks[j + 1].base in _UNITS or toks[j + 1].base in _TENS)):
                pass
            else:
                break
            j += 1
        value = total + cur
        if value and j > i:
            out.append((Decimal(value), i, j))
            for k in range(i, j):
                used[k] = True
        i = max(j, i + 1)
    return out


def _is_name(t: _Tok, i: int, text: str, vocab: set[str]) -> bool:
    """A capitalized word that names something. The first word of a
    sentence is a name only when it isn't a common word: 'Bananas are...'
    is not a name if 'bananas' is ever written in lower case."""
    raw, b = t.raw, t.base
    if not raw[0].isupper() or not b.isalpha():
        return False
    if (b in _NAME_SKIP or b in _STOP or b in _MONTH_WORDS
            or b in _WEEKDAYS or b in _NEGATIONS):
        return False
    initial = i == 0 or text[:t.start].rstrip()[-1:] in (':;"(\'-')
    if not initial:
        return True
    if raw.isupper() and len(raw) > 1:
        return True                     # NASA, UNESCO
    return not (b in _COMMON_START or b in vocab
                or b.endswith(("ed", "ing", "ly")))


def _following(toks: list[_Tok], i: int, limit: int = 3) -> list[str]:
    """Stems of the next few meaningful words after token i - what a
    'not' at i is about. Stops at 'but' ('not Akureyri but Reykjavik')."""
    out = []
    for t in toks[i + 1:]:
        if t.base in _CONTRAST:
            break
        if t.base in _STOP or t.neg:
            continue
        out.append(_stem(t.base))
        if len(out) >= limit:
            break
    return out


def _parse_claim(text: str, vocab: set[str]) -> _Claim:
    """Pull out what a source must contain for this sentence to stand:
    its numbers, dates and names (strict) and its other words (soft)."""
    toks = _tokens(text)
    low = text.lower()
    starts = [t.start for t in toks]
    used = [False] * len(toks)
    keys: list[_Key] = []

    def at(pos: int) -> int:
        return max(0, min(len(toks) - 1, bisect_left(starts, pos)))

    def consume(s: int, e: int) -> None:
        for k in range(bisect_left(starts, s), len(toks)):
            if toks[k].start >= e:
                break
            used[k] = True

    chars = list(low)
    for alts, s, e in _find_dates(low):
        keys.append(_Key("date", alts, text[s:e].strip(" ,."), at(s)))
        consume(s, e)
        chars[s:e] = " " * (e - s)
    masked = "".join(chars)
    for value, _unscaled, s, e in _find_numbers(masked):
        keys.append(_Key("number", value, text[s:e].strip(), at(s)))
        consume(s, e)
    for value, i0, i1 in _word_numbers(toks, used, text):
        keys.append(_Key("number", value,
                         text[toks[i0].start:toks[i1 - 1].end], i0))

    words: list[tuple[str, float, str, int]] = []
    names: set[str] = set()
    neg_follow: list[str] = []
    group, last_name = 0, -9
    for i, t in enumerate(toks):
        if used[i]:
            continue
        if t.neg and not (t.base == "no" and i + 1 < len(toks)
                          and toks[i + 1].base.isdigit()):
            neg_follow.extend(_following(toks, i))
            continue
        b = t.base
        if any(c.isdigit() for c in b):
            if any(c.isalpha() for c in b) and not _ORDINAL_DIGITS.fullmatch(b):
                keys.append(_Key("id", b, t.raw, i))       # A380, CO2
            continue
        if _is_name(t, i, text, vocab):
            joined = last_name == i - 1 or (
                last_name == i - 2 and toks[i - 1].base in _CONNECTORS)
            if not joined:
                group += 1
            keys.append(_Key("name", b, t.raw, i, group, _name_forms(b)))
            names.add(b)
            last_name = i
            continue
        if b in _STOP or len(b) < 2:
            continue
        st = _stem(b)
        generic = b in _GENERIC or st in _GENERIC_STEMS
        words.append((st, GENERIC_WEIGHT if generic else 1.0, b, i))
    return _Claim(keys, words, {w[0] for w in words}, names, neg_follow)


def _checkable(claim: _Claim) -> bool:
    """Does the sentence say anything about the world at all?"""
    return bool(claim.keys) or any(
        w[2] not in _META and w[0] not in _META_STEMS for w in claim.words)


# ─── the evidence, indexed once per answer ──────────────────────────

@dataclass
class _Item:
    ev: Evidence
    text: str                   # what an encoder reads
    tool: bool
    tokens: set[str] = field(default_factory=set)
    stems: set[str] = field(default_factory=set)
    prefixes: set[str] = field(default_factory=set)
    forms: set[str] = field(default_factory=set)
    numbers: set[Decimal] = field(default_factory=set)
    ymd: set[tuple] = field(default_factory=set)
    ym: set[tuple] = field(default_factory=set)
    md: set[tuple] = field(default_factory=set)
    neg_follow: set[str] = field(default_factory=set)
    lower: set[str] = field(default_factory=set)


def _add_date(it: _Item, y, m, d, as_numbers: bool = True) -> None:
    if y is not None and d is not None:
        it.ymd.add((y, m, d))
    if y is not None:
        it.ym.add((y, m))
        it.numbers.add(Decimal(y))
    if d is not None:
        it.md.add((m, d))
        if as_numbers:
            it.numbers.add(Decimal(d))


def _index_item(ev: Evidence) -> _Item:
    """Everything a sentence could be matched on, as sets: the source's
    words, stems, names, numbers and dates. Its title counts too
    ('wikipedia:Galileo Galilei'), since the model reads it."""
    body = _fold(ev.text or "")
    title = _fold(re.sub(r"[:_/|#]+", " ", ev.source or ""))
    full = f"{body} {title}".strip()
    it = _Item(ev=ev, text=ev.text or "", tool=(ev.kind == "tool"
                                               or (ev.source or "").startswith("tool:")))
    toks = _tokens(full)
    used = [False] * len(toks)
    low = full.lower()
    chars = list(low)
    starts = [t.start for t in toks]
    for alts, s, e in _find_dates(low):
        for y, m, d in alts:
            _add_date(it, y, m, d)
        chars[s:e] = " " * (e - s)
        for k in range(bisect_left(starts, s), len(toks)):
            if toks[k].start >= e:
                break
            used[k] = True
    for value, unscaled, s, e in _find_numbers("".join(chars)):
        it.numbers.add(value)
        if unscaled is not None:
            it.numbers.add(unscaled)
        for k in range(bisect_left(starts, s), len(toks)):
            if toks[k].start >= e:
                break
            used[k] = True
    for value, _i0, _i1 in _word_numbers(toks, used, full):
        it.numbers.add(value)
    if ev.created_at:                   # "saved 2026-07-02" is shown to the model
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", ev.created_at)
        if m and _valid(int(m.group(1)), int(m.group(2)), int(m.group(3))):
            _add_date(it, int(m.group(1)), int(m.group(2)), int(m.group(3)),
                      as_numbers=False)
    for t in toks:
        it.tokens.add(t.base)
        st = _stem(t.base)
        it.stems.add(st)
        if len(st) >= 6:
            it.prefixes.add(st[:6])
        it.forms |= _name_forms(t.base)
    body_toks = [t for t in toks if t.end <= len(body)]
    for i, t in enumerate(body_toks):
        if t.raw.islower():
            it.lower.add(t.base)
        if t.neg:
            it.neg_follow.update(_following(body_toks, i))
    return it


class _Index:
    def __init__(self, evidence: list[Evidence]):
        self.items = [_index_item(e) for e in evidence]
        self.by_n = {it.ev.n: it for it in self.items}
        self.lower: set[str] = set()
        for it in self.items:
            self.lower |= it.lower


class _Aligner:
    """encoder.focus_alignment, one word against one source, cached. Any
    failure switches it off: the checker then runs on words alone."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.ok = encoder is not None and hasattr(encoder, "focus_alignment")
        self.cache: dict[tuple[str, int], float] = {}

    def hit(self, word: str, it: _Item) -> bool:
        if not self.ok:
            return False
        key = (word, id(it))
        if key not in self.cache:
            try:
                self.cache[key] = float(
                    self.encoder.focus_alignment([word], [it.text])[0])
            except Exception:
                self.ok = False
                return False
        return self.cache[key] >= ALIGN_MIN


# ─── matching ───────────────────────────────────────────────────────

def _key_in(k: _Key, it: _Item) -> bool:
    if k.kind == "number":
        return k.value in it.numbers
    if k.kind == "name":
        return not k.forms.isdisjoint(it.forms)
    if k.kind == "date":
        for y, m, d in k.value:
            if y is not None and d is not None:
                if (y, m, d) in it.ymd:
                    return True
            elif y is not None:
                if (y, m) in it.ym:
                    return True
            elif (m, d) in it.md:
                return True
        return False
    return k.value in it.tokens


def _word_in(stem: str, it: _Item) -> bool:
    return stem in it.stems or (len(stem) >= 6 and stem[:6] in it.prefixes)


def _stem_hit(stem: str, stems: set[str]) -> bool:
    return stem in stems or (len(stem) >= 6 and any(
        s[:6] == stem[:6] for s in stems if len(s) >= 6))


@dataclass
class _Verdict:
    ok: bool
    score: float
    missing_keys: list[str]
    missing_words: list[str]
    problem: str = ""           # negation | mixed


def _negation_clash(claim: _Claim, items: list[_Item]) -> bool:
    """'Galileo was not born in Pisa' against 'Galileo was born in Pisa' -
    or the source's 'not' dropped by the sentence."""
    if claim.neg_follow:
        if any(it.neg_follow for it in items):
            return False
        return any(_word_in(s, it) for s in claim.neg_follow for it in items)
    mine = claim.stems | {_stem(n) for n in claim.names}
    return any(_stem_hit(s, mine) for it in items for s in it.neg_follow)


def _bound(claim: _Claim, items: list[_Item], aligner: _Aligner) -> bool:
    """When several sources back one sentence together, each number must
    come from a source about the same thing: 'Marie Curie was born in
    1564 [1][2]' must not pass because [1] says Galileo was born in 1564."""
    names = [k for k in claim.keys if k.kind == "name"]
    for k in claim.keys:
        cands = [it for it in items if _key_in(k, it)]
        if k.kind in ("number", "date"):
            if any(it.tool for it in cands):
                continue                # a computed value needs no subject
            window = [(s, w) for s, wt, w, p in claim.words
                      if wt >= 1 and abs(p - k.pos) <= WINDOW]

            def good(it: _Item) -> bool:
                if names and not any(_key_in(nk, it) for nk in names):
                    return False
                return not window or any(
                    _word_in(s, it) or aligner.hit(w, it) for s, w in window)
        else:
            # a name must share its source with another of the sentence's
            # keys - one that some remembered (non-tool) source holds; a
            # computed value never names anybody
            others = [o for o in claim.keys
                      if o is not k and (o.kind != "name" or o.group != k.group)
                      and any(_key_in(o, it) for it in items if not it.tool)]
            if not others:
                continue

            def good(it: _Item) -> bool:
                return any(_key_in(o, it) for o in others)
        if not any(good(it) for it in cands):
            return False
    return True


def _judge(claim: _Claim, items: list[_Item], aligner: _Aligner,
           use_encoder: bool = True) -> _Verdict:
    """Do these sources (one, or several cited together) back the claim?"""
    missing_keys = [k.label for k in claim.keys
                    if not any(_key_in(k, it) for it in items)]
    tool_only = all(it.tool for it in items)
    got = total = 0.0
    misses: list[str] = []
    informative = 0
    for stem, weight, word, _pos in claim.words:
        total += weight
        if weight >= 1:
            informative += 1
        hit = any(_word_in(stem, it) for it in items) or (
            use_encoder and any(aligner.hit(word, it) for it in items))
        if hit:
            got += weight
        elif weight >= 1 and word not in misses:
            misses.append(word)
    overlap = got / total if total else 1.0
    if tool_only:                       # "1564 + 77 = 1641" has no words to match
        overlap, misses = 1.0, []
    n_keys = len(claim.keys)
    key_share = (n_keys - len(missing_keys)) / n_keys if n_keys else 1.0
    score = round(overlap * key_share, 3)
    ok = (not missing_keys and overlap >= MIN_OVERLAP
          and len(misses) <= informative // MISS_EVERY)
    problem = ""
    if ok and _negation_clash(claim, items):
        ok, problem = False, "negation"
    elif ok and len(items) > 1 and not _bound(claim, items, aligner):
        ok, problem = False, "mixed"
    return _Verdict(ok, score, missing_keys, misses, problem)


def _best_single(claim: _Claim, items: list[_Item], aligner: _Aligner
                 ) -> tuple[_Item | None, _Verdict | None]:
    """The one source that backs the claim best (or comes closest)."""
    best_ok = None
    ranked = []
    for it in items:
        v = _judge(claim, [it], aligner, use_encoder=False)
        if v.ok and (best_ok is None or v.score > best_ok[1].score):
            best_ok = (it, v)
        ranked.append(((-len(v.missing_keys), v.score), it, v))
    if best_ok:
        return best_ok
    if not ranked:
        return None, None
    ranked.sort(key=lambda r: r[0], reverse=True)
    if aligner.ok:                      # meaning may close the word gap
        for _rank, it, v in ranked[:3]:
            if not v.missing_keys:
                v2 = _judge(claim, [it], aligner)
                if v2.ok:
                    return it, v2
    return ranked[0][1], ranked[0][2]


# ─── sentences that claim nothing ───────────────────────────────────

_GREETING = re.compile(r"""^(?:
    (?:hi|hello|hey|hiya|greetings|good\s+(?:morning|afternoon|evening|day))
        (?:\s+there)?(?:[\s,]+[\w'-]+)?
  | (?:thanks|thank\s+you)(?:\s+(?:so|very)\s+much)?(?:\s+for\s+.*)?
  | you'?re\s+(?:very\s+|most\s+)?welcome.*
  | no\s+problem | no\s+worries | sure(?:\s+thing)? | of\s+course | certainly
  | absolutely | okay | ok | alright | all\s+right | got\s+it | understood
  | great | good | nice | excellent | perfect | indeed | yes | yeah | yep
  | no | nope | right | exactly | correct | that'?s\s+(?:right|correct)
  | (?:great|good|interesting|excellent|fair)\s+question
  | (?:i'?m\s+|i\s+am\s+)?(?:happy|glad)\s+(?:to|i\s+could)\s+help.*
  | (?:i\s+)?hope\s+(?:this|that|it)\s+helps.*
  | let\s+me\s+know\s+.* | feel\s+free\s+.*
  | (?:is\s+there\s+)?anything\s+else.*
  | have\s+a\s+(?:nice|good|great|lovely)\s+.* | good\s+luck.* | enjoy.*
)$""", re.X)

# offers of more help - claims nothing as long as it names no number
_OFFER = re.compile(r"""^(?:
    (?:if\s+you(?:'d|\s+would)?\s+(?:like|want|need|wish)|should\s+you\s+\w+)
        .*(?:i\s+can|i'?ll|i\s+could|let\s+me|just\s+ask|ask\s+me).*
  | (?:just\s+)?ask\s+(?:me\s+)?(?:if|anytime|any\s+time|away|about).*
  | you\s+can\s+(?:always\s+)?ask\s+me.*
  | (?:i\s+can|i\s+could|i'?ll|i\s+will|let\s+me|shall\s+i|want\s+me\s+to)
        \s+(?:also\s+|gladly\s+)?(?:look|search|dig|find|check|fetch|learn|
        help|explain|summarize|remember|keep|try|see|think|
        tell\s+you\s+(?:more\s+)?about)\b.*
)$""", re.X)

_ABSTAIN = re.compile(r"""\b(?:
    i\s+(?:do\s+not|don'?t|did\s+not|didn'?t)\s+(?:know|have|see|find|
        remember|recall|have\s+(?:any|enough))\b
  | i'?m\s+not\s+(?:sure|certain|aware)\b | i\s+am\s+not\s+(?:sure|certain|aware)\b
  | i\s+(?:could\s*n[o']t|can\s*n[o']t|cannot|was\s+unable\s+to|am\s+unable\s+to|
        wasn'?t\s+able\s+to)\s+(?:find|tell|say|confirm|verify|see|answer|
        determine|locate)\b
  | i\s+have\s+no\s+(?:information|record|records|data|memory|memories|sources?|
        details?|idea)\b
  | (?:sources?|memory|memories|notes|evidence|records?|information)
        (?:\s+(?:i\s+have|provided|available|given|here))?\s+
        (?:do(?:es)?\s+not|don'?t|doesn'?t|did\s+not|didn'?t|never)\s+
        (?:say|mention|specify|state|cover|include|contain|tell|indicate|give|
        provide|list|address|discuss|answer|show)\b
  | (?:there\s+(?:is|are|was)\s+)?no\s+(?:information|mention|record|data|
        details?|sources?)\s+(?:about|on|of|regarding|for|in|that)\b
  | nothing\s+in\s+(?:my|the)\s+(?:sources?|memory|notes)\b
  | not\s+(?:mentioned|covered|stated|specified|included|given)\s+in\s+
        (?:my|the|any)\b
)""", re.X)

_CLAUSES = re.compile(
    r"\s*;\s*|,?\s+\b(?:but|however|although|though|whereas|yet|except\s+that)"
    r"\b,?\s*|,\s+(?=(?:he|she|it|they|this|that|these|those|the|his|her|its|"
    r"their|there|i|we|which|who|where|when)\b)", re.I)


def _no_claim(body: str, has_cites: bool) -> tuple[str, str]:
    """(reason, text to check): a reason when the sentence makes no claim,
    else the part of it that does ('I'm not sure, but he was born in 1564'
    -> 'he was born in 1564')."""
    low = body.lower().strip()
    plain = low.rstrip(" .!?:;,")
    if low.endswith("?"):
        if not low.startswith("did you know"):
            return "a question", ""
        body = re.sub(r"^did you know(?:\s+that)?\s*", "", body,
                      flags=re.I).rstrip(" ?")
        low = plain = body.lower()
    if _GREETING.match(plain):
        return "a greeting", ""
    if not has_cites and _OFFER.match(plain) and not re.search(r"\d", plain):
        return "an offer", ""
    if _ABSTAIN.search(low):
        kept = [c for c in _CLAUSES.split(body)
                if c and c.strip() and not _ABSTAIN.search(c.lower())]
        if not kept:
            return "says it doesn't know", ""
        return "", " ".join(c.strip() for c in kept)
    if low.endswith(":") and not has_cites and not re.search(r"\d", low):
        return "a lead-in", ""
    return "", body


# ─── one sentence ───────────────────────────────────────────────────

def _bad_cite_note(missing: list[int]) -> str:
    many = len(missing) > 1
    return (f"bad citation: there {'are' if many else 'is'} no "
            f"source{'s' if many else ''} {_refs(missing)}")


def _why_not(v: _Verdict, ns: list[int], cited: bool) -> str:
    """A plain-words note on why the sources don't back the sentence."""
    who = _refs(ns) if cited else "no source"
    plural = cited and len(ns) > 1
    if v.missing_keys:
        if cited:
            verb = "don't" if plural else "doesn't"
            return f"{who} {verb} mention {_join(v.missing_keys)}"
        return f"no source mentions {_join(v.missing_keys)}"
    if v.problem == "negation":
        return (f"{who} {'say' if plural else 'says'} otherwise "
                "(a 'not' differs)" if cited else
                "the closest source says otherwise (a 'not' differs)")
    if v.problem == "mixed":
        return (f"mixes details from {who} that no single one of them "
                "puts together")
    words = f" (nothing about {_join(v.missing_words[:3])})" \
        if v.missing_words else ""
    if cited:
        return f"{who} {'don' if plural else 'doesn'}'t say this{words}"
    return f"no source says this{words}"


def _check_sentence(sentence: str, kind: str, index: _Index,
                    vocab: set[str], aligner: _Aligner) -> SentenceCheck:
    cites = _parse_cites(sentence)
    missing = [c for c in cites if c not in index.by_n]
    if kind in ("heading", "table_header"):
        return SentenceCheck(sentence, "no_claim", cites,
                             note="a heading" if kind == "heading"
                             else "a table header")
    body = _fold(_strip_markup(sentence, table=(kind == "table_row")))
    reason, claim_text = _no_claim(body, bool(cites))
    claim = _parse_claim(claim_text, vocab) if not reason else None
    if claim is not None and not _checkable(claim):
        reason = "nothing to check"
    if reason:
        if missing:
            return SentenceCheck(sentence, "unsupported", cites,
                                 note=_bad_cite_note(missing))
        return SentenceCheck(sentence, "no_claim", cites, note=reason)

    valid = [index.by_n[c] for c in cites if c in index.by_n]
    status, note, best, verdict = "unsupported", "", None, None
    if valid:
        singles = [(it, _judge(claim, [it], aligner)) for it in valid]
        oks = [s for s in singles if s[1].ok]
        if oks:
            best, verdict = max(oks, key=lambda s: s[1].score)
            status = "supported"
            if len(valid) > 1:
                note = f"backed by [{best.ev.n}]"
        elif len(valid) > 1:
            verdict = _judge(claim, valid, aligner)
            best = max(singles, key=lambda s: s[1].score)[0]
            if verdict.ok:
                status = "supported"
                note = f"backed by {_refs([it.ev.n for it in valid])} together"
        else:
            best, verdict = singles[0]
        if status != "supported":
            note = _why_not(verdict, [it.ev.n for it in valid], cited=True)
            others = [it for it in index.items if it not in valid]
            right, rv = _best_single(claim, others, _Aligner(None))
            if right is not None and rv.ok:
                note += f"; [{right.ev.n}] does say it"
    elif index.items:
        best, verdict = _best_single(claim, index.items, aligner)
        if verdict.ok and not missing:
            status, note = "supported", f"not cited; backed by [{best.ev.n}]"
        elif verdict.ok:
            note = f"[{best.ev.n}] says it"
        else:
            note = _why_not(verdict, [], cited=False)
    else:
        note = "there were no sources to check it against"
    if missing:
        status = "unsupported"
        note = _bad_cite_note(missing) + (f"; {note}" if note else "")
    return SentenceCheck(sentence, status, cites,
                         best_evidence=best.ev.n if best else None,
                         score=verdict.score if verdict else 0.0, note=note)


# ─── public ─────────────────────────────────────────────────────────

def check_answer(answer: str, evidence: list[Evidence],
                 encoder=None) -> list[SentenceCheck]:
    """Telp's verdict on every sentence of the model's answer, in order.

    status is 'supported' (its sources contain what it says), 'unsupported'
    (with a note saying what is missing), or 'no_claim' (a greeting, an
    'I don't know', a question, a lead-in). encoder is optional: anything
    with focus_alignment(words, texts) (lattice.semantic_encoder) lets a
    sentence's words match their source by meaning as well as spelling."""
    text = _clean(answer)
    if not text:
        return []
    index = _Index(list(evidence or []))
    vocab = set(index.lower)
    for t in _tokens(_fold(text)):
        if t.raw.islower():
            vocab.add(t.base)
    aligner = _Aligner(encoder)
    return [_check_sentence(text[a:b], kind, index, vocab, aligner)
            for kind, a, b in _sentence_spans(text)]


def _mark_point(text: str, a: int, b: int) -> int:
    """Where a mark goes: after trailing citations, else before the final
    full stop ('...1650 [unverified].'), else at the end."""
    seg = text[a:b]
    if _CITE_TAIL.search(seg):
        return b
    m = re.search(r"[.!?…]+[\"'”’)\]*_]*\s*$", seg)
    return a + m.start() if m else b


def _source_line(e: Evidence) -> str:
    label = e.source or {"tool": "a tool result", "fact": "a stored fact",
                         "conversation": "this conversation"}.get(
        e.kind, "Telp's memory")
    when = f", {e.created_at[:10]}" if e.created_at else ""
    return f"[{e.n}] {label}{when}"


def annotate(answer: str, checks: list[SentenceCheck],
             evidence: list[Evidence], cite_matches: bool = True) -> str:
    """The answer as the user sees it: sentences Telp couldn't verify end
    in ' [unverified]', a supported sentence the model forgot to cite gets
    the citation of the source that backs it (cite_matches), the model's
    own source list is replaced by a 'Sources:' list of the evidence
    actually used (number, source, date), and when nothing could be
    verified the answer says so plainly."""
    text = _clean(answer)
    if not text:
        return ""
    by_n = {e.n: e for e in evidence or []}
    edits: list[tuple[int, int, str]] = []      # (start, end, replacement)
    used: list[int] = []
    cursor = 0
    for c in checks:
        at = text.find(c.sentence, cursor) if c.sentence else -1
        if at < 0:
            continue
        end = cursor = at + len(c.sentence)
        used.extend(n for n in c.cites if n in by_n and n not in used)
        if c.status == "unsupported":
            p = _mark_point(text, at, end)
            edits.append((p, p, UNVERIFIED))
        elif (c.status == "supported" and not c.cites and cite_matches
              and c.best_evidence in by_n):
            p = _mark_point(text, at, end)
            edits.append((p, p, f" [{c.best_evidence}]"))
            if c.best_evidence not in used:
                used.append(c.best_evidence)
    for u in _layout(text):
        if u.kind == "refs":
            edits.append((u.start, u.end, ""))
    for a, b, rep in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        text = text[:a] + rep + text[b:]
    body = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text).strip()

    claims = [c for c in checks if c.status != "no_claim"]
    supported = sum(c.status == "supported" for c in claims)
    parts = [body] if body else []
    if claims and not supported:
        parts.append("I couldn't verify any of this against my sources, so "
                     "please treat it with caution." if by_n else
                     "I had no sources to check this against, so none of it "
                     "is verified - please treat it with caution.")
    elif supported < len(claims):
        parts.append("(Sentences marked [unverified] aren't backed by my "
                     "sources.)")
    if used:
        parts.append("Sources:\n" + "\n".join(_source_line(by_n[n])
                                              for n in used))
    return "\n\n".join(parts)


def summarize(checks: list[SentenceCheck]) -> dict:
    """Counts for logs and the harness: how much of the answer stands."""
    claims = [c for c in checks if c.status != "no_claim"]
    supported = sum(c.status == "supported" for c in claims)
    unsupported = len(claims) - supported
    if not claims:
        verdict = "no_claims"
    elif not unsupported:
        verdict = "verified"
    elif supported:
        verdict = "partly_verified"
    else:
        verdict = "unverified"
    return {
        "sentences": len(checks),
        "claims": len(claims),
        "supported": supported,
        "unsupported": unsupported,
        "no_claim": len(checks) - len(claims),
        "supported_share": round(supported / len(claims), 3) if claims else 0.0,
        "uncited": sum(1 for c in claims if not c.cites),
        "bad_citations": sum(1 for c in checks
                             if c.note.startswith("bad citation")),
        "verdict": verdict,
        "unsupported_sentences": [c.sentence for c in claims
                                  if c.status == "unsupported"],
    }
