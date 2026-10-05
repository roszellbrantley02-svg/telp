"""
lattice/facts.py - the shared vocabulary of Telp's fact layer.

A Fact is one (subject, relation, object) statement plus where it came
from. Facts are pulled out of stored sentences (lattice/fact_extract.py)
or imported from structured sources, held in an HDC fact memory
(lattice/hdc_facts.py) that answers questions by binding/unbinding, and
turned back into fresh English sentences (mind/nlg.py) - every clause of
which can be traced to the Facts it says.

This module is the contract between those parts: the Fact record, the
relation registry (what each relation means and how it reads in English),
the query and result shapes, and name normalization. It has no heavy
dependencies on purpose.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# ─── the record ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Fact:
    """One statement: subject --relation--> obj, with provenance.

    subject   canonical entity name, e.g. "Galileo Galilei"
    relation  a key of RELATIONS, e.g. "born_in"
    obj       the value as it should be shown, e.g. "Pisa", "1564",
              "astronomer", "the University of Padua"
    source    the memory source tag, e.g. "wikipedia:Galileo Galilei",
              "user_taught", "wikidata:Q307"
    text      the stored sentence the fact was read from ("" for
              structured imports) - what "how do you know?" cites
    memory_id the lattice row id of that sentence (None for imports)
    created_at when that memory was saved (ISO string) if known
    qualifiers extra detail as sorted (key, value) pairs, e.g.
              (("as_of", "2021"),) for a population, (("from", "1592"),
              ("to", "1610")) for a teaching post
    """
    subject: str
    relation: str
    obj: str
    source: str = ""
    text: str = ""
    memory_id: int | None = None
    created_at: str | None = None
    qualifiers: tuple[tuple[str, str], ...] = ()

    def key(self) -> tuple[str, str, str]:
        """Identity for dedup: same subject, relation and value."""
        return (norm_key(self.subject), self.relation, norm_key(self.obj))

    def qualifier(self, name: str) -> str | None:
        return dict(self.qualifiers).get(name)


# ─── the relation registry ──────────────────────────────────────────
#
# Object kinds:   place, date (a full date, "15 February 1564"), year,
#                 person, org, work (a book/play/painting...), thing,
#                 class (a category noun: "astronomer", "city"),
#                 nationality (an adjective: "Italian"), number, pronoun
# Subject kinds:  person, place, org, work, thing, any
#
# `clause` is the predicate that follows the subject: "{s} {clause}".
#   {o}   the object as given
#   {a_o} the object with an indefinite article ("an astronomer")
#   {be}  "was" or "is" - the realizer picks "was" for people who have
#         died (or have a death fact) and for past-only relations; "is"
#         otherwise
# `sentence` (optional) is a standalone form when subject-first reads
#   badly ("The capital of {s} is {o}.").
# `past` = True when the clause is always past tense (born, died, wrote).

@dataclass(frozen=True)
class RelationSpec:
    name: str
    label: str                 # human label: "place of birth"
    subj_kind: str
    obj_kind: str
    multi: bool                # can a subject have several values?
    clause: str
    sentence: str = ""
    past: bool = False
    internal: bool = False     # bookkeeping only, never verbalized


def _r(name, label, subj, obj, multi, clause, sentence="", past=False,
       internal=False):
    return name, RelationSpec(name, label, subj, obj, multi, clause,
                              sentence, past, internal)


RELATIONS: dict[str, RelationSpec] = dict([
    # identity
    _r("instance_of", "kind of thing", "any", "class", True, "{be} {a_o}"),
    _r("occupation", "occupation", "person", "class", True, "{be} {a_o}"),
    _r("nationality", "nationality", "person", "nationality", True,
       "{be} {o}"),
    _r("alias", "other name", "any", "thing", True,
       "{be} also known as {o}"),
    # life
    _r("born_in", "place of birth", "person", "place", False,
       "was born in {o}", past=True),
    _r("born_on", "date of birth", "person", "date", False,
       "was born on {o}", past=True),
    _r("born_year", "year of birth", "person", "year", False,
       "was born in {o}", past=True),
    _r("died_in", "place of death", "person", "place", False,
       "died in {o}", past=True),
    _r("died_on", "date of death", "person", "date", False,
       "died on {o}", past=True),
    _r("died_year", "year of death", "person", "year", False,
       "died in {o}", past=True),
    _r("lived_in", "residence", "person", "place", True, "lived in {o}",
       past=True),
    _r("educated_at", "education", "person", "org", True,
       "studied at {o}", past=True),
    _r("taught_at", "teaching post", "person", "org", True,
       "taught at {o}", past=True),
    _r("worked_at", "employer", "person", "org", True, "worked at {o}",
       past=True),
    _r("member_of", "membership", "any", "org", True,
       "{be} a member of {o}"),
    _r("spouse", "spouse", "person", "person", True, "married {o}",
       past=True),
    _r("parent", "parent", "person", "person", True,
       "{be} the child of {o}"),
    _r("child", "child", "person", "person", True,
       "{be} the parent of {o}"),
    _r("award", "award", "any", "thing", True, "received {o}", past=True),
    # achievements and works
    _r("known_for", "known for", "any", "thing", True,
       "{be} known for {o}"),
    _r("invented", "invention", "person", "thing", True, "invented {o}",
       past=True),
    _r("discovered", "discovery", "person", "thing", True,
       "discovered {o}", past=True),
    _r("developed", "development", "any", "thing", True, "developed {o}",
       past=True),
    _r("wrote", "written work", "person", "work", True, "wrote {o}",
       past=True),
    _r("composed", "composition", "person", "work", True, "composed {o}",
       past=True),
    _r("painted", "painting", "person", "work", True, "painted {o}",
       past=True),
    _r("founded", "founded", "any", "org", True, "founded {o}", past=True),
    # organisations, works
    _r("founded_by", "founder", "org", "person", True,
       "was founded by {o}", past=True),
    _r("founded_year", "year founded", "org", "year", False,
       "was founded in {o}", past=True),
    _r("author", "author", "work", "person", True, "was written by {o}",
       past=True),
    _r("published_year", "year published", "work", "year", False,
       "was published in {o}", past=True),
    _r("headquarters", "headquarters", "org", "place", False,
       "{be} headquartered in {o}"),
    # places
    _r("capital", "capital", "place", "place", False,
       "has {o} as its capital", sentence="The capital of {s} is {o}."),
    _r("capital_of", "capital of", "place", "place", False,
       "{be} the capital of {o}"),
    _r("located_in", "location", "any", "place", True, "{be} in {o}"),
    _r("country", "country", "any", "place", False, "{be} in {o}"),
    _r("part_of", "part of", "any", "thing", True, "{be} part of {o}"),
    _r("population", "population", "place", "number", False,
       "has a population of {o}",
       sentence="{s} has a population of {o}."),
    _r("area", "area", "place", "number", False, "covers {o}"),
    _r("official_language", "official language", "place", "thing", True,
       "has {o} as an official language",
       sentence="The official language of {s} is {o}."),
    _r("currency", "currency", "place", "thing", False,
       "uses {o} as its currency"),
    # astronomy and nature
    _r("orbits", "orbits", "thing", "thing", False, "orbits {o}"),
    # bookkeeping
    _r("pronoun", "pronoun", "person", "pronoun", False, "", internal=True),
])

PERSON_RELATIONS = frozenset(n for n, r in RELATIONS.items()
                             if r.subj_kind == "person")
DEATH_RELATIONS = frozenset({"died_in", "died_on", "died_year"})


# ─── queries the fact layer answers ─────────────────────────────────

@dataclass
class Lookup:
    """"Where was Galileo born?" -> Lookup("Galileo", "born_in")"""
    subject: str
    relation: str


@dataclass
class Who:
    """"Who was born in Pisa and taught in Padua?" ->
    Who([("born_in", "Pisa"), ("taught_at", "Padua")])
    Values are the user's words; the memory matches them loosely
    ("Padua" matches "the University of Padua")."""
    constraints: list[tuple[str, str]]
    answer_kind: str = "any"        # person / place / org / work / any


@dataclass
class Describe:
    """"Tell me about Galileo" / "Who was Galileo?" """
    subject: str


@dataclass
class Analogy:
    """"Galileo is to Pisa as Newton is to what?" """
    a: str
    b: str
    c: str


@dataclass
class Chain:
    """"In what country was Galileo born?" ->
    Chain("Galileo", ["born_in", "country"])"""
    subject: str
    relations: list[str]


@dataclass
class Check:
    """"Was Galileo born in Pisa?" -> yes / no / don't know"""
    subject: str
    relation: str
    obj: str


@dataclass
class Compare:
    """"What do Galileo and Kepler have in common?" """
    a: str
    b: str


FactQuery = Lookup | Who | Describe | Analogy | Chain | Check | Compare


# ─── results ────────────────────────────────────────────────────────

@dataclass
class Answer:
    """A value the memory found, with the verified Facts behind it.
    score is the HDC similarity that surfaced it (0.5 = chance)."""
    value: str
    score: float
    facts: list[Fact] = field(default_factory=list)


@dataclass
class Match:
    """An entity found by a Who query."""
    entity: str
    score: float
    satisfied: list[Fact] = field(default_factory=list)
    missing: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Realization:
    """Generated English plus every Fact it states (for citations)."""
    text: str
    facts: list[Fact] = field(default_factory=list)


# ─── names ──────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.I)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def norm_key(s: str) -> str:
    """Matching key for names and values: case- and accent-insensitive,
    no leading article, no trailing punctuation, single spaces.
    "The University of Padua." -> "university of padua";
    "Reykjavík" -> "reykjavik"."""
    s = strip_accents(str(s)).casefold().strip()
    s = s.strip(" \t\n.,;:!?\"'()[]")
    s = _LEADING_ARTICLE.sub("", s)
    return _WS.sub(" ", s)


def with_article(noun: str) -> str:
    """"astronomer" -> "an astronomer"; leaves proper nouns and phrases
    that already start with a determiner alone."""
    n = noun.strip()
    if not n:
        return n
    if re.match(r"^(?:a|an|the|one|some|his|her|its|their)\b", n, re.I):
        return n
    if n[0].isupper() and not n.isupper():
        return n                       # proper noun: "Italian" stays bare
    # sound-based exceptions to the first-letter rule
    low = n.lower()
    if re.match(r"^(?:hour|honest|honou?r|heir)", low):
        return "an " + n
    if re.match(r"^(?:uni|use|usu|eu|one|ewe|u[bcfhjkqrst][aeiou])", low):
        return "a " + n
    return ("an " if low[0] in "aeiou" else "a ") + n
