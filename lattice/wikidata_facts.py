"""
lattice/wikidata_facts.py - Wikidata statements as Telp facts.

Wikidata holds the infobox side of an encyclopedia as typed statements:
Q307 (Galileo Galilei) --P19 place of birth--> Q13375 (Pisa). This module
turns those statements into the Fact records of lattice/facts.py so that a
structured import merges with what Telp reads from the article itself:

  * the subject is the item's English Wikipedia title - the same name the
    article's sentences are stored under ("wikipedia:Galileo Galilei")
  * every object is an English label ("Pisa", "astronomer"), a date written
    the way a Wikipedia lead writes it ("15 February 1564", "44 BC"), or a
    number with thousands separators ("393,600")
  * the source is "wikidata:<QID>" and text is "" - a structured fact has no
    sentence behind it; the source tag is its citation

Precision over recall. A statement is dropped, never guessed, when
  - its value has no English label (Telp must never say "Q13375"),
  - it is deprecated, or marked uncertain ("circa", "disputed"),
  - a one-value relation (born_in, capital ...) has disagreeing values,
  - a date is less precise than the fact needs: a year-precision birth date
    gives born_year only - a day is never made up,
  - a present-tense relation (capital, country, currency ...) has ended.

Pure mapping, no network: tools/build_knowledge.py fetches the JSON
(wbgetentities, labels, demonyms) and hands it here.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from itertools import combinations
from typing import Any, Union

from lattice.facts import RELATIONS, Fact, norm_key

HUMAN = "Q5"                      # instance of (P31) human

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")
_MAX_DAY = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# sourcing circumstances (P1480) / nature of statement (P5102) values that
# turn a statement into a guess - not something Telp states as fact
UNCERTAIN = frozenset({
    "Q5727902",     # circa
    "Q18122778",    # presumably
    "Q56644435",    # probably
    "Q18912752",    # disputed
    "Q30230067",    # possibly
})

# P21 sex or gender -> the pronoun the sentence writer uses
PRONOUNS = {
    "Q6581097": "he",     # male
    "Q2449503": "he",     # trans man
    "Q6581072": "she",    # female
    "Q1052281": "she",    # trans woman
    "Q48270": "they",     # non-binary
}

# area units -> square kilometres
AREA_UNIT = "km2"
_TO_KM2 = {
    "Q712226": Decimal("1"),                   # square kilometre
    "Q232291": Decimal("2.589988110336"),      # square mile
    "Q35852": Decimal("0.01"),                 # hectare
    "Q25343": Decimal("0.000001"),             # square metre
    "Q81292": Decimal("0.0040468564224"),      # acre
}

# instance-of values that describe the wiki page, not the thing
_META_CLASSES = frozenset({
    "Q4167836",     # Wikimedia category
    "Q4167410",     # Wikimedia disambiguation page
    "Q13406463",    # Wikimedia list article
    "Q17442446",    # Wikimedia internal item
    "Q11266439",    # Wikimedia template
    "Q14204246",    # Wikimedia project page
})

MAX_ALIASES = 5


# ─── the property map ───────────────────────────────────────────────

@dataclass(frozen=True)
class PropertyRule:
    """How one Wikidata property becomes Telp facts.

    relation       the fact stated about the item itself (None when the
                   property only yields an inverse or a derived fact)
    kind           what the value is: "item" (another entity, shown by its
                   English label), "time", "quantity", "demonym" (a country,
                   shown as its people's adjective), "discovery" (P61: the
                   fact goes on the person, verb chosen by the item's type)
    inverse        the fact stated about the VALUE, with this item as its
                   object: Hamlet --author--> Shakespeare also gives
                   Shakespeare --wrote--> Hamlet
    year_relation  for times: the year-only fact ("born_year")
    ended          what an ended statement (end time, P582) means: "keep"
                   it; "skip" it (present-tense facts - a former capital is
                   not "the capital"); "span" (keep it, with from/to years)
    point_year     add a ("year", ...) qualifier from point in time (P585)
    earliest       for times: several values are normal and the earliest
                   counts (a work's first publication)
    """
    relation: str | None
    kind: str = "item"
    inverse: str | None = None
    year_relation: str | None = None
    ended: str = "keep"
    point_year: bool = False
    earliest: bool = False


PROPERTY_MAP: dict[str, PropertyRule] = {
    # identity
    "P31": PropertyRule("instance_of", ended="skip"),       # not for humans
    "P106": PropertyRule("occupation"),
    "P27": PropertyRule("nationality", kind="demonym"),
    # life
    "P19": PropertyRule("born_in"),
    "P569": PropertyRule("born_on", kind="time", year_relation="born_year"),
    "P20": PropertyRule("died_in"),
    "P570": PropertyRule("died_on", kind="time", year_relation="died_year"),
    "P69": PropertyRule("educated_at", ended="span"),
    "P108": PropertyRule("worked_at", ended="span"),
    "P26": PropertyRule("spouse", ended="span"),
    "P22": PropertyRule("parent"),                          # father
    "P25": PropertyRule("parent"),                          # mother
    "P40": PropertyRule("child"),
    "P166": PropertyRule("award", point_year=True),
    "P463": PropertyRule("member_of", ended="span"),        # skip for places
    # achievements and works
    "P800": PropertyRule("known_for"),
    "P50": PropertyRule("author", inverse="wrote"),
    "P577": PropertyRule(None, kind="time", year_relation="published_year",
                         earliest=True),
    "P61": PropertyRule(None, kind="discovery"),            # -> discovered /
                                                            #    invented
    "P86": PropertyRule(None, inverse="composed"),          # composer
    "P170": PropertyRule(None, inverse="painted"),          # creator, paintings
    "P178": PropertyRule(None, inverse="developed"),        # developer
    # organisations
    "P112": PropertyRule("founded_by", inverse="founded"),
    "P571": PropertyRule(None, kind="time", year_relation="founded_year"),
    "P159": PropertyRule("headquarters", ended="skip"),
    # places
    "P36": PropertyRule("capital", inverse="capital_of", ended="skip"),
    "P17": PropertyRule("country", ended="skip"),
    "P131": PropertyRule("located_in", ended="skip"),
    "P1082": PropertyRule("population", kind="quantity"),
    "P2046": PropertyRule("area", kind="quantity"),
    "P37": PropertyRule("official_language", ended="skip"),
    "P38": PropertyRule("currency", ended="skip"),
    "P361": PropertyRule("part_of", ended="skip"),
    # astronomy
    "P397": PropertyRule("orbits", ended="skip"),
}

# which item types make P61 "discoverer or inventor" a discovery and which
# an invention (matched as whole words in the item's class labels)
_DISCOVERED_WORDS = (
    "chemical element", "element", "isotope", "planet", "dwarf planet",
    "moon", "natural satellite", "comet", "asteroid", "minor planet", "star",
    "galaxy", "nebula", "exoplanet", "species", "taxon", "particle", "law",
    "effect", "phenomenon", "mineral", "chemical compound", "compound",
    "chemical substance", "molecule", "virus", "bacterium", "disease",
    "constellation", "astronomical object", "radiation", "vitamin",
    "hormone", "protein", "enzyme", "antibiotic")
_INVENTED_WORDS = (
    "device", "machine", "instrument", "tool", "apparatus", "invention",
    "engine", "vehicle", "technology", "appliance", "equipment", "weapon",
    "firearm", "software", "programming language", "game", "toy", "sport",
    "aircraft", "car", "automobile", "motor", "battery", "lamp", "camera",
    "printer", "telescope", "microscope", "computer", "telephone",
    "writing system", "alphabet", "cipher", "material")
_ORG_WORDS = (
    "organization", "organisation", "company", "business", "enterprise",
    "corporation", "university", "college", "school", "institute", "academy",
    "society", "association", "political party", "party", "club", "team",
    "band", "agency", "foundation", "city", "town", "village",
    "municipality", "country", "state", "empire", "kingdom", "republic",
    "bank", "museum", "library", "church", "league", "union", "federation",
    "orchestra", "newspaper", "publisher", "airline", "brand", "dynasty")
# properties only organisations and settlements carry
_ORG_PROPS = ("P112", "P159", "P452", "P1454", "P488", "P169", "P1128",
              "P2196", "P36", "P1082", "P749", "P355")


def _words_rx(words: Iterable[str]) -> re.Pattern:
    alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{alt})(?:s|es)?\b")


_DISCOVERED_RX = _words_rx(_DISCOVERED_WORDS)
_INVENTED_RX = _words_rx(_INVENTED_WORDS)
_ORG_RX = _words_rx(_ORG_WORDS)
_PAINTING_RX = _words_rx(("painting", "fresco", "mural"))
_SCREEN_RX = _words_rx(("film", "television", "series", "video game",
                        "anime", "episode"))


# ─── reading snaks ──────────────────────────────────────────────────

_QID = re.compile(r"^Q\d+$")
_ID_ONLY = re.compile(r"^[QPL]\d+$")          # "Q13375", "P19", "L7"
_EMBEDDED_QID = re.compile(r"\bQ\d{2,}\b")
_TIME = re.compile(r"^([+-])(\d{1,16})-(\d{2})-(\d{2})T")


def _value(snak: Any) -> Any:
    """The datavalue of a snak with a real value ('somevalue' - unknown -
    and 'novalue' snaks have none)."""
    if not isinstance(snak, dict) or snak.get("snaktype", "value") != "value":
        return None
    return (snak.get("datavalue") or {}).get("value")


def _item_id(snak: Any) -> str | None:
    """The QID an item-valued snak points at, or None."""
    v = _value(snak)
    if not isinstance(v, dict) or v.get("entity-type", "item") != "item":
        return None
    qid = v.get("id")
    if not qid and v.get("numeric-id") is not None:
        qid = f"Q{v['numeric-id']}"          # older dumps carry only this
    return qid if isinstance(qid, str) and _QID.match(qid) else None


def _rank(st: dict) -> str:
    return st.get("rank", "normal")


def _qualifiers(st: dict, pid: str) -> list[dict]:
    return (st.get("qualifiers") or {}).get(pid) or []


def _ended(st: dict) -> bool:
    """Has an end time (P582) - a known one or 'somevalue'. An explicit
    'novalue' end time means it never ended."""
    return any(s.get("snaktype", "value") != "novalue"
               for s in _qualifiers(st, "P582"))


def _uncertain(st: dict) -> bool:
    return any(_item_id(s) in UNCERTAIN
               for pid in ("P1480", "P5102") for s in _qualifiers(st, pid))


def _statements(claims: dict, pid: str, *, best: bool,
                skip_ended: bool = False) -> list[dict]:
    """The statements of one property Telp may use: never deprecated, never
    uncertain; with skip_ended, only those still true; with best, only the
    preferred ones when any are preferred (Wikidata's 'best rank')."""
    sts = [s for s in claims.get(pid) or []
           if isinstance(s, dict) and _rank(s) != "deprecated"]
    if skip_ended:
        sts = [s for s in sts if not _ended(s)]
    if best:
        preferred = [s for s in sts if _rank(s) == "preferred"]
        sts = preferred or sts
    return [s for s in sts if not _uncertain(s)]


def _item_values(claims: dict, pid: str) -> list[str]:
    """Every non-deprecated item value of a property, in order."""
    out = []
    for st in claims.get(pid) or []:
        if isinstance(st, dict) and _rank(st) != "deprecated":
            q = _item_id(st.get("mainsnak"))
            if q and q not in out:
                out.append(q)
    return out


# ─── dates ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WikiTime:
    """A Wikidata time at the precision it was stated with. BC years are
    negative (-44 is 44 BC; Wikidata's JSON has no year 0). month and day
    are None when the source didn't state them - never filled in."""
    year: int
    month: int | None = None
    day: int | None = None

    @property
    def precision(self) -> int:
        return 3 if self.day else 2 if self.month else 1

    def year_text(self) -> str:
        return f"{-self.year} BC" if self.year < 0 else str(self.year)

    def date_text(self) -> str | None:
        """'15 February 1564' - only for day-precise times."""
        if self.day is None or self.month is None:
            return None
        return f"{self.day} {MONTHS[self.month - 1]} {self.year_text()}"

    def agrees(self, other: "WikiTime") -> bool:
        """Could both describe the same moment? (1564 agrees with
        15 February 1564; 15 and 16 February 1564 do not.)"""
        if self.year != other.year:
            return False
        if self.month and other.month and self.month != other.month:
            return False
        if self.day and other.day and self.day != other.day:
            return False
        return True

    def sort_key(self) -> tuple[int, int, int]:
        return (self.year, self.month or 0, self.day or 0)


def parse_time(value: Any) -> WikiTime | None:
    """A Wikidata time value -> WikiTime, or None when it is less precise
    than a year (decade, century ...) or malformed.

    {"time": "+1564-02-15T00:00:00Z", "precision": 11} -> 15 February 1564
    {"time": "+1564-00-00T00:00:00Z", "precision": 9}  -> 1564
    {"time": "-0044-03-15T00:00:00Z", "precision": 11} -> 15 March 44 BC
    """
    if not isinstance(value, dict):
        return None
    m = _TIME.match(str(value.get("time", "")))
    if not m:
        return None
    sign, y, mo, d = m.groups()
    year = int(y)
    if year == 0:
        return None
    if sign == "-":
        year = -year
    try:
        precision = int(value.get("precision", 11))
    except (TypeError, ValueError):
        return None
    if precision < 9:                      # decade, century, millennium ...
        return None
    month = int(mo) if precision >= 10 and 1 <= int(mo) <= 12 else None
    day = None
    if precision >= 11 and month and 1 <= int(d) <= _MAX_DAY[month - 1]:
        day = int(d)
    return WikiTime(year, month, day)


def _time(snak: Any) -> WikiTime | None:
    return parse_time(_value(snak))


def _qualifier_time(st: dict, pid: str) -> WikiTime | None:
    for s in _qualifiers(st, pid):
        t = _time(s)
        if t:
            return t
    return None


def _span(st: dict) -> list[tuple[str, str]]:
    """from/to years of a statement (start time P580, end time P582)."""
    out = []
    start, end = _qualifier_time(st, "P580"), _qualifier_time(st, "P582")
    if start:
        out.append(("from", start.year_text()))
    if end:
        out.append(("to", end.year_text()))
    return out


# ─── numbers ────────────────────────────────────────────────────────

def _amount(snak: Any) -> tuple[Decimal, str] | None:
    """(amount, unit QID or "1") of a quantity snak."""
    v = _value(snak)
    if not isinstance(v, dict) or "amount" not in v:
        return None
    try:
        amount = Decimal(str(v["amount"]))
    except InvalidOperation:
        return None
    unit = str(v.get("unit") or "1").rstrip("/").rsplit("/", 1)[-1]
    return amount, unit


def format_count(n: Decimal | int) -> str:
    """390000 -> '390,000' (a whole number stays whole)."""
    n = Decimal(n)
    if n == n.to_integral_value():
        return f"{int(n):,}"
    return f"{n.normalize():,f}"


def format_area(km2: Decimal | float) -> str:
    """102775 -> '102,775 km2'; 0.49 -> '0.49 km2'."""
    x = float(km2)
    if x >= 100:
        s = f"{round(x):,}"
    elif x >= 10:
        s = f"{x:,.1f}".rstrip("0").rstrip(".")
    else:
        s = f"{x:.2f}".rstrip("0").rstrip(".")
        if s in ("", "0"):
            s = f"{x:.2g}"
    return f"{s} {AREA_UNIT}"


# ─── demonyms and names ─────────────────────────────────────────────

_PERSON_NOUN = re.compile(r"(?:er|ers|man|men|woman|women|people|folk)$")


def demonym_from_claims(claims: dict) -> str | None:
    """A country's nationality adjective from its demonym (P1549) claims:
    {"P1549": [statements]} -> "Italian". Only English values; preferred
    first; person nouns ("Icelander") and plurals ("Italians") are passed
    over - "Galileo was Icelander" would be wrong English. None when no
    English adjective is stated."""
    sts = [s for s in (claims or {}).get("P1549") or []
           if isinstance(s, dict) and _rank(s) != "deprecated"]
    sts.sort(key=lambda s: _rank(s) != "preferred")       # stable
    for st in sts:
        v = _value(st.get("mainsnak"))
        if not isinstance(v, dict) or v.get("language") != "en":
            continue
        text = str(v.get("text", "")).strip()
        if not text or not text[0].isupper() or any(c.isdigit() for c in text):
            continue
        if _PERSON_NOUN.search(text) or (text.endswith("s")
                                         and not text.endswith("ss")):
            continue
        return text
    return None


def entity_label(entity: dict) -> str | None:
    """The English label of an entity as wbgetentities returns it (with
    languages=en and languagefallback, a name-only 'mul' label may stand in
    for English)."""
    labels = (entity or {}).get("labels") or {}
    for key in ("en", "mul"):
        lab = labels.get(key)
        if isinstance(lab, dict) and lab.get("value"):
            lang = str(lab.get("language", key))
            if lang == "mul" or lang == "en" or lang.startswith("en-"):
                return str(lab["value"]).strip() or None
    return None


def enwiki_title(entity: dict) -> str | None:
    link = ((entity or {}).get("sitelinks") or {}).get("enwiki")
    if isinstance(link, dict) and link.get("title"):
        return str(link["title"])
    return None


def _base_title(title: str) -> str:
    """'Python (programming language)' -> 'Python'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip() or title


def _clean(text: Any) -> str | None:
    """A value fit to show: a non-empty, single-line name with letters or
    digits that is not, and does not contain, a raw Wikidata id."""
    if not isinstance(text, str):
        return None
    t = " ".join(text.split())
    if (not t or len(t) > 200 or _ID_ONLY.match(t)
            or _EMBEDDED_QID.search(t) or "://" in t):
        return None
    if not any(c.isalnum() for c in t):
        return None
    return t


# ─── the mapping ────────────────────────────────────────────────────

Lookup = Union[Mapping[str, Any], Callable[[str], Any], None]


def _get(source: Lookup, key: str) -> Any:
    if source is None:
        return None
    if callable(source) and not isinstance(source, Mapping):
        return source(key)
    return source.get(key)


def _entities(entities_json: Any) -> dict[str, dict]:
    """Accept a whole wbgetentities / Special:EntityData response
    ({"entities": {...}}), a QID -> entity mapping, a list of entities, or
    one entity."""
    if isinstance(entities_json, list):
        return {e.get("id", ""): e for e in entities_json
                if isinstance(e, dict)}
    if not isinstance(entities_json, dict):
        return {}
    if isinstance(entities_json.get("entities"), dict):
        return entities_json["entities"]
    if "claims" in entities_json and "id" in entities_json:
        return {entities_json["id"]: entities_json}
    return {k: v for k, v in entities_json.items() if isinstance(v, dict)}


@dataclass
class _Sink:
    """Collects facts: validated, deduplicated, in first-seen order."""
    created_at: str | None
    facts: list[Fact] = field(default_factory=list)
    _seen: set = field(default_factory=set)

    def add(self, subject: str, relation: str, obj: str, source: str,
            quals: Iterable[tuple[str, str]] = ()) -> None:
        subject, obj = _clean(subject), _clean(obj)
        if not subject or not obj or relation not in RELATIONS:
            return
        if norm_key(subject) == norm_key(obj) and relation != "alias":
            return                                 # "Iceland is in Iceland"
        f = Fact(subject, relation, obj, source, "", None, self.created_at,
                 tuple(sorted((k, v) for k, v in quals if v)))
        if f.key() in self._seen:
            return
        self._seen.add(f.key())
        self.facts.append(f)


@dataclass
class _Item:
    """One item being mapped, and the lookups it needs."""
    qid: str
    entity: dict
    subject: str
    labels: Lookup
    titles: Lookup
    demonyms: Lookup
    entities: dict[str, dict]
    sink: _Sink

    def __post_init__(self) -> None:
        self.claims: dict = self.entity.get("claims") or {}
        self.source = f"wikidata:{self.qid}"
        self.human = HUMAN in _item_values(self.claims, "P31")
        self.own = self.label(self.qid) or _base_title(self.subject)
        classes = (_item_values(self.claims, "P31")
                   + _item_values(self.claims, "P279"))
        self.classes = " | ".join(l.lower() for l in map(self.label, classes)
                                  if l)

    def label(self, qid: str) -> str | None:
        """English label of any entity: given labels first, then an entity
        in the same response."""
        lab = _clean(_get(self.labels, qid))
        if lab:
            return lab
        ent = self.entities.get(qid)
        return _clean(entity_label(ent)) if ent else None

    def name(self, qid: str) -> str | None:
        """What to call an entity as a SUBJECT: its English Wikipedia title
        (so the fact sits with that article's facts), else its label."""
        title = _clean(_get(self.titles, qid))
        if not title and qid in self.entities:
            title = _clean(enwiki_title(self.entities[qid]))
        return title or self.label(qid)

    def add(self, relation: str, obj: str, quals=(),
            subject: str | None = None) -> None:
        self.sink.add(subject or self.subject, relation, obj, self.source,
                      quals)


def facts_from_entities(entities_json: Any, labels: Lookup,
                        demonyms: Lookup, title_of: Lookup, *,
                        created_at: str | None = None) -> list[Fact]:
    """Wikidata items -> Telp facts.

    entities_json  wbgetentities output (props claims, labels, aliases,
                   sitelinks): {"entities": {QID: entity}} or the inner dict
    labels         QID -> English label for every item the claims point at
                   (missing -> that statement is skipped)
    demonyms       country QID -> nationality adjective ("Italian")
    title_of       QID -> English Wikipedia title; an item is mapped only
                   when it has one (it becomes the subject), and inverse
                   facts use it to name their subject
    Labels, demonyms and title_of may be mappings or functions.
    """
    entities = _entities(entities_json)
    sink = _Sink(created_at)
    for key, ent in entities.items():
        qid = ent.get("id") or key
        if "missing" in ent or not _QID.match(str(qid)):
            continue
        subject = _clean(_get(title_of, qid) or _get(title_of, key)
                         or enwiki_title(ent))
        if not subject:
            continue
        item = _Item(qid, ent, subject, labels, title_of, demonyms, entities,
                     sink)
        _map_item(item)
    return sink.facts


def _map_item(it: _Item) -> None:
    for pid, rule in PROPERTY_MAP.items():
        if pid not in it.claims:
            continue
        if rule.kind == "item":
            _map_items(it, pid, rule)
        elif rule.kind == "time":
            _map_time(it, pid, rule)
        elif rule.kind == "quantity":
            _map_quantity(it, pid, rule)
        elif rule.kind == "demonym":
            _map_demonyms(it, pid, rule)
        elif rule.kind == "discovery":
            _map_discovery(it, pid)
    _map_pronoun(it)
    _map_aliases(it)


def _gate(it: _Item, pid: str) -> bool:
    """Properties whose meaning depends on what the item is."""
    if pid == "P31":
        return not it.human            # "Galileo was a human" says nothing
    if pid == "P170":
        return bool(_PAINTING_RX.search(it.classes))   # creator -> painted
    if pid == "P86":                   # a film's composer scored it
        return not _SCREEN_RX.search(it.classes)
    return True


def _map_items(it: _Item, pid: str, rule: PropertyRule) -> None:
    if not _gate(it, pid):
        return
    ended = rule.ended
    if pid == "P463" and not it.human:
        ended = "skip"                 # a country's memberships: current only
    single = bool(rule.relation) and not RELATIONS[rule.relation].multi
    sts = _statements(it.claims, pid, best=single,
                      skip_ended=ended == "skip")
    pairs = [(st, q) for st in sts
             if (q := _item_id(st.get("mainsnak"))) and q != it.qid
             and not (pid == "P31" and q in _META_CLASSES)]
    if single and len({q for _, q in pairs}) != 1:
        return                         # none, or values that disagree
    done: set[str] = set()
    for st, q in pairs:
        if q in done:
            continue
        done.add(q)
        label = it.label(q)
        if not label:
            continue                   # never a raw QID: no label, no fact
        quals: list[tuple[str, str]] = []
        if ended == "span":
            quals += _span(st)
        if rule.point_year:
            t = _qualifier_time(st, "P585")
            if t:
                quals.append(("year", t.year_text()))
        if rule.relation:
            it.add(rule.relation, label, quals)
        if rule.inverse:
            who = it.name(q)
            if who:
                it.add(rule.inverse, it.own, subject=who)


def _map_time(it: _Item, pid: str, rule: PropertyRule) -> None:
    if pid == "P571" and not _org_like(it):
        return                         # a painting's "inception" isn't a founding
    if pid == "P577" and _SCREEN_RX.search(it.classes):
        return                         # films are released, not published
    sts = _statements(it.claims, pid, best=True)
    times = [t for st in sts if (t := _time(st.get("mainsnak")))]
    if not times:
        return
    if rule.earliest:
        first = min(times, key=WikiTime.sort_key)
        if rule.year_relation:
            it.add(rule.year_relation, first.year_text())
        return
    if not all(a.agrees(b) for a, b in combinations(times, 2)):
        # sources disagree on the day: the year may still be certain
        if rule.year_relation and len({t.year for t in times}) == 1:
            it.add(rule.year_relation, times[0].year_text())
        return
    best = max(times, key=lambda t: t.precision)
    date = best.date_text()
    if rule.relation and date:
        it.add(rule.relation, date)
    if rule.year_relation:
        it.add(rule.year_relation, best.year_text())


def _org_like(it: _Item) -> bool:
    if it.human:
        return False
    return (any(p in it.claims for p in _ORG_PROPS)
            or bool(_ORG_RX.search(it.classes)))


def _map_quantity(it: _Item, pid: str, rule: PropertyRule) -> None:
    # a value for part of the place (P518 "applies to part": land only,
    # the urban area ...) is not the place's own figure
    sts = [st for st in _statements(it.claims, pid, best=True)
           if not _qualifiers(st, "P518")]
    if pid == "P1082":
        _map_population(it, sts, rule)
    elif pid == "P2046":
        _map_area(it, sts, rule)


def _map_population(it: _Item, sts: list[dict], rule: PropertyRule) -> None:
    rows = []
    for st in sts:
        a = _amount(st.get("mainsnak"))
        if a and a[0] > 0 and a[0] == a[0].to_integral_value():
            rows.append((_qualifier_time(st, "P585"), a[0]))
    if not rows:
        return
    dated = [(t, n) for t, n in rows if t]
    if dated:
        latest = max(t.sort_key() for t, _ in dated)
        newest = {n for t, n in dated if t.sort_key() == latest}
        if len(newest) != 1:
            return                     # two counts for the same date
        when = next(t for t, _ in dated if t.sort_key() == latest)
        it.add(rule.relation, format_count(newest.pop()),
               [("as_of", when.year_text())])
    elif len({n for _, n in rows}) == 1:
        it.add(rule.relation, format_count(rows[0][1]))


def _map_area(it: _Item, sts: list[dict], rule: PropertyRule) -> None:
    values = []
    for st in sts:
        a = _amount(st.get("mainsnak"))
        if a and a[1] in _TO_KM2 and a[0] > 0:
            values.append(a[0] * _TO_KM2[a[1]])
    if not values:
        return
    # sources round differently (102,775 vs 103,000): within 5% they say
    # the same thing; further apart they disagree
    first = values[0]
    if all(abs(v - first) <= first * Decimal("0.05") for v in values):
        it.add(rule.relation, format_area(first))


def _map_demonyms(it: _Item, pid: str, rule: PropertyRule) -> None:
    for st in _statements(it.claims, pid, best=False):
        q = _item_id(st.get("mainsnak"))
        word = _clean(_get(it.demonyms, q)) if q else None
        if word:
            it.add(rule.relation, word)


def _discovery_verb(classes: str) -> str | None:
    found, made = _DISCOVERED_RX.search(classes), _INVENTED_RX.search(classes)
    if found and not made:
        return "discovered"
    if made and not found:
        return "invented"
    return None                        # unknown or mixed type: say nothing


def _map_discovery(it: _Item, pid: str) -> None:
    verb = _discovery_verb(it.classes)
    if not verb:
        return
    quals = []
    when = [t for st in _statements(it.claims, "P575", best=True)
            if (t := _time(st.get("mainsnak")))]
    if when and len({t.year for t in when}) == 1:
        quals.append(("year", when[0].year_text()))
    for st in _statements(it.claims, pid, best=False):
        q = _item_id(st.get("mainsnak"))
        who = it.name(q) if q else None
        if who:
            it.add(verb, it.own, quals, subject=who)


def _map_pronoun(it: _Item) -> None:
    if not it.human:
        return
    genders = {_item_id(st.get("mainsnak"))
               for st in _statements(it.claims, "P21", best=True)}
    if len(genders) == 1:
        p = PRONOUNS.get(genders.pop())
        if p:
            it.add("pronoun", p)


def _map_aliases(it: _Item) -> None:
    """Other names, kept conservative: aliases feed name resolution, so a
    code like "IS" (Iceland) must never be able to capture the word "is"."""
    names = [it.label(it.qid) or ""]
    aliases = (it.entity.get("aliases") or {})
    for key in ("en", "mul"):
        names += [a.get("value", "") for a in aliases.get(key) or []
                  if isinstance(a, dict)]
    seen = {norm_key(it.subject)}
    n = 0
    for raw in names:
        alias = _clean(raw)
        if not alias or len(alias) < 4 or len(alias) > 60:
            continue
        if alias.isupper() or not any(c.isupper() for c in alias):
            continue                   # codes ("ISL") and common nouns
        if it.human and "," in alias:
            continue                   # "Galilei, Galileo" - catalogue form
        k = norm_key(alias)
        if k in seen:
            continue
        seen.add(k)
        it.add("alias", alias)
        n += 1
        if n >= MAX_ALIASES:
            break


# ─── what to fetch next ─────────────────────────────────────────────

def referenced_ids(entities_json: Any) -> set[str]:
    """Every item the mapping will need a label (or title) for: the values
    of mapped item properties, plus the class items used to tell a
    discovery from an invention. Country demonyms are separate
    (citizenship_ids)."""
    pids = [p for p, r in PROPERTY_MAP.items()
            if r.kind in ("item", "discovery")]
    out: set[str] = set()
    for key, ent in _entities(entities_json).items():
        claims = ent.get("claims") or {}
        own = ent.get("id") or key
        extra = ["P279"] if "P61" in claims else []
        for pid in pids + extra:
            for q in _item_values(claims, pid):
                if q != own:
                    out.add(q)
    return out


def citizenship_ids(entities_json: Any) -> set[str]:
    """Countries of citizenship (P27) whose demonyms the mapping needs."""
    out: set[str] = set()
    for ent in _entities(entities_json).values():
        out.update(_item_values(ent.get("claims") or {}, "P27"))
    return out
