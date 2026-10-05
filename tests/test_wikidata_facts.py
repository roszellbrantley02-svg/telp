"""Wikidata statements -> Telp facts (lattice/wikidata_facts.py).

The fixtures below are written in exactly the shape wbgetentities returns
(claims -> mainsnak -> datavalue -> value, qualifiers, ranks), built by the
small helpers at the top. FIXTURE_GALILEO_P19 is one statement spelled out
in full, as the API sends it, to keep the helpers honest.
"""
from __future__ import annotations

import copy
import re

import pytest

from lattice.facts import RELATIONS, Fact
from lattice.wikidata_facts import (PROPERTY_MAP, citizenship_ids,
                                    demonym_from_claims, entity_label,
                                    facts_from_entities, format_area,
                                    format_count, parse_time, referenced_ids)

WD = "http://www.wikidata.org/entity/"
GREGORIAN = WD + "Q1985727"
JULIAN = WD + "Q1985786"
KM2, SQ_MI = "Q712226", "Q232291"
CIRCA = "Q5727902"


# ─── wbgetentities-shaped builders ──────────────────────────────────

def snak_item(pid: str, qid: str) -> dict:
    return {"snaktype": "value", "property": pid,
            "hash": f"{pid.lower()}{qid.lower()}0d1e2f",
            "datavalue": {"value": {"entity-type": "item",
                                    "numeric-id": int(qid[1:]), "id": qid},
                          "type": "wikibase-entityid"},
            "datatype": "wikibase-item"}


def snak_time(pid: str, time: str, precision: int = 11,
              calendar: str = GREGORIAN) -> dict:
    return {"snaktype": "value", "property": pid,
            "hash": f"{pid.lower()}t{abs(hash(time)) % 10**8}",
            "datavalue": {"value": {"time": time, "timezone": 0, "before": 0,
                                    "after": 0, "precision": precision,
                                    "calendarmodel": calendar},
                          "type": "time"},
            "datatype": "time"}


def snak_quantity(pid: str, amount: str, unit: str = "1") -> dict:
    return {"snaktype": "value", "property": pid,
            "hash": f"{pid.lower()}q{amount}",
            "datavalue": {"value": {"amount": amount,
                                    "unit": unit if unit == "1" else WD + unit},
                          "type": "quantity"},
            "datatype": "quantity"}


def snak_text(pid: str, text: str, lang: str = "en") -> dict:
    return {"snaktype": "value", "property": pid, "hash": f"{pid}{text}",
            "datavalue": {"value": {"text": text, "language": lang},
                          "type": "monolingualtext"},
            "datatype": "monolingualtext"}


_N = [0]


def statement(mainsnak: dict, rank: str = "normal",
              qualifiers: list[dict] | None = None) -> dict:
    _N[0] += 1
    st = {"mainsnak": mainsnak, "type": "statement",
          "id": f"Q0${_N[0]:08d}-0000-4000-8000-000000000000", "rank": rank,
          "references": []}
    if qualifiers:
        quals: dict[str, list] = {}
        for q in qualifiers:
            quals.setdefault(q["property"], []).append(q)
        st["qualifiers"] = quals
        st["qualifiers-order"] = list(quals)
    return st


def I(pid, qid, rank="normal", quals=None):            # noqa: E743
    return statement(snak_item(pid, qid), rank, quals)


def T(pid, time, precision=11, rank="normal", quals=None, calendar=GREGORIAN):
    return statement(snak_time(pid, time, precision, calendar), rank, quals)


def Q(pid, amount, unit="1", rank="normal", quals=None):
    return statement(snak_quantity(pid, amount, unit), rank, quals)


def year(pid, y):
    """A year-precision time qualifier."""
    sign = "-" if y < 0 else "+"
    return snak_time(pid, f"{sign}{abs(y):04d}-00-00T00:00:00Z", 9)


def entity(qid: str, label: str, *statements: dict, title: str | None = None,
           aliases: tuple[str, ...] = ()) -> dict:
    claims: dict[str, list] = {}
    for st in statements:
        claims.setdefault(st["mainsnak"]["property"], []).append(
            copy.deepcopy(st))
    ent = {"pageid": int(qid[1:]) + 100, "ns": 0, "title": qid,
           "lastrevid": 2_100_000_000, "modified": "2026-09-30T12:00:00Z",
           "type": "item", "id": qid,
           "labels": {"en": {"language": "en", "value": label}},
           "descriptions": {},
           "aliases": {"en": [{"language": "en", "value": a}
                              for a in aliases]} if aliases else {},
           "claims": claims, "sitelinks": {}}
    if title:
        ent["sitelinks"]["enwiki"] = {"site": "enwiki", "title": title,
                                      "badges": []}
    return ent


# one statement exactly as wbgetentities sends it
FIXTURE_GALILEO_P19 = {
    "mainsnak": {
        "snaktype": "value", "property": "P19",
        "hash": "6a0d05e2ef8cbf7b0f5bb6c2cbf8f5d0e3b5b4a5",
        "datavalue": {"value": {"entity-type": "item", "numeric-id": 13375,
                                "id": "Q13375"},
                      "type": "wikibase-entityid"},
        "datatype": "wikibase-item"},
    "type": "statement", "id": "Q307$1f2c3a8e-4b1d-6c0a-9f3e-0a2b4c6d8e10",
    "rank": "normal",
    "references": [{"hash": "fa278ebfc458360e5aed63d5058cca83c46134f1",
                    "snaks": {"P143": [{"snaktype": "value",
                                        "property": "P143",
                                        "datavalue": {"value": {
                                            "entity-type": "item",
                                            "numeric-id": 328,
                                            "id": "Q328"},
                                            "type": "wikibase-entityid"},
                                        "datatype": "wikibase-item"}]},
                    "snaks-order": ["P143"]}],
}


# ─── the world of the fixtures ──────────────────────────────────────

LABELS = {
    "Q5": "human", "Q13375": "Pisa", "Q1957": "Arcetri",
    "Q11063": "astronomer", "Q169470": "physicist",
    "Q170790": "mathematician", "Q645663": "University of Pisa",
    "Q193510": "University of Padua", "Q358549": "Vincenzo Galilei",
    "Q18810135": "Giulia Ammannati", "Q240718": "Maria Celeste",
    "Q1066436": "Sidereus Nuncius", "Q338489": "Accademia dei Lincei",
    "Q154849": "Grand Duchy of Tuscany", "Q38": "Italy",
    "Q6581097": "male", "Q6581072": "female",
    # places
    "Q112099": "island country", "Q3624078": "sovereign state",
    "Q6256": "country", "Q1764": "Reykjavík", "Q294": "Icelandic",
    "Q131473": "Icelandic króna", "Q25417": "Danish krone",
    "Q1065": "United Nations", "Q7184": "NATO", "Q62623": "Kalmar Union",
    "Q52062": "Nordic countries", "Q189": "Iceland",
    # works, organisations
    "Q25379": "play", "Q7725634": "literary work",
    "Q692": "William Shakespeare", "Q4830453": "business",
    "Q891723": "public company", "Q19837": "Steve Jobs",
    "Q483382": "Steve Wozniak", "Q189471": "Cupertino",
    "Q30": "United States",
    # sky and science
    "Q30014": "gas giant", "Q525": "Sun", "Q11344": "chemical element",
    "Q7186": "Marie Curie", "Q37463": "Pierre Curie",
    "Q38104": "Nobel Prize in Physics", "Q44585": "Nobel Prize in Chemistry",
    "Q36": "Poland", "Q142": "France", "Q34266": "Russian Empire",
    "Q2342494": "communication device", "Q34286": "Alexander Graham Bell",
    "Q1936384": "branch of mathematics", "Q935": "Isaac Newton",
    "Q3305213": "painting", "Q762": "Leonardo da Vinci",
}
TITLES = {
    "Q307": "Galileo Galilei", "Q189": "Iceland", "Q1764": "Reykjavík",
    "Q41567": "Hamlet", "Q692": "William Shakespeare", "Q312": "Apple Inc.",
    "Q19837": "Steve Jobs", "Q319": "Jupiter", "Q1128": "Radium",
    "Q7186": "Marie Curie", "Q11035": "Telephone",
    "Q34286": "Alexander Graham Bell", "Q12418": "Mona Lisa",
    "Q762": "Leonardo da Vinci", "Q859": "Plato", "Q1048": "Julius Caesar",
}
DEMONYMS = {"Q38": "Italian", "Q36": "Polish", "Q142": "French"}

GALILEO = entity(
    "Q307", "Galileo Galilei",
    I("P31", "Q5"),
    I("P21", "Q6581097"),
    I("P106", "Q11063"), I("P106", "Q169470"), I("P106", "Q170790"),
    I("P27", "Q154849"), I("P27", "Q38"),
    FIXTURE_GALILEO_P19,
    T("P569", "+1564-02-15T00:00:00Z", 11, calendar=JULIAN),
    I("P20", "Q1957"),
    T("P570", "+1642-01-08T00:00:00Z"),
    I("P69", "Q645663", quals=[year("P580", 1581), year("P582", 1585)]),
    I("P108", "Q193510", quals=[year("P580", 1592), year("P582", 1610)]),
    I("P22", "Q358549"), I("P25", "Q18810135"), I("P40", "Q240718"),
    I("P800", "Q1066436"),
    I("P463", "Q338489", quals=[year("P580", 1611)]),
    title="Galileo Galilei",
    aliases=("Galileo", "Galilei, Galileo", "Galileo di Vincenzo Bonaiuti "
             "de' Galilei"))

ICELAND = entity(
    "Q189", "Iceland",
    I("P31", "Q112099"), I("P31", "Q3624078"), I("P31", "Q6256"),
    I("P31", "Q4167836"),                         # a wiki-page class
    I("P36", "Q1764"),
    I("P37", "Q294"),
    I("P38", "Q131473"),
    I("P38", "Q25417", quals=[year("P582", 1918)]),     # the old currency
    I("P17", "Q189"),                             # itself - says nothing
    Q("P1082", "+315556", quals=[year("P585", 2011)]),
    Q("P1082", "+359122", quals=[year("P585", 2021)]),
    Q("P1082", "+383726", rank="preferred", quals=[year("P585", 2024)]),
    Q("P1082", "+999999", rank="deprecated", quals=[year("P585", 2025)]),
    Q("P1082", "+233034", quals=[year("P585", 2024),
                                 snak_item("P518", "Q1764")]),
    Q("P2046", "+102775", KM2),
    I("P463", "Q1065", quals=[year("P580", 1946)]),
    I("P463", "Q7184"),
    I("P463", "Q62623", quals=[year("P580", 1397), year("P582", 1523)]),
    I("P361", "Q52062"),
    title="Iceland",
    aliases=("IS", "ISL", "ice", "Republic of Iceland", "🇮🇸"))

HAMLET = entity(
    "Q41567", "Hamlet",
    I("P31", "Q25379"), I("P31", "Q7725634"),
    I("P50", "Q692"),
    T("P577", "+1623-00-00T00:00:00Z", 9),
    T("P577", "+1603-00-00T00:00:00Z", 9),
    T("P571", "+1599-00-00T00:00:00Z", 9),      # written, not "founded"
    title="Hamlet")

APPLE = entity(
    "Q312", "Apple Inc.",
    I("P31", "Q4830453"), I("P31", "Q891723"),
    I("P112", "Q19837"), I("P112", "Q483382"),
    I("P112", "Q332498"),                         # no English label known
    T("P571", "+1976-04-01T00:00:00Z", 11),
    I("P159", "Q189471"),
    I("P17", "Q30"),
    title="Apple Inc.")

JUPITER = entity("Q319", "Jupiter", I("P31", "Q30014"), I("P397", "Q525"),
                 title="Jupiter")

RADIUM = entity(
    "Q1128", "radium",
    I("P31", "Q11344"),
    I("P61", "Q7186"), I("P61", "Q37463"),
    T("P575", "+1898-00-00T00:00:00Z", 9),
    title="Radium")

TELEPHONE = entity(
    "Q11035", "telephone",
    I("P279", "Q2342494"),
    I("P61", "Q34286"),
    title="Telephone")

CALCULUS = entity(
    "Q149972", "calculus", I("P31", "Q1936384"), I("P61", "Q935"),
    title="Calculus")

MONA_LISA = entity(
    "Q12418", "Mona Lisa",
    I("P31", "Q3305213"), I("P170", "Q762"),
    T("P571", "+1503-00-00T00:00:00Z", 9),
    title="Mona Lisa")

MARIE_CURIE = entity(
    "Q7186", "Marie Curie",
    I("P31", "Q5"), I("P21", "Q6581072"),
    I("P27", "Q36"), I("P27", "Q142"), I("P27", "Q34266"),
    I("P166", "Q38104", quals=[year("P585", 1903)]),
    I("P166", "Q44585", quals=[year("P585", 1911)]),
    I("P26", "Q37463", quals=[year("P580", 1895), year("P582", 1906)]),
    title="Marie Curie")

PLATO = entity(
    "Q859", "Plato", I("P31", "Q5"),
    T("P569", "-0428-00-00T00:00:00Z", 9),
    T("P570", "-0348-00-00T00:00:00Z", 9),
    title="Plato")

CAESAR = entity(
    "Q1048", "Julius Caesar", I("P31", "Q5"),
    T("P569", "-0100-07-12T00:00:00Z", 11, calendar=JULIAN,
      quals=[snak_item("P1480", CIRCA)]),               # "c. 12 July 100 BC"
    T("P570", "-0044-03-15T00:00:00Z", 11, calendar=JULIAN),
    title="Julius Caesar")

ALL_ENTITIES = {e["id"]: e for e in (
    GALILEO, ICELAND, HAMLET, APPLE, JUPITER, RADIUM, TELEPHONE, CALCULUS,
    MONA_LISA, MARIE_CURIE, PLATO, CAESAR)}


def wbgetentities(*ents: dict) -> dict:
    """A wbgetentities response body."""
    return {"entities": {e["id"]: e for e in ents}, "success": 1}


def facts_of(*ents: dict, labels=None, demonyms=None, titles=None
             ) -> list[Fact]:
    return facts_from_entities(
        wbgetentities(*ents), LABELS if labels is None else labels,
        DEMONYMS if demonyms is None else demonyms,
        TITLES if titles is None else titles)


def triples(facts: list[Fact]) -> set[tuple[str, str, str]]:
    return {(f.subject, f.relation, f.obj) for f in facts}


def values(facts: list[Fact], subject: str, relation: str) -> list[str]:
    return [f.obj for f in facts
            if f.subject == subject and f.relation == relation]


# ─── the property map ───────────────────────────────────────────────

REQUIRED = {
    "P31": "instance_of", "P106": "occupation", "P27": "nationality",
    "P19": "born_in", "P569": "born_on", "P20": "died_in",
    "P570": "died_on", "P69": "educated_at", "P108": "worked_at",
    "P26": "spouse", "P22": "parent", "P25": "parent", "P40": "child",
    "P166": "award", "P800": "known_for", "P50": "author",
    "P112": "founded_by", "P159": "headquarters", "P36": "capital",
    "P17": "country", "P131": "located_in", "P1082": "population",
    "P2046": "area", "P37": "official_language", "P38": "currency",
    "P397": "orbits", "P463": "member_of", "P361": "part_of",
}


def test_property_map_covers_the_required_properties():
    for pid, rel in REQUIRED.items():
        assert PROPERTY_MAP[pid].relation == rel, pid
    assert PROPERTY_MAP["P569"].year_relation == "born_year"
    assert PROPERTY_MAP["P570"].year_relation == "died_year"
    assert PROPERTY_MAP["P577"].year_relation == "published_year"
    assert PROPERTY_MAP["P571"].year_relation == "founded_year"
    assert PROPERTY_MAP["P50"].inverse == "wrote"
    assert PROPERTY_MAP["P112"].inverse == "founded"
    assert PROPERTY_MAP["P36"].inverse == "capital_of"
    assert PROPERTY_MAP["P61"].kind == "discovery"
    for rule in PROPERTY_MAP.values():         # only registered relations
        for rel in (rule.relation, rule.inverse, rule.year_relation):
            assert rel is None or rel in RELATIONS


def test_every_fact_is_well_formed_and_cites_its_item():
    facts = facts_of(*ALL_ENTITIES.values())
    assert facts
    raw_id = re.compile(r"\b[QPL]\d+\b")
    for f in facts:
        assert f.relation in RELATIONS
        assert re.fullmatch(r"wikidata:Q\d+", f.source)
        assert f.text == "" and f.memory_id is None
        assert f.subject and f.obj
        for v in (f.subject, f.obj, *(v for _, v in f.qualifiers)):
            assert not raw_id.search(v), f
        assert list(f.qualifiers) == sorted(f.qualifiers)
    assert len({f.key() for f in facts}) == len(facts)      # no duplicates


# ─── people ─────────────────────────────────────────────────────────

def test_a_person():
    facts = facts_of(GALILEO)
    t = triples(facts)
    g = "Galileo Galilei"
    for rel, obj in [
        ("occupation", "astronomer"), ("occupation", "physicist"),
        ("occupation", "mathematician"), ("nationality", "Italian"),
        ("born_in", "Pisa"), ("born_on", "15 February 1564"),
        ("born_year", "1564"), ("died_in", "Arcetri"),
        ("died_on", "8 January 1642"), ("died_year", "1642"),
        ("educated_at", "University of Pisa"),
        ("worked_at", "University of Padua"),
        ("parent", "Vincenzo Galilei"), ("parent", "Giulia Ammannati"),
        ("child", "Maria Celeste"), ("known_for", "Sidereus Nuncius"),
        ("member_of", "Accademia dei Lincei"), ("pronoun", "he"),
    ]:
        assert (g, rel, obj) in t, (rel, obj)
    # humans get no "Galileo was a human"; the citizenship country without a
    # demonym (Grand Duchy of Tuscany) gives no nationality
    assert values(facts, g, "instance_of") == []
    assert values(facts, g, "nationality") == ["Italian"]
    assert all(f.source == "wikidata:Q307" for f in facts)
    # spans become from/to qualifiers
    post = next(f for f in facts if f.relation == "worked_at")
    assert post.qualifiers == (("from", "1592"), ("to", "1610"))
    assert post.qualifier("from") == "1592"
    study = next(f for f in facts if f.relation == "educated_at")
    assert dict(study.qualifiers) == {"from": "1581", "to": "1585"}


def test_awards_carry_their_year_and_spouses_their_span():
    facts = facts_of(MARIE_CURIE)
    awards = {f.obj: f.qualifier("year") for f in facts
              if f.relation == "award"}
    assert awards == {"Nobel Prize in Physics": "1903",
                      "Nobel Prize in Chemistry": "1911"}
    spouse = next(f for f in facts if f.relation == "spouse")
    assert spouse.obj == "Pierre Curie"
    assert spouse.qualifiers == (("from", "1895"), ("to", "1906"))
    # several citizenships: one nationality per known demonym
    assert sorted(values(facts, "Marie Curie", "nationality")) == \
        ["French", "Polish"]
    assert values(facts, "Marie Curie", "pronoun") == ["she"]


def test_aliases_are_conservative():
    g = values(facts_of(GALILEO), "Galileo Galilei", "alias")
    assert g == ["Galileo", "Galileo di Vincenzo Bonaiuti de' Galilei"]
    # codes ("IS" would capture the word "is"), common nouns and symbols
    # never become names
    assert values(facts_of(ICELAND), "Iceland", "alias") == \
        ["Republic of Iceland"]


# ─── dates ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("time,precision,expected", [
    ("+1564-02-15T00:00:00Z", 11, ("15 February 1564", "1564")),
    ("+1564-02-00T00:00:00Z", 10, (None, "1564")),        # month only
    ("+1564-00-00T00:00:00Z", 9, (None, "1564")),
    ("+1564-01-01T00:00:00Z", 9, (None, "1564")),         # never a made-up day
    ("+1560-00-00T00:00:00Z", 8, None),                   # a decade
    ("+1500-00-00T00:00:00Z", 7, None),                   # a century
    ("-0044-03-15T00:00:00Z", 11, ("15 March 44 BC", "44 BC")),
    ("-0428-00-00T00:00:00Z", 9, (None, "428 BC")),
    ("+0800-12-25T00:00:00Z", 11, ("25 December 800", "800")),
    ("+1564-02-31T00:00:00Z", 11, (None, "1564")),        # no such day
    ("+0000-00-00T00:00:00Z", 9, None),                   # no year 0
])
def test_parse_time(time, precision, expected):
    t = parse_time({"time": time, "precision": precision,
                    "calendarmodel": GREGORIAN})
    if expected is None:
        assert t is None
    else:
        assert (t.date_text(), t.year_text()) == expected


def person(*statements):
    return entity("Q42", "Ada Example", I("P31", "Q5"), *statements,
                  title="Ada Example")


def born(facts):
    return (values(facts, "Ada Example", "born_on"),
            values(facts, "Ada Example", "born_year"))


def test_year_precision_gives_the_year_only():
    assert born(facts_of(person(T("P569", "+1815-00-00T00:00:00Z", 9)))) == \
        ([], ["1815"])
    assert born(facts_of(person(T("P569", "+1815-12-00T00:00:00Z", 10)))) == \
        ([], ["1815"])
    assert born(facts_of(person(T("P569", "+1810-00-00T00:00:00Z", 8)))) == \
        ([], [])


def test_bc_dates():
    facts = facts_of(PLATO, CAESAR)
    assert values(facts, "Plato", "born_year") == ["428 BC"]
    assert values(facts, "Plato", "born_on") == []
    assert values(facts, "Plato", "died_year") == ["348 BC"]
    assert values(facts, "Julius Caesar", "died_on") == ["15 March 44 BC"]
    assert values(facts, "Julius Caesar", "died_year") == ["44 BC"]
    # "c. 12 July 100 BC" is a guess: not stated
    assert values(facts, "Julius Caesar", "born_on") == []
    assert values(facts, "Julius Caesar", "born_year") == []


def test_disagreeing_dates():
    # two days in the same year: the year is certain, the day is not
    same_year = person(T("P569", "+1815-12-10T00:00:00Z"),
                       T("P569", "+1815-12-11T00:00:00Z"))
    assert born(facts_of(same_year)) == ([], ["1815"])
    # Old Style / New Style across a new year: even the year is unsure
    old_new = person(T("P569", "+1642-12-25T00:00:00Z", calendar=JULIAN),
                     T("P569", "+1643-01-04T00:00:00Z"))
    assert born(facts_of(old_new)) == ([], [])
    # a year-only value agrees with a full date: keep the full date
    agree = person(T("P569", "+1815-00-00T00:00:00Z", 9),
                   T("P569", "+1815-12-10T00:00:00Z"))
    assert born(facts_of(agree)) == (["10 December 1815"], ["1815"])
    # a preferred value settles it
    preferred = person(T("P569", "+1815-12-10T00:00:00Z", rank="preferred"),
                       T("P569", "+1816-01-04T00:00:00Z"))
    assert born(facts_of(preferred)) == (["10 December 1815"], ["1815"])


def test_deprecated_statements_are_ignored():
    ent = person(T("P569", "+1815-12-11T00:00:00Z", rank="deprecated"),
                 T("P569", "+1815-00-00T00:00:00Z", 9),
                 I("P19", "Q13375", rank="deprecated"))
    facts = facts_of(ent)
    assert born(facts) == ([], ["1815"])
    assert values(facts, "Ada Example", "born_in") == []


def test_single_valued_relations_need_one_answer():
    pick = person(I("P19", "Q13375"), I("P19", "Q1957", rank="preferred"))
    assert values(facts_of(pick), "Ada Example", "born_in") == ["Arcetri"]
    clash = person(I("P19", "Q13375"), I("P19", "Q1957"))
    assert values(facts_of(clash), "Ada Example", "born_in") == []
    # multi-valued relations keep every non-deprecated value
    multi = person(I("P106", "Q11063"), I("P106", "Q169470", "preferred"),
                   I("P106", "Q170790", "deprecated"))
    assert sorted(values(facts_of(multi), "Ada Example", "occupation")) == \
        ["astronomer", "physicist"]


# ─── places ─────────────────────────────────────────────────────────

def test_a_country():
    facts = facts_of(ICELAND)
    t = triples(facts)
    assert sorted(values(facts, "Iceland", "instance_of")) == \
        ["country", "island country", "sovereign state"]
    assert ("Iceland", "capital", "Reykjavík") in t
    assert ("Reykjavík", "capital_of", "Iceland") in t      # the inverse
    assert ("Iceland", "official_language", "Icelandic") in t
    # the krone ended in 1918: only the current currency
    assert values(facts, "Iceland", "currency") == ["Icelandic króna"]
    assert values(facts, "Iceland", "country") == []        # itself
    assert ("Iceland", "part_of", "Nordic countries") in t
    # a place's memberships: current ones only, no Kalmar Union
    assert sorted(values(facts, "Iceland", "member_of")) == \
        ["NATO", "United Nations"]
    inverse = next(f for f in facts if f.relation == "capital_of")
    assert inverse.source == "wikidata:Q189"


def test_population_is_the_latest_count_with_its_date():
    facts = facts_of(ICELAND)
    pop = [f for f in facts if f.relation == "population"]
    # preferred 2024 count; not the deprecated 2025 one, not the 2024 count
    # that applies only to part of the country (P518)
    assert len(pop) == 1
    assert pop[0].obj == "383,726"
    assert pop[0].qualifiers == (("as_of", "2024"),)


def test_population_without_preferred_rank_takes_the_newest():
    town = entity("Q900", "Example Town",
                  Q("P1082", "+1200", quals=[year("P585", 1990)]),
                  Q("P1082", "+5400", quals=[year("P585", 2020)]),
                  Q("P1082", "+3100", quals=[year("P585", 2001)]),
                  title="Example Town")
    pop = [f for f in facts_of(town) if f.relation == "population"]
    assert [(f.obj, f.qualifier("as_of")) for f in pop] == [("5,400", "2020")]


def test_area_in_square_kilometres():
    facts = facts_of(ICELAND)
    assert values(facts, "Iceland", "area") == ["102,775 km2"]
    miles = entity("Q901", "Example Island", Q("P2046", "+100", SQ_MI),
                   title="Example Island")
    assert values(facts_of(miles), "Example Island", "area") == ["259 km2"]
    # far-apart figures disagree: no area
    clash = entity("Q902", "Example Lake", Q("P2046", "+10", KM2),
                   Q("P2046", "+30", KM2), title="Example Lake")
    assert values(facts_of(clash), "Example Lake", "area") == []


def test_number_formats():
    assert format_count(390000) == "390,000"
    assert format_count(7) == "7"
    assert format_area(0.49) == "0.49 km2"
    assert format_area(12.5) == "12.5 km2"


# ─── works, organisations, discoveries ──────────────────────────────

def test_a_work_and_its_author():
    facts = facts_of(HAMLET)
    t = triples(facts)
    assert ("Hamlet", "author", "William Shakespeare") in t
    assert ("William Shakespeare", "wrote", "Hamlet") in t
    assert values(facts, "Hamlet", "published_year") == ["1603"]  # earliest
    assert values(facts, "Hamlet", "founded_year") == []     # not an org
    assert values(facts, "Hamlet", "instance_of") == ["play", "literary work"]


def test_an_organisation():
    facts = facts_of(APPLE)
    t = triples(facts)
    assert ("Apple Inc.", "founded_by", "Steve Jobs") in t
    assert ("Apple Inc.", "founded_by", "Steve Wozniak") in t
    assert ("Steve Jobs", "founded", "Apple Inc.") in t
    # no enwiki title known for Wozniak here: his label names him
    assert ("Steve Wozniak", "founded", "Apple Inc.") in t
    assert values(facts, "Apple Inc.", "founded_year") == ["1976"]
    assert values(facts, "Apple Inc.", "headquarters") == ["Cupertino"]
    assert values(facts, "Apple Inc.", "country") == ["United States"]
    # Q332498 has no English label: that founder is left out entirely
    assert len(values(facts, "Apple Inc.", "founded_by")) == 2


def test_orbits():
    assert values(facts_of(JUPITER), "Jupiter", "orbits") == ["Sun"]
    assert values(facts_of(JUPITER), "Jupiter", "instance_of") == \
        ["gas giant"]


def test_discoverer_or_inventor_goes_on_the_person():
    facts = facts_of(RADIUM, TELEPHONE, CALCULUS)
    t = triples(facts)
    assert ("Marie Curie", "discovered", "radium") in t
    assert ("Pierre Curie", "discovered", "radium") in t
    found = next(f for f in facts if f.subject == "Marie Curie")
    assert found.qualifiers == (("year", "1898"),)
    assert found.source == "wikidata:Q1128"
    # a subclass of "communication device" is invented, not discovered
    assert ("Alexander Graham Bell", "invented", "telephone") in t
    # a branch of mathematics: neither word fits, so nothing is said
    assert not [f for f in facts if f.obj == "calculus"]


def test_creator_of_a_painting_painted_it():
    facts = facts_of(MONA_LISA)
    assert ("Leonardo da Vinci", "painted", "Mona Lisa") in triples(facts)
    assert values(facts, "Mona Lisa", "founded_year") == []


def test_composer_but_not_of_a_film_score():
    symphony = entity("Q903", "Symphony No. 9", I("P86", "Q762"),
                      title="Symphony No. 9 (Example)")
    film = entity("Q904", "Example Film", I("P31", "Q11424"),
                  I("P86", "Q762"), title="Example Film")
    labels = dict(LABELS, Q11424="film")
    facts = facts_of(symphony, film, labels=labels)
    assert ("Leonardo da Vinci", "composed", "Symphony No. 9") in \
        triples(facts)
    assert not [f for f in facts if f.obj == "Example Film"]


# ─── precision guards ───────────────────────────────────────────────

def test_missing_labels_mean_no_fact_never_a_raw_id():
    facts = facts_of(GALILEO, ICELAND, labels={})
    rels = {f.relation for f in facts}
    # with no labels, only facts that need none survive: dates, numbers,
    # the demonym, the pronoun and names
    assert rels <= {"born_on", "born_year", "died_on", "died_year",
                    "population", "area", "nationality", "pronoun",
                    "alias"}
    assert "born_in" not in rels and "capital" not in rels
    for f in facts:
        assert not re.search(r"\bQ\d+\b", f.subject + " " + f.obj)
    # a label that is itself an id is not a label
    sneaky = dict(LABELS, Q13375="Q13375")
    assert values(facts_of(GALILEO, labels=sneaky), "Galileo Galilei",
                  "born_in") == []


def test_items_without_an_english_wikipedia_title_are_not_subjects():
    orphan = copy.deepcopy(JUPITER)
    orphan["sitelinks"] = {}
    assert facts_of(orphan, titles={}) == []
    # the sitelink in the entity itself is enough
    assert values(facts_of(JUPITER, titles={}), "Jupiter", "orbits") == \
        ["Sun"]


def test_title_not_label_is_the_subject():
    ent = entity("Q28865", "Python", I("P31", "Q9143"),
                 title="Python (programming language)")
    facts = facts_of(ent, labels=dict(LABELS, Q9143="programming language"),
                     titles={"Q28865": "Python (programming language)"})
    assert ("Python (programming language)", "instance_of",
            "programming language") in triples(facts)
    assert ("Python (programming language)", "alias", "Python") in \
        triples(facts)


def test_input_shapes_and_missing_entities():
    whole = {"entities": {"Q319": JUPITER,
                          "Q99999999": {"id": "Q99999999", "missing": ""}},
             "success": 1}
    a = facts_from_entities(whole, LABELS, DEMONYMS, TITLES)
    b = facts_from_entities([JUPITER], LABELS, DEMONYMS, TITLES)
    c = facts_from_entities(JUPITER, LABELS, DEMONYMS, TITLES.get)
    d = facts_from_entities({"Q319": JUPITER}, LABELS.get, DEMONYMS, TITLES)
    assert a == b == c == d and a


def test_created_at_is_carried():
    facts = facts_from_entities(wbgetentities(JUPITER), LABELS, DEMONYMS,
                                TITLES, created_at="2026-10-05T12:00:00+00:00")
    assert {f.created_at for f in facts} == {"2026-10-05T12:00:00+00:00"}


# ─── what the builder fetches next ──────────────────────────────────

def test_referenced_ids_and_citizenships():
    refs = referenced_ids(wbgetentities(GALILEO, TELEPHONE))
    assert {"Q13375", "Q1957", "Q11063", "Q645663", "Q193510", "Q358549",
            "Q34286", "Q2342494"} <= refs
    assert "Q154849" not in refs           # citizenship: demonyms instead
    assert "Q307" not in refs and "Q6581097" not in refs
    deprecated = person(I("P19", "Q13375", rank="deprecated"))
    assert "Q13375" not in referenced_ids({"Q42": deprecated})
    assert citizenship_ids(wbgetentities(GALILEO, MARIE_CURIE)) == \
        {"Q154849", "Q38", "Q36", "Q142", "Q34266"}


def test_demonyms():
    iceland = {"P1549": [statement(snak_text("P1549", "Icelander")),
                         statement(snak_text("P1549", "Íslendingur", "is")),
                         statement(snak_text("P1549", "Icelandic"))]}
    assert demonym_from_claims(iceland) == "Icelandic"
    italy = {"P1549": [statement(snak_text("P1549", "Italians")),
                       statement(snak_text("P1549", "Italian"),
                                 rank="preferred")]}
    assert demonym_from_claims(italy) == "Italian"
    swiss = {"P1549": [statement(snak_text("P1549", "Swiss"))]}
    assert demonym_from_claims(swiss) == "Swiss"
    nouns_only = {"P1549": [statement(snak_text("P1549", "Icelander"))]}
    assert demonym_from_claims(nouns_only) is None
    assert demonym_from_claims({}) is None


def test_entity_label_prefers_english_and_accepts_mul():
    assert entity_label({"labels": {"en": {"language": "en",
                                           "value": "Pisa"}}}) == "Pisa"
    # languagefallback: a name-only 'mul' label served for English
    assert entity_label({"labels": {"en": {"language": "mul",
                                           "for-language": "en",
                                           "value": "Ada Lovelace"}}}) == \
        "Ada Lovelace"
    assert entity_label({"labels": {"de": {"language": "de",
                                           "value": "Pisa"}}}) is None
    assert entity_label({}) is None
