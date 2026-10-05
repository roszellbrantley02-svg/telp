"""The HDC fact memory: hypervectors answer, stored facts verify and cite.

Facts are hand-built from tests/fact_corpus.py EXPECTED (no extractor
involved). Besides correctness on that small encyclopedia, these tests pin
the properties the design rests on: loose matching that never overreaches,
sharding that keeps recall perfect for a big entity, accuracy and speed at
5,000 entities, and - by damaging records - that answers really come from
the hypervectors, with the symbolic index only confirming them.
"""
from __future__ import annotations

import random
import time
from collections import defaultdict

import numpy as np
import pytest

import lattice.hdc_facts as hdc
from lattice.facts import Fact, norm_key
from lattice.hdc_facts import FactMemory, loose_match
from tests.fact_corpus import EXPECTED, FORBIDDEN

SUBJECTS = sorted({s for s, _, _ in EXPECTED})


def corpus_facts() -> list[Fact]:
    """EXPECTED as Facts; each subject's article is one 'memory'."""
    return [Fact(s, r, o, source=f"wikipedia:{s}",
                 memory_id=100 + SUBJECTS.index(s))
            for s, r, o in EXPECTED]


@pytest.fixture
def mem() -> FactMemory:
    m = FactMemory()
    assert m.add_many(corpus_facts()) == len(EXPECTED)
    return m


def triples(facts) -> list[tuple[str, str, str]]:
    return [(f.subject, f.relation, f.obj) for f in facts]


def values(answers) -> set[str]:
    return {norm_key(a.value) for a in answers}


def flip_bits(mem: FactMemory, entity: str, fraction: float,
              seed: int = 0) -> None:
    """Damage an entity's record(s) in place: flip `fraction` of the bits."""
    mem._sync()
    rng = np.random.default_rng(seed)
    for row in mem._entity_rows[norm_key(entity)]:
        mask = np.zeros(mem.n_words * 8, dtype=np.uint8)
        packed = np.packbits(rng.random(mem.dim) < fraction)
        mask[:len(packed)] = packed
        mem._records.words[row] ^= mask.view(np.uint64)


# ─── representation ─────────────────────────────────────────────────

def test_atoms_are_deterministic_balanced_and_near_orthogonal():
    a, b = FactMemory(), FactMemory()
    x = a._rel_atom("born_in")
    assert np.array_equal(x, b._rel_atom("born_in"))       # nothing stored
    assert np.array_equal(hdc._make_atom("value", "pisa", a.dim, a.n_words),
                          a._value_vec("pisa"))
    bits = np.unpackbits(x.view(np.uint8))
    assert bits[a.dim:].sum() == 0                          # pad bits clear
    assert abs(bits[:a.dim].mean() - 0.5) < 6 * a.sigma
    # independent atoms agree on ~half their bits: within 6 sigma of 0.5
    bank = hdc._Bank(a.dim, a.n_words)
    for i in range(50):
        bank.put(a._value_vec(f"value {i}"), i)
    sims = bank.similarity(x)
    assert np.all(np.abs(sims - 0.5) < 6 * a.sigma)
    assert bank.similarity(a._value_vec("value 7"))[7] == 1.0


def test_records_are_packed_bits(mem):
    s = mem.stats()
    assert s["bytes_per_record"] == 1256          # 1,250 bytes + word pad
    assert mem._records.words.dtype == np.uint64
    assert s["records"] == s["entities"] == len(SUBJECTS)
    assert s["accept_similarity"] == pytest.approx(0.53)
    assert s["shard_size"] == 48


def test_popcount_table_fallback_matches(monkeypatch, mem):
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2 ** 63, size=(40, 157), dtype=np.uint64)
    assert np.array_equal(hdc._popcount_rows_table(x),
                          hdc._popcount_rows(x))
    want = mem.lookup("Galileo", "born_in")
    monkeypatch.setattr(hdc, "_popcount_rows", hdc._popcount_rows_table)
    got = mem.lookup("Galileo", "born_in")
    assert [(a.value, a.score) for a in got] == \
        [(a.value, a.score) for a in want]


def test_bundle_keeps_items_at_the_predicted_similarity():
    m = FactMemory()
    for k in (5, 21, 47):
        vecs = np.stack([m._value_vec(f"item {k} {i}") for i in range(k)])
        rec = m._bundle(vecs, None)
        bank = hdc._Bank(m.dim, m.n_words)
        for i, v in enumerate(vecs):
            bank.put(v, i)
        sims = bank.similarity(rec)
        expected = 0.5 + 0.4 / np.sqrt(k)
        assert abs(sims.mean() - expected) < 0.015, k
        assert sims.min() > m.accept                 # every item recoverable


# ─── names ──────────────────────────────────────────────────────────

def test_resolve_names(mem):
    assert mem.resolve("galileo galilei") == "Galileo Galilei"
    assert mem.resolve("Galileo") == "Galileo Galilei"        # first name
    assert mem.resolve("Newton") == "Isaac Newton"            # surname
    assert mem.resolve("the University of Padua") == "University of Padua"
    assert mem.resolve("REYKJAVIK") is None                   # only a value
    assert mem.resolve("Stadt") is None       # name parts: people only
    assert mem.resolve("") is None
    assert mem.resolve("Curie") == "Marie Curie"
    mem.add(Fact("Pierre Curie", "born_in", "Paris"))
    assert mem.resolve("Curie") is None                        # ambiguous
    assert mem.resolve("Pierre") == "Pierre Curie"


def test_resolve_disambiguator_and_alias_facts():
    m = FactMemory()
    m.add(Fact("Mercury (planet)", "orbits", "the Sun"))
    m.add(Fact("Galileo Galilei", "alias", "Galileo di Vincenzo Bonaiuti "
               "de' Galilei"))
    m.add(Fact("Galileo Galilei", "born_in", "Pisa"))
    assert m.resolve("Mercury") == "Mercury (planet)"
    assert m.resolve("galileo di vincenzo bonaiuti de' galilei") == \
        "Galileo Galilei"
    m.add(Fact("Mercury (element)", "instance_of", "chemical element"))
    assert m.resolve("Mercury") is None                        # ambiguous


# ─── lookup ─────────────────────────────────────────────────────────

def test_lookup_finds_every_expected_fact(mem):
    want = defaultdict(set)
    for s, r, o in EXPECTED:
        want[(s, r)].add(norm_key(o))
    for (s, r), objs in want.items():
        got = mem.lookup(s, r, k=10)
        assert values(got) == objs, (s, r)
        for a in got:
            assert a.score >= mem.accept
            assert all(f.subject == s and f.relation == r for f in a.facts)
            assert a.facts[0].source == f"wikipedia:{s}"
            assert a.facts[0].memory_id == 100 + SUBJECTS.index(s)


def test_lookup_unknowns_return_nothing(mem):
    assert mem.lookup("Leonardo da Vinci", "born_in") == []
    assert mem.lookup("Galileo", "no_such_relation") == []
    assert mem.lookup("Galileo", "capital") == []     # relation he lacks
    assert mem.lookup("Galileo", "born_in", k=0) == []


def test_lookup_reads_inverse_facts(mem):
    """"Italy capital Rome" also answers "what is Rome the capital of?" -
    Rome has no facts of its own, so this is a scan for bind(capital,
    ROME) across every record."""
    [a] = mem.lookup("Rome", "capital_of")
    assert a.value == "Italy" and a.score >= mem.accept
    assert triples(a.facts) == [("Italy", "capital", "Rome")]
    assert mem.check("Warsaw", "capital_of", "Poland")[0] is True
    [w] = mem.lookup("Hamlet", "author")
    assert w.value == "William Shakespeare"
    [born] = mem.chain("Hamlet", ["author", "born_in"])
    assert born.value == "Stratford-upon-Avon"
    assert triples(born.facts) == [
        ("William Shakespeare", "wrote", "Hamlet"),
        ("William Shakespeare", "born_in", "Stratford-upon-Avon")]
    # stored both ways: one answer citing both facts
    mem.add(Fact("Rome", "capital_of", "Italy"))
    [both] = mem.lookup("Rome", "capital_of")
    assert triples(both.facts) == [("Rome", "capital_of", "Italy"),
                                   ("Italy", "capital", "Rome")]


def test_values_named_by_alias_or_surname(mem):
    mem.add(Fact("Hamlet", "author", "William Shakespeare"))
    mem.add(Fact("Macbeth", "author", "William Shakespeare"))
    found = mem.who([("author", "Shakespeare")], k=10)
    assert {m.entity for m in found} == {"Hamlet", "Macbeth"}
    assert mem.check("Hamlet", "author", "Shakespeare")[0] is True
    [a] = mem.analogy("Hamlet", "Shakespeare", "Macbeth")
    assert a.value == "William Shakespeare"


def test_forbidden_misreadings_never_come_back(mem):
    for s, r, o in FORBIDDEN:
        assert norm_key(o) not in values(mem.lookup(s, r, k=20))
        assert mem.check(s, r, o)[0] is not True


def test_duplicate_and_empty_facts_are_ignored(mem):
    n = len(mem)
    assert mem.add(corpus_facts()[0]) is False
    assert mem.add(Fact("Galileo Galilei", "born_in", "  ")) is False
    assert len(mem) == n
    with pytest.raises(TypeError):
        mem.add(("Galileo", "born_in", "Pisa"))
    # the same claim from a second source: one record item, two citations
    assert mem.add(Fact("Galileo Galilei", "born_in", "Pisa",
                        source="user_taught"))
    [ans] = mem.lookup("Galileo", "born_in")
    assert {f.source for f in ans.facts} == {"wikipedia:Galileo Galilei",
                                             "user_taught"}


# ─── who ────────────────────────────────────────────────────────────

def test_who_with_loose_values(mem):
    [top, *_] = mem.who([("born_in", "Pisa"), ("taught_at", "Padua")])
    assert top.entity == "Galileo Galilei" and top.missing == []
    assert ("Galileo Galilei", "taught_at", "University of Padua") in \
        triples(top.satisfied)
    [ice] = mem.who([("capital", "Reykjavik")])               # accents
    assert ice.entity == "Iceland"
    assert triples(ice.satisfied) == [("Iceland", "capital", "Reykjavík")]
    found = mem.who([("educated_at", "Padua")])
    assert [m.entity for m in found] == ["Nicolaus Copernicus"]


def test_who_full_matches_before_partial(mem):
    found = mem.who([("born_in", "London"), ("occupation", "mathematician")])
    assert found[0].entity == "Ada Lovelace" and found[0].missing == []
    partial = {m.entity: m.missing for m in found[1:]}
    assert partial["Isaac Newton"] == [("born_in", "London")]
    assert all(m.satisfied for m in found)


def test_who_answer_kind(mem):
    assert [m.entity for m in mem.who([("country", "Italy")])] == ["Pisa"]
    assert mem.who([("country", "Italy")], answer_kind="person") == []
    assert [m.entity for m in mem.who([("country", "Italy")],
                                      answer_kind="place")] == ["Pisa"]


def test_who_reasons_through_containment(mem):
    [m] = mem.who([("born_in", "Italy")], answer_kind="person")
    assert m.entity == "Galileo Galilei"
    assert triples(m.satisfied) == [("Galileo Galilei", "born_in", "Pisa"),
                                    ("Pisa", "country", "Italy")]
    assert mem.who([("born_in", "Tuscany")])[0].entity == "Galileo Galilei"


def test_containment_over_several_hops(mem):
    mem.add_many([Fact("Arcetri", "located_in", "Florence"),
                  Fact("Florence", "located_in", "Tuscany"),
                  Fact("Tuscany", "country", "Italy"),
                  Fact("Björk", "born_in", "Reykjavík")])
    [m] = mem.who([("died_in", "Italy")])
    assert m.entity == "Galileo Galilei"
    assert triples(m.satisfied) == [
        ("Galileo Galilei", "died_in", "Arcetri"),
        ("Arcetri", "located_in", "Florence"),
        ("Florence", "located_in", "Tuscany"),
        ("Tuscany", "country", "Italy")]
    ok, facts = mem.check("Galileo", "died_in", "Tuscany")
    assert ok is True and len(facts) == 3
    # a country's capital lies inside it
    [b] = mem.who([("born_in", "Iceland")])
    assert b.entity == "Björk"
    assert ("Iceland", "capital", "Reykjavík") in triples(b.satisfied)
    assert mem.check("Björk", "born_in", "Iceland")[0] is True
    # containment never runs backwards: born in Italy isn't born in Pisa
    mem.add(Fact("Maria Montessori", "born_in", "Italy"))
    assert mem.check("Maria Montessori", "born_in", "Pisa") == (None, [])
    assert "Maria Montessori" not in \
        [m.entity for m in mem.who([("born_in", "Pisa")])]


def test_who_loose_matching_never_overreaches(mem):
    mem.add(Fact("Nikola Tesla", "died_in", "New York City"))
    assert mem.who([("died_in", "York")]) == []
    assert [m.entity for m in mem.who([("died_in", "New York")])] == \
        ["Nikola Tesla"]
    assert mem.who([("born_in", "Atlantis")]) == []
    assert mem.who([]) == []


def test_loose_match_rules():
    assert loose_match("Padua", "the University of Padua")
    assert loose_match("Reykjavik", "Reykjavík")
    assert loose_match("Mexico", "Mexico City")
    assert loose_match("Europe", "Southern Europe")
    assert loose_match("1564", "15 February 1564")
    assert loose_match("February 1564", "15 February 1564")
    assert loose_match("Feb 15th, 1564", "15 February 1564")
    assert loose_match("Nobel Prize", "Nobel Prize in Physics")
    assert loose_match("390,000", "about 390,000")
    assert not loose_match("York", "New York City")
    assert not loose_match("Virginia", "West Virginia")
    assert not loose_match("Mexico City", "Mexico")      # user more specific
    assert not loose_match("University", "University of Padua")
    assert not loose_match("16 February 1564", "15 February 1564")
    assert not loose_match("Cambridge", "Trinity College, Cambridge")
    assert not loose_match("Prize", "Nobel Prize in Physics")


# ─── analogy, chain, check, compare, describe ───────────────────────

def test_analogy_galileo_pisa_newton(mem):
    answers = mem.analogy("Galileo", "Pisa", "Newton")
    assert answers[0].value == "Woolsthorpe"
    assert triples(answers[0].facts) == [
        ("Galileo Galilei", "born_in", "Pisa"),
        ("Isaac Newton", "born_in", "Woolsthorpe")]
    # the link found by unbinding is the one relation that holds
    assert values(answers) == {"woolsthorpe"}


def test_analogy_other_shapes(mem):
    # loose b: Padua -> taught_at (University of Padua)
    [a] = mem.analogy("Galileo", "Padua", "Newton")
    assert a.value == "University of Cambridge"
    # b is the entity: Pisa is to Galileo as Woolsthorpe is to ?
    [r] = mem.analogy("Pisa", "Galileo", "Woolsthorpe")
    assert r.value == "Isaac Newton"
    assert ("Galileo Galilei", "born_in", "Pisa") in triples(r.facts)
    assert mem.analogy("Galileo", "Mars", "Newton") == []
    assert mem.analogy("Nobody", "Pisa", "Newton") == []


def test_chain_follows_each_hop(mem):
    [a] = mem.chain("Galileo", ["born_in", "country"])
    assert a.value == "Italy"
    assert triples(a.facts) == [("Galileo Galilei", "born_in", "Pisa"),
                                ("Pisa", "country", "Italy")]
    [cap] = mem.chain("Pisa", ["country", "capital"])
    assert cap.value == "Rome"
    # Woolsthorpe has no facts: the path stops rather than guessing
    assert mem.chain("Newton", ["born_in", "country"]) == []
    assert mem.chain("Galileo", []) == []


def test_check_yes_no_unknown(mem):
    ok, facts = mem.check("Galileo", "born_in", "Pisa")
    assert ok is True and triples(facts) == [
        ("Galileo Galilei", "born_in", "Pisa")]
    assert mem.check("Galileo", "taught_at", "Padua")[0] is True
    assert mem.check("Galileo", "born_on", "1564")[0] is True
    ok, facts = mem.check("Galileo", "born_in", "Italy")      # containment
    assert ok is True and ("Pisa", "country", "Italy") in triples(facts)
    # no: a single-valued relation verifiably holds something else
    ok, facts = mem.check("Galileo", "born_year", "1565")
    assert ok is False and triples(facts) == [
        ("Galileo Galilei", "born_year", "1564")]
    assert mem.check("Galileo", "born_on", "16 February 1564")[0] is False
    # a name never seen may be another name for Rome: unknown, not "no"
    assert mem.check("Italy", "capital", "Roma") == (None, [])
    assert mem.check("Italy", "capital", "Milan") == (None, [])
    mem.add(Fact("Milan", "instance_of", "city"))
    ok, facts = mem.check("Italy", "capital", "Milan")
    assert ok is False and triples(facts) == [("Italy", "capital", "Rome")]
    mem.add(Fact("Rome", "alias", "Roma"))
    assert mem.check("Italy", "capital", "Roma")[0] is True
    # unknown
    assert mem.check("Galileo", "occupation", "chemist") == (None, [])
    assert mem.check("Galileo", "born_on", "15 February 1564 at noon")[0] \
        is None
    assert mem.check("Leonardo", "born_in", "Vinci") == (None, [])
    assert mem.check("Galileo", "spouse", "Marina Gamba") == (None, [])
    # born in Pisa doesn't rule out Florence until Florence is known to be
    # another city
    assert mem.check("Galileo", "born_in", "Florence") == (None, [])
    mem.add(Fact("Florence", "instance_of", "city"))
    ok, facts = mem.check("Galileo", "born_in", "Florence")
    assert ok is False and triples(facts) == [
        ("Galileo Galilei", "born_in", "Pisa")]


def test_check_dates_and_numbers():
    m = FactMemory()
    m.add(Fact("Iceland", "population", "about 390,000"))
    m.add(Fact("Ada Lovelace", "born_year", "1815"))
    assert m.check("Iceland", "population", "390000")[0] is True
    assert m.check("Iceland", "population", "400,000")[0] is None  # "about"
    assert m.check("Ada", "born_year", "10 December 1815")[0] is True
    assert m.check("Ada", "born_year", "1816")[0] is False


def test_compare(mem):
    shared = mem.compare("Marie Curie", "Albert Einstein")
    assert [(r, v) for r, v, _ in shared] == \
        [("award", "Nobel Prize in Physics")]
    assert {f.subject for f in shared[0][2]} == {"Marie Curie",
                                                 "Albert Einstein"}
    mem.add(Fact("Johannes Kepler", "occupation", "astronomer"))
    assert ("occupation", "astronomer") in \
        [(r, v) for r, v, _ in mem.compare("Galileo", "Kepler")]
    assert mem.compare("Galileo", "Galileo Galilei") == []
    assert mem.compare("Galileo", "Nobody") == []


def test_describe_reproduces_facts_about(mem):
    for s in SUBJECTS:
        assert mem.describe(s) == mem.facts_about(s), s
    assert mem.describe("Nobody") == []
    assert mem.stats()["rejected"] == 0


# ─── forgetting ─────────────────────────────────────────────────────

def test_remove_memory_forgets_one_article(mem):
    gid = 100 + SUBJECTS.index("Galileo Galilei")
    n_galileo = sum(1 for s, _, _ in EXPECTED if s == "Galileo Galilei")
    assert mem.remove_memory(gid) == n_galileo
    assert mem.remove_memory(gid) == 0
    assert mem.resolve("Galileo") is None
    assert mem.lookup("Galileo Galilei", "born_in") == []
    assert mem.who([("born_in", "Pisa")]) == []
    assert mem.analogy("Galileo", "Pisa", "Newton") == []
    assert mem.lookup("Newton", "born_in")[0].value == "Woolsthorpe"
    s = mem.stats()
    assert s["facts"] == len(EXPECTED) - n_galileo
    assert s["entities"] == s["records"] == len(SUBJECTS) - 1


def test_remove_one_sentence_rebuilds_the_record():
    m = FactMemory()
    m.add(Fact("Galileo Galilei", "born_in", "Pisa", memory_id=1))
    m.add(Fact("Galileo Galilei", "died_in", "Arcetri", memory_id=2))
    m.add(Fact("Galileo Galilei", "died_in", "Arcetri", source="user",
               memory_id=3))
    m.add(Fact("Galileo Galilei", "occupation", "astronomer", memory_id=2))
    assert m.remove_memory(2) == 2
    assert values(m.lookup("Galileo", "born_in")) == {"pisa"}
    assert m.lookup("Galileo", "occupation") == []
    [died] = m.lookup("Galileo", "died_in")            # still backed by #3
    assert [f.memory_id for f in died.facts] == [3]
    assert m.describe("Galileo") == m.facts_about("Galileo")
    # claim-level forgetting
    assert m.remove(Fact("Galileo Galilei", "born_in", "Pisa", memory_id=1))
    assert not m.remove(Fact("Galileo Galilei", "born_in", "Pisa"))
    assert m.lookup("Galileo", "born_in") == []
    assert m.remove_memory(1) == 0
    m.clear()
    assert len(m) == 0 and m.lookup("Galileo", "born_in") == []


# ─── sharding ───────────────────────────────────────────────────────

def big_entity(n: int) -> list[Fact]:
    rels = ["wrote", "composed", "painted", "discovered", "invented"]
    return [Fact("Polymath Prime", rels[i % len(rels)], f"Opus {i}",
                 memory_id=i) for i in range(n)]


def test_sixty_fact_entity_is_sharded_with_full_recall():
    m = FactMemory()
    facts = big_entity(60)
    m.add_many(facts)
    m.add_many(corpus_facts())
    assert m.stats()["sharded_entities"] == 1
    assert len(m._entity_rows[norm_key("Polymath Prime")]) == 2
    for rel in {f.relation for f in facts}:
        want = {norm_key(f.obj) for f in facts if f.relation == rel}
        assert values(m.lookup("Polymath Prime", rel, k=100)) == want
    assert m.describe("Polymath Prime") == m.facts_about("Polymath Prime")
    assert m.who([("painted", "Opus 2"), ("wrote", "Opus 55")])[0].entity \
        == "Polymath Prime"


def test_sharding_is_what_keeps_recall():
    """300 facts in one record put each item ~4.6 sigma above chance -
    under the 6 sigma bar; shards of <= 48 keep them ~11.5 sigma above."""
    facts = big_entity(300)

    def recall(m: FactMemory) -> float:
        m.add_many(facts)
        got = sum(len(m.lookup("Polymath Prime", rel, k=1000))
                  for rel in {f.relation for f in facts})
        return got / len(facts)

    unsharded = FactMemory()
    unsharded.shard_size = 10 ** 6
    assert recall(unsharded) < 0.5
    assert recall(FactMemory()) == 1.0


# ─── HDC does the work ──────────────────────────────────────────────

def test_damaged_record_degrades_answers(mem):
    """The index still holds every fact, but answers come from the record:
    light damage is shrugged off, heavy damage silences the entity."""
    flip_bits(mem, "Galileo Galilei", 0.10)
    assert mem.lookup("Galileo", "born_in")[0].value == "Pisa"
    assert mem.describe("Galileo") == mem.facts_about("Galileo")
    flip_bits(mem, "Galileo Galilei", 0.45, seed=1)    # now ~50% flipped
    assert len(mem.facts_about("Galileo")) == 13       # index intact
    assert mem.lookup("Galileo", "born_in") == []
    assert mem.describe("Galileo") == []
    assert mem.check("Galileo", "born_in", "Pisa") == (None, [])
    assert mem.analogy("Galileo", "Pisa", "Newton") == []
    assert "Galileo Galilei" not in \
        [m.entity for m in mem.who([("born_in", "Pisa")])]
    # the record is rebuilt from the facts on the next change
    mem.add(Fact("Galileo Galilei", "known_for", "the telescope"))
    assert mem.lookup("Galileo", "born_in")[0].value == "Pisa"


def test_damage_curve_is_gradual():
    m = FactMemory()
    m.add_many(corpus_facts())
    recovered = []
    for frac in (0.0, 0.2, 0.3, 0.4, 0.5):
        m._dirty.add(norm_key("Galileo Galilei"))      # fresh record
        flip_bits(m, "Galileo Galilei", frac, seed=7)
        recovered.append(len(m.describe("Galileo")))
    assert recovered[0] == recovered[1] == 13
    assert recovered == sorted(recovered, reverse=True)
    assert recovered[-1] == 0


def test_unverifiable_proposals_are_rejected_and_counted(mem):
    """Give Galileo Newton's record: HDC now proposes Newton's values for
    Galileo, and the index refuses every one of them."""
    mem._sync()
    g = mem._entity_rows[norm_key("Galileo Galilei")][0]
    n = mem._entity_rows[norm_key("Isaac Newton")][0]
    mem._records.words[g] = mem._records.words[n]
    before = mem.stats()["rejected"]
    assert mem.lookup("Galileo", "born_in") == []
    assert mem.stats()["rejected"] == before + 1
    assert [m.entity for m in mem.who([("born_in", "Woolsthorpe")])] == \
        ["Isaac Newton"]
    assert mem.describe("Galileo") == []
    assert mem.stats()["rejected"] > before + 1


def test_smaller_dimension_still_works():
    m = FactMemory(dim=4096)
    m.add_many(corpus_facts())
    assert m.stats()["bytes_per_record"] == 512
    assert m.shard_size == 19
    assert m.lookup("Galileo", "born_in")[0].value == "Pisa"
    assert m.describe("Galileo") == m.facts_about("Galileo")
    with pytest.raises(ValueError):
        FactMemory(dim=100)


# ─── scale ──────────────────────────────────────────────────────────

POOLS = {
    "occupation": [f"occupation {i}" for i in range(60)],
    "nationality": [f"Nation{i}ese" for i in range(40)],
    "born_in": [f"Town {i}" for i in range(400)],
    "died_in": [f"Town {i}" for i in range(400)],
    "born_year": [str(1500 + i) for i in range(300)],
    "died_year": [str(1560 + i) for i in range(300)],
    "educated_at": [f"University of Place {i}" for i in range(200)],
    "award": [f"Prize {i}" for i in range(100)],
    "member_of": [f"Society {i}" for i in range(150)],
}


@pytest.fixture(scope="module")
def world():
    """5,000 synthetic entities x 10 facts (nine pooled, one unique)."""
    rng = random.Random(7)
    truth: dict[tuple[str, str], str] = {}
    holders: dict[tuple[str, str], set[str]] = defaultdict(set)
    facts = []
    for e in range(5000):
        name = f"Scholar{e:05d}"
        for rel, pool in POOLS.items():
            truth[(name, rel)] = rng.choice(pool)
        truth[(name, "wrote")] = f"Opus {e}"
        for rel in list(POOLS) + ["wrote"]:
            v = truth[(name, rel)]
            facts.append(Fact(name, rel, v, source="synthetic", memory_id=e))
            holders[(rel, norm_key(v))].add(name)
    m = FactMemory()
    assert m.add_many(facts) == 50_000
    m.stats()                                   # build every record
    return m, truth, holders, rng


def test_scale_lookup_accuracy_and_speed(world):
    m, truth, _, rng = world
    names = sorted({n for n, _ in truth})
    rels = list(POOLS) + ["wrote"]
    queries = [(rng.choice(names), rng.choice(rels)) for _ in range(400)]
    t0 = time.perf_counter()
    results = [m.lookup(n, r) for n, r in queries]
    per_query = (time.perf_counter() - t0) / len(queries)
    correct = sum(len(a) == 1 and a[0].value == truth[q]
                  for q, a in zip(queries, results))
    assert correct / len(queries) >= 0.99
    assert per_query < 0.050


def test_scale_who_top1_accuracy_and_speed(world):
    m, truth, holders, rng = world
    names = sorted({n for n, _ in truth})
    queries = []
    for i in range(300):
        name = rng.choice(names)
        if i % 3 == 0:
            rels = ["wrote"]
        else:
            rels = rng.sample(list(POOLS), 2 + i % 2)
        cons = [(r, truth[(name, r)]) for r in rels]
        answers = set.intersection(*(holders[(r, norm_key(v))]
                                     for r, v in cons))
        queries.append((cons, answers))
    t0 = time.perf_counter()
    results = [m.who(cons) for cons, _ in queries]
    per_query = (time.perf_counter() - t0) / len(queries)
    correct = sum(bool(found) and found[0].entity in answers
                  and not found[0].missing
                  for (_, answers), found in zip(queries, results))
    assert correct / len(queries) >= 0.99
    assert per_query < 0.050
    assert m.stats()["rejected"] == 0
