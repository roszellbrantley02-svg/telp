"""
lattice/corpus_unified.py - multi-domain corpus with UNIFIED primary relation.

Same 5 domains as corpus_multidomain.py — countries, animals, foods,
authors, sports — but every domain's "primary feature" relation gets
the SAME name: HAS_PRIMARY.

The hypothesis: if we use one shared role vector for the primary
relation across all domains, the "direction" we derive from any pair
should generalize to ANY other domain. Germany:Berlin :: Lion:Savannah
should work via pure algebra.

Secondary relations stay type-specific (IS_TYPE_X) so we can still
tell domains apart for cleanup memory selection.
"""
from __future__ import annotations

import random
from lattice.corpus_multidomain import (
    COUNTRIES_MD, ANIMALS, FOODS, AUTHORS, SPORTS, HELDOUT,
)


COUNTRY_TEMPLATES = [
    ("{capital} is the capital of {country}.",
        [("country", "HAS_PRIMARY", "capital")]),
    ("The capital of {country} is {capital}.",
        [("country", "HAS_PRIMARY", "capital")]),
    ("{country}'s capital is {capital}.",
        [("country", "HAS_PRIMARY", "capital")]),
    ("{country} is located in {continent}.",
        [("country", "IS_LOCATION_TYPE", "continent")]),
    ("{country} is a country in {continent}.",
        [("country", "IS_LOCATION_TYPE", "continent")]),
]

ANIMAL_TEMPLATES = [
    ("The {animal} lives in the {habitat}.",
        [("animal", "HAS_PRIMARY", "habitat")]),
    ("A {animal} is found in {habitat}.",
        [("animal", "HAS_PRIMARY", "habitat")]),
    ("{animal}'s habitat is {habitat}.",
        [("animal", "HAS_PRIMARY", "habitat")]),
    ("The {animal} is a {class_}.",
        [("animal", "IS_TAXON_TYPE", "class_")]),
    ("A {animal} belongs to the {class_} class.",
        [("animal", "IS_TAXON_TYPE", "class_")]),
]

FOOD_TEMPLATES = [
    ("{food} comes from {origin}.",
        [("food", "HAS_PRIMARY", "origin")]),
    ("{food} originated in {origin}.",
        [("food", "HAS_PRIMARY", "origin")]),
    ("The origin of {food} is {origin}.",
        [("food", "HAS_PRIMARY", "origin")]),
    ("{food} is a {type_} food.",
        [("food", "IS_FLAVOR_TYPE", "type_")]),
    ("{food} is classified as {type_}.",
        [("food", "IS_FLAVOR_TYPE", "type_")]),
]

AUTHOR_TEMPLATES = [
    ("{author} wrote {work}.",
        [("author", "HAS_PRIMARY", "work")]),
    ("{work} was written by {author}.",
        [("author", "HAS_PRIMARY", "work")]),
    ("{author} is known for {work}.",
        [("author", "HAS_PRIMARY", "work")]),
    ("{author} works in the {genre} genre.",
        [("author", "IS_LITERATURE_TYPE", "genre")]),
    ("{author} is a {genre} writer.",
        [("author", "IS_LITERATURE_TYPE", "genre")]),
]

SPORT_TEMPLATES = [
    ("{sport} uses a {equipment}.",
        [("sport", "HAS_PRIMARY", "equipment")]),
    ("The main equipment for {sport} is a {equipment}.",
        [("sport", "HAS_PRIMARY", "equipment")]),
    ("{sport} requires a {equipment}.",
        [("sport", "HAS_PRIMARY", "equipment")]),
    ("{sport} is played on a {venue}.",
        [("sport", "IS_VENUE_TYPE", "venue")]),
    ("The venue for {sport} is a {venue}.",
        [("sport", "IS_VENUE_TYPE", "venue")]),
]


def build_unified_corpus(seed: int = 0):
    rng = random.Random(seed)
    pairs = []
    for c, cap, cont in COUNTRIES_MD:
        ctx = {"country": c, "capital": cap, "continent": cont}
        tpls = list(COUNTRY_TEMPLATES); rng.shuffle(tpls)
        for t, specs in tpls:
            pairs.append((t.format(**ctx), [(ctx[s], r, ctx[o]) for s,r,o in specs]))
    for a, h, cls in ANIMALS:
        ctx = {"animal": a, "habitat": h, "class_": cls}
        tpls = list(ANIMAL_TEMPLATES); rng.shuffle(tpls)
        for t, specs in tpls:
            pairs.append((t.format(**ctx), [(ctx[s], r, ctx[o]) for s,r,o in specs]))
    for f, o, t_ in FOODS:
        ctx = {"food": f, "origin": o, "type_": t_}
        tpls = list(FOOD_TEMPLATES); rng.shuffle(tpls)
        for t, specs in tpls:
            pairs.append((t.format(**ctx), [(ctx[s], r, ctx[o]) for s,r,o in specs]))
    for au, w, g in AUTHORS:
        ctx = {"author": au, "work": w, "genre": g}
        tpls = list(AUTHOR_TEMPLATES); rng.shuffle(tpls)
        for t, specs in tpls:
            pairs.append((t.format(**ctx), [(ctx[s], r, ctx[o]) for s,r,o in specs]))
    for sp, eq, ven in SPORTS:
        ctx = {"sport": sp, "equipment": eq, "venue": ven}
        tpls = list(SPORT_TEMPLATES); rng.shuffle(tpls)
        for t, specs in tpls:
            pairs.append((t.format(**ctx), [(ctx[s], r, ctx[o]) for s,r,o in specs]))
    rng.shuffle(pairs)
    return pairs


# Domain-specific candidate sets for restricted query (sanity tests)
PRIMARY_BY_DOMAIN = {
    "country": [(c[0], c[1]) for c in COUNTRIES_MD[:-1]],
    "animal":  [(a[0], a[1]) for a in ANIMALS[:-1]],
    "food":    [(f[0], f[1]) for f in FOODS[:-1]],
    "author":  [(a[0], a[1]) for a in AUTHORS[:-1]],
    "sport":   [(s[0], s[1]) for s in SPORTS[:-1]],
}

ATTRIBS_BY_DOMAIN = {
    "country": list({c[1] for c in COUNTRIES_MD}),
    "animal":  list({a[1] for a in ANIMALS}),
    "food":    list({f[1] for f in FOODS}),
    "author":  list({a[1] for a in AUTHORS}),
    "sport":   list({s[1] for s in SPORTS}),
}
