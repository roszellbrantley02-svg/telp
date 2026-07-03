"""
lattice/corpus_multidomain.py - multi-domain corpus with PARALLEL relational structure.

Five domains, each with the SAME shape of relation:
  ENTITY --HAS_PRIMARY_FEATURE--> ATTRIBUTE
  ENTITY --IN_BROADER_CATEGORY--> CATEGORY

Domains:
  1. Country     → HAS_CAPITAL     → Capital City     | IN_CONTINENT → Continent
  2. Animal      → HAS_HABITAT     → Habitat           | IN_CLASS     → Class
  3. Food        → HAS_ORIGIN      → Country of Origin | IN_TYPE      → Food Type
  4. Author      → HAS_FAMOUS_WORK → Famous Work       | IN_GENRE     → Genre
  5. Sport       → HAS_EQUIPMENT   → Primary Equipment | IN_VENUE     → Venue Type

The parallel structure means: if HDC space has emergent meta-structure,
the "HAS_PRIMARY_FEATURE" directions in each domain should be similar
in HD space — even though the specific relation roles differ.

Note: We deliberately use DIFFERENT relation names per domain. The
question is whether the underlying geometric structure of
"entity → defining attribute" is shared across them despite the
different role vectors.
"""
from __future__ import annotations

import random


# ─── COUNTRIES ────────────────────────────────────────────────────


COUNTRIES_MD = [
    # (country, capital, continent)
    ("USA",            "Washington",   "North America"),
    ("France",         "Paris",        "Europe"),
    ("Japan",          "Tokyo",        "Asia"),
    ("UnitedKingdom",  "London",       "Europe"),
    ("Italy",          "Rome",         "Europe"),
    ("Spain",          "Madrid",       "Europe"),
    ("China",          "Beijing",      "Asia"),
    ("Russia",         "Moscow",       "Europe"),
    ("Brazil",         "Brasilia",     "SouthAmerica"),
    ("Canada",         "Ottawa",       "North America"),
    ("Australia",      "Canberra",     "Oceania"),
    ("Argentina",      "BuenosAires",  "SouthAmerica"),
    ("Egypt",          "Cairo",        "Africa"),
    ("Greece",         "Athens",       "Europe"),
    ("Germany",        "Berlin",       "Europe"),    # heldout
]


# ─── ANIMALS ──────────────────────────────────────────────────────


ANIMALS = [
    # (animal, habitat, class)
    ("Lion",      "Savannah",  "Mammal"),
    ("Tiger",     "Jungle",    "Mammal"),
    ("Polar Bear","Arctic",    "Mammal"),
    ("Camel",     "Desert",    "Mammal"),
    ("Whale",     "Ocean",     "Mammal"),
    ("Eagle",     "Mountains", "Bird"),
    ("Penguin",   "Antarctica","Bird"),
    ("Parrot",    "Rainforest","Bird"),
    ("Owl",       "Forest",    "Bird"),
    ("Shark",     "Ocean",     "Fish"),
    ("Trout",     "Rivers",    "Fish"),
    ("Salmon",    "Rivers",    "Fish"),
    ("Frog",      "Ponds",     "Amphibian"),
    ("Snake",     "Grasslands","Reptile"),
    ("Wolf",      "Forest",    "Mammal"),    # heldout
]


# ─── FOODS ────────────────────────────────────────────────────────


FOODS = [
    # (food, origin_country, type)
    ("Pizza",        "Italy",        "Savory"),
    ("Sushi",        "Japan",        "Savory"),
    ("Tacos",        "Mexico",       "Savory"),
    ("Croissant",    "France",       "Pastry"),
    ("Curry",        "India",        "Savory"),
    ("Paella",       "Spain",        "Savory"),
    ("Fish Chips",   "UnitedKingdom","Savory"),
    ("Borscht",      "Russia",       "Savory"),
    ("Schnitzel",    "Austria",      "Savory"),
    ("Goulash",      "Hungary",      "Savory"),
    ("Hamburger",    "USA",          "Savory"),
    ("Couscous",     "Morocco",      "Savory"),
    ("Pho",          "Vietnam",      "Savory"),
    ("Kimchi",       "Korea",        "Savory"),
    ("Pretzel",      "Germany",      "Snack"),     # heldout
]


# ─── AUTHORS ──────────────────────────────────────────────────────


AUTHORS = [
    # (author, famous_work, genre)
    ("Shakespeare",  "Hamlet",       "Drama"),
    ("Tolstoy",      "WarAndPeace",  "Novel"),
    ("Dickens",      "OliverTwist",  "Novel"),
    ("Hemingway",    "OldManAndSea", "Novel"),
    ("Austen",       "PrideAndPrejudice","Novel"),
    ("Orwell",       "Nineteen84",   "Dystopia"),
    ("Twain",        "TomSawyer",    "Novel"),
    ("Dostoevsky",   "CrimeAndPunishment","Novel"),
    ("Homer",        "Iliad",        "Epic"),
    ("Dante",        "DivineComedy", "Epic"),
    ("Cervantes",    "DonQuixote",   "Novel"),
    ("Wilde",        "DorianGray",   "Novel"),
    ("Poe",          "TheRaven",     "Poem"),
    ("Joyce",        "Ulysses",      "Novel"),
    ("Kafka",        "Metamorphosis","Novel"),    # heldout
]


# ─── SPORTS ───────────────────────────────────────────────────────


SPORTS = [
    # (sport, equipment, venue)
    ("Tennis",      "Racket",       "Court"),
    ("Soccer",      "Ball",         "Field"),
    ("Hockey",      "Stick",        "Rink"),
    ("Golf",        "Club",         "Course"),
    ("Baseball",    "Bat",          "Stadium"),
    ("Bowling",     "Pins",         "Alley"),
    ("Archery",     "Bow",          "Range"),
    ("Boxing",      "Gloves",       "Ring"),
    ("Cricket",     "Bat",          "Pitch"),
    ("Fencing",     "Sword",        "Strip"),
    ("Skiing",      "Skis",         "Slope"),
    ("Surfing",     "Board",        "Beach"),
    ("Cycling",     "Bike",         "Track"),
    ("Rowing",      "Oars",         "Lake"),
    ("Basketball",  "Ball",         "Court"),    # heldout
]


# Templates per domain — note: relation names DIFFER per domain
COUNTRY_TEMPLATES = [
    ("{capital} is the capital of {country}.",
        [("country", "HAS_CAPITAL", "capital")]),
    ("The capital of {country} is {capital}.",
        [("country", "HAS_CAPITAL", "capital")]),
    ("{country}'s capital is {capital}.",
        [("country", "HAS_CAPITAL", "capital")]),
    ("{country} is located in {continent}.",
        [("country", "IN_CONTINENT", "continent")]),
    ("{country} is a country in {continent}.",
        [("country", "IN_CONTINENT", "continent")]),
]

ANIMAL_TEMPLATES = [
    ("The {animal} lives in the {habitat}.",
        [("animal", "HAS_HABITAT", "habitat")]),
    ("A {animal} is found in {habitat}.",
        [("animal", "HAS_HABITAT", "habitat")]),
    ("{animal}'s habitat is {habitat}.",
        [("animal", "HAS_HABITAT", "habitat")]),
    ("The {animal} is a {class_}.",
        [("animal", "IN_CLASS", "class_")]),
    ("A {animal} belongs to the {class_} class.",
        [("animal", "IN_CLASS", "class_")]),
]

FOOD_TEMPLATES = [
    ("{food} comes from {origin}.",
        [("food", "HAS_ORIGIN", "origin")]),
    ("{food} originated in {origin}.",
        [("food", "HAS_ORIGIN", "origin")]),
    ("The origin of {food} is {origin}.",
        [("food", "HAS_ORIGIN", "origin")]),
    ("{food} is a {type_} food.",
        [("food", "IN_TYPE", "type_")]),
    ("{food} is classified as {type_}.",
        [("food", "IN_TYPE", "type_")]),
]

AUTHOR_TEMPLATES = [
    ("{author} wrote {work}.",
        [("author", "HAS_FAMOUS_WORK", "work")]),
    ("{work} was written by {author}.",
        [("author", "HAS_FAMOUS_WORK", "work")]),
    ("{author} is known for {work}.",
        [("author", "HAS_FAMOUS_WORK", "work")]),
    ("{author} works in the {genre} genre.",
        [("author", "IN_GENRE", "genre")]),
    ("{author} is a {genre} writer.",
        [("author", "IN_GENRE", "genre")]),
]

SPORT_TEMPLATES = [
    ("{sport} uses a {equipment}.",
        [("sport", "HAS_EQUIPMENT", "equipment")]),
    ("The main equipment for {sport} is a {equipment}.",
        [("sport", "HAS_EQUIPMENT", "equipment")]),
    ("{sport} requires a {equipment}.",
        [("sport", "HAS_EQUIPMENT", "equipment")]),
    ("{sport} is played on a {venue}.",
        [("sport", "IN_VENUE", "venue")]),
    ("The venue for {sport} is a {venue}.",
        [("sport", "IN_VENUE", "venue")]),
]


def build_multidomain_corpus(seed: int = 0):
    """Generate the combined multi-domain tagged corpus."""
    rng = random.Random(seed)
    pairs = []

    for c, cap, cont in COUNTRIES_MD:
        ctx = {"country": c, "capital": cap, "continent": cont}
        tpls = list(COUNTRY_TEMPLATES); rng.shuffle(tpls)
        for tpl_str, triple_specs in tpls:
            sent = tpl_str.format(**ctx)
            triples = [(ctx[s], r, ctx[o]) for s, r, o in triple_specs]
            pairs.append((sent, triples))

    for a, hab, cls in ANIMALS:
        ctx = {"animal": a, "habitat": hab, "class_": cls}
        tpls = list(ANIMAL_TEMPLATES); rng.shuffle(tpls)
        for tpl_str, triple_specs in tpls:
            sent = tpl_str.format(**ctx)
            triples = [(ctx[s], r, ctx[o]) for s, r, o in triple_specs]
            pairs.append((sent, triples))

    for f, origin, typ in FOODS:
        ctx = {"food": f, "origin": origin, "type_": typ}
        tpls = list(FOOD_TEMPLATES); rng.shuffle(tpls)
        for tpl_str, triple_specs in tpls:
            sent = tpl_str.format(**ctx)
            triples = [(ctx[s], r, ctx[o]) for s, r, o in triple_specs]
            pairs.append((sent, triples))

    for au, work, genre in AUTHORS:
        ctx = {"author": au, "work": work, "genre": genre}
        tpls = list(AUTHOR_TEMPLATES); rng.shuffle(tpls)
        for tpl_str, triple_specs in tpls:
            sent = tpl_str.format(**ctx)
            triples = [(ctx[s], r, ctx[o]) for s, r, o in triple_specs]
            pairs.append((sent, triples))

    for sp, eq, ven in SPORTS:
        ctx = {"sport": sp, "equipment": eq, "venue": ven}
        tpls = list(SPORT_TEMPLATES); rng.shuffle(tpls)
        for tpl_str, triple_specs in tpls:
            sent = tpl_str.format(**ctx)
            triples = [(ctx[s], r, ctx[o]) for s, r, o in triple_specs]
            pairs.append((sent, triples))

    rng.shuffle(pairs)
    return pairs


# Per-domain "primary feature" relations (the parallel ones we want to test)
PRIMARY_RELATIONS = {
    "country": ("HAS_CAPITAL",      [(c[0], c[1]) for c in COUNTRIES_MD[:-1]]),
    "animal":  ("HAS_HABITAT",      [(a[0], a[1]) for a in ANIMALS[:-1]]),
    "food":    ("HAS_ORIGIN",       [(f[0], f[1]) for f in FOODS[:-1]]),
    "author":  ("HAS_FAMOUS_WORK",  [(a[0], a[1]) for a in AUTHORS[:-1]]),
    "sport":   ("HAS_EQUIPMENT",    [(s[0], s[1]) for s in SPORTS[:-1]]),
}

HELDOUT = {
    "country": (COUNTRIES_MD[-1][0], COUNTRIES_MD[-1][1]),   # Germany -> Berlin
    "animal":  (ANIMALS[-1][0], ANIMALS[-1][1]),             # Wolf -> Forest
    "food":    (FOODS[-1][0], FOODS[-1][1]),                 # Pretzel -> Germany
    "author":  (AUTHORS[-1][0], AUTHORS[-1][1]),             # Kafka -> Metamorphosis
    "sport":   (SPORTS[-1][0], SPORTS[-1][1]),               # Basketball -> Ball
}


if __name__ == "__main__":
    pairs = build_multidomain_corpus()
    print(f"Multi-domain corpus: {len(pairs)} sentences")
    by_domain = {"country": 0, "animal": 0, "food": 0, "author": 0, "sport": 0}
    for s, ts in pairs:
        for _s, r, _o in ts:
            if r == "HAS_CAPITAL" or r == "IN_CONTINENT": by_domain["country"] += 1; break
            if r == "HAS_HABITAT"  or r == "IN_CLASS":    by_domain["animal"]  += 1; break
            if r == "HAS_ORIGIN"   or r == "IN_TYPE":     by_domain["food"]    += 1; break
            if r == "HAS_FAMOUS_WORK" or r == "IN_GENRE": by_domain["author"]  += 1; break
            if r == "HAS_EQUIPMENT" or r == "IN_VENUE":   by_domain["sport"]   += 1; break
    print("Triples by domain:", by_domain)
    print("\nSample sentences:")
    for s, ts in pairs[:8]:
        print(f"  {s}  --  {ts[0]}")
