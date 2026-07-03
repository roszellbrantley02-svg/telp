"""
lattice/corpus_10domain.py - 10 domains, two shared relations, scale test.

Pushes the cross-domain analogy result from 5 -> 10 domains.
Adds a second shared relation (HAS_CATEGORY) to test if multiple
universal directions can coexist in HD space.

If 100% one-shot cross-domain analogy holds at 10 domains, HDC's
algebra genuinely scales for universal relational structure.
"""
from __future__ import annotations

import random


# Each entry: (entity, primary_attribute, category)
COUNTRIES_10 = [
    ("USA",          "Washington",    "NorthAmerica"),
    ("France",       "Paris",         "Europe"),
    ("Japan",        "Tokyo",         "Asia"),
    ("UnitedKingdom","London",        "Europe"),
    ("Italy",        "Rome",          "Europe"),
    ("Spain",        "Madrid",        "Europe"),
    ("China",        "Beijing",       "Asia"),
    ("Brazil",       "Brasilia",      "SouthAmerica"),
    ("Egypt",        "Cairo",         "Africa"),
    ("Australia",    "Canberra",      "Oceania"),
    ("Germany",      "Berlin",        "Europe"),    # heldout
]

ANIMALS_10 = [
    ("Lion",         "Savannah",      "Mammal"),
    ("Tiger",        "Jungle",        "Mammal"),
    ("PolarBear",    "Arctic",        "Mammal"),
    ("Camel",        "Desert",        "Mammal"),
    ("Whale",        "Ocean",         "Mammal"),
    ("Eagle",        "Mountains",     "Bird"),
    ("Penguin",      "Antarctica",    "Bird"),
    ("Owl",          "Forest",        "Bird"),
    ("Shark",        "Ocean",         "Fish"),
    ("Snake",        "Grasslands",    "Reptile"),
    ("Wolf",         "Forest",        "Mammal"),    # heldout
]

FOODS_10 = [
    ("Pizza",        "Italy",         "Savory"),
    ("Sushi",        "Japan",         "Savory"),
    ("Tacos",        "Mexico",        "Savory"),
    ("Croissant",    "France",        "Pastry"),
    ("Curry",        "India",         "Savory"),
    ("Paella",       "Spain",         "Savory"),
    ("Borscht",      "Russia",        "Savory"),
    ("Pho",          "Vietnam",       "Savory"),
    ("Kimchi",       "Korea",         "Savory"),
    ("Hamburger",    "USA",           "Savory"),
    ("Pretzel",      "Germany",       "Snack"),     # heldout
]

AUTHORS_10 = [
    ("Shakespeare",  "Hamlet",        "Drama"),
    ("Tolstoy",      "WarAndPeace",   "Novel"),
    ("Dickens",      "OliverTwist",   "Novel"),
    ("Hemingway",    "OldManAndSea",  "Novel"),
    ("Orwell",       "Nineteen84",    "Dystopia"),
    ("Homer",        "Iliad",         "Epic"),
    ("Dante",        "DivineComedy",  "Epic"),
    ("Wilde",        "DorianGray",    "Novel"),
    ("Joyce",        "Ulysses",       "Novel"),
    ("Twain",        "TomSawyer",     "Novel"),
    ("Kafka",        "Metamorphosis", "Novella"),   # heldout
]

SPORTS_10 = [
    ("Tennis",       "Racket",        "Racquet"),
    ("Soccer",       "Ball",          "TeamField"),
    ("Hockey",       "Stick",         "TeamIce"),
    ("Golf",         "Club",          "Solo"),
    ("Baseball",     "Bat",           "TeamField"),
    ("Bowling",      "Pins",          "Solo"),
    ("Archery",      "Bow",           "Solo"),
    ("Boxing",       "Gloves",        "Combat"),
    ("Skiing",       "Skis",          "Solo"),
    ("Cycling",      "Bike",          "Solo"),
    ("Basketball",   "Ball",          "TeamCourt"), # heldout
]

# NEW domains beyond v2
MUSICIANS_10 = [
    ("Mozart",       "Requiem",       "Classical"),
    ("Beethoven",    "FifthSymphony", "Classical"),
    ("Bach",         "Goldberg",      "Classical"),
    ("Beatles",      "AbbeyRoad",     "Rock"),
    ("Hendrix",      "PurpleHaze",    "Rock"),
    ("Madonna",      "LikeAVirgin",   "Pop"),
    ("Coltrane",     "GiantSteps",    "Jazz"),
    ("Davis",        "KindOfBlue",    "Jazz"),
    ("Marley",       "OneLove",       "Reggae"),
    ("Eminem",       "LoseYourself",  "Rap"),
    ("Dylan",        "Blowin",        "Folk"),      # heldout
]

MOVIES_10 = [
    ("Inception",    "Nolan",         "SciFi"),
    ("Casablanca",   "Curtiz",        "Romance"),
    ("Psycho",       "Hitchcock",     "Thriller"),
    ("Godfather",    "Coppola",       "Crime"),
    ("Schindler",    "Spielberg",     "Historical"),
    ("Vertigo",      "Hitchcock",     "Thriller"),
    ("Goodfellas",   "Scorsese",      "Crime"),
    ("Tarkovsky",    "Solaris",       "SciFi"),    # author/work swapped intentionally? Let me fix
    ("PulpFiction",  "Tarantino",     "Crime"),
    ("Citizen",      "Welles",        "Drama"),
    ("Parasite",     "BongJoonHo",    "Thriller"),  # heldout
]
# Fix the Tarkovsky entry — should be (Solaris, Tarkovsky, SciFi)
MOVIES_10[7] = ("Solaris", "Tarkovsky", "SciFi")

TOOLS_10 = [
    ("Hammer",       "Nail",          "Striking"),
    ("Screwdriver",  "Screw",         "Rotating"),
    ("Saw",          "Wood",          "Cutting"),
    ("Drill",        "Hole",          "Rotating"),
    ("Wrench",       "Bolt",          "Rotating"),
    ("Pliers",       "Wire",          "Gripping"),
    ("Chisel",       "Stone",         "Striking"),
    ("Plane",        "Wood",          "Cutting"),
    ("File",         "Metal",         "Smoothing"),
    ("Anvil",        "Iron",          "Striking"),
    ("Trowel",       "Mortar",        "Spreading"), # heldout
]

DRINKS_10 = [
    ("Espresso",     "Italy",         "Hot"),
    ("Sake",         "Japan",         "Alcohol"),
    ("Tequila",      "Mexico",        "Alcohol"),
    ("Champagne",    "France",        "Alcohol"),
    ("Chai",         "India",         "Hot"),
    ("Sangria",      "Spain",         "Alcohol"),
    ("Vodka",        "Russia",        "Alcohol"),
    ("Soju",         "Korea",         "Alcohol"),
    ("Mate",         "Argentina",     "Hot"),
    ("Cola",         "USA",           "Cold"),
    ("Bier",         "Germany",       "Alcohol"),   # heldout
]

COMPUTERS_10 = [
    ("MacBook",      "M3",            "Laptop"),
    ("ThinkPad",     "Intel",         "Laptop"),
    ("Mac",          "Apple",         "Desktop"),
    ("Surface",      "Intel",         "Tablet"),
    ("iPhone",       "Apple",         "Phone"),
    ("Pixel",        "Tensor",        "Phone"),
    ("Galaxy",       "Snapdragon",    "Phone"),
    ("PS5",          "AMD",           "Console"),
    ("Switch",       "Tegra",         "Console"),
    ("RaspberryPi",  "ARM",           "SBC"),
    ("Dell",         "Intel",         "Desktop"),   # heldout
]


DOMAINS = {
    "country":   (COUNTRIES_10,  "country",   "capital",   "continent"),
    "animal":    (ANIMALS_10,    "animal",    "habitat",   "class"),
    "food":      (FOODS_10,      "food",      "origin",    "type"),
    "author":    (AUTHORS_10,    "author",    "work",      "genre"),
    "sport":     (SPORTS_10,     "sport",     "equipment", "venue"),
    "musician":  (MUSICIANS_10,  "musician",  "album",     "genre"),
    "movie":     (MOVIES_10,     "movie",     "director",  "genre"),
    "tool":      (TOOLS_10,      "tool",      "target",    "action"),
    "drink":     (DRINKS_10,     "drink",     "origin",    "type"),
    "computer":  (COMPUTERS_10,  "computer",  "chip",      "form"),
}


# Templates using UNIFIED relation names: HAS_PRIMARY and HAS_CATEGORY
TEMPLATES_PER_DOMAIN = {
    "country":  ["{capital} is the capital of {country}.",
                  "{country}'s capital is {capital}.",
                  "{country} is in {continent}."],
    "animal":   ["The {animal} lives in {habitat}.",
                  "{animal}'s habitat is {habitat}.",
                  "The {animal} is a {class_}."],
    "food":     ["{food} comes from {origin}.",
                  "The origin of {food} is {origin}.",
                  "{food} is {type_}."],
    "author":   ["{author} wrote {work}.",
                  "{work} was written by {author}.",
                  "{author} is a {genre} writer."],
    "sport":    ["{sport} uses a {equipment}.",
                  "The equipment for {sport} is a {equipment}.",
                  "{sport} is played at {venue}."],
    "musician": ["{musician} made {album}.",
                  "{album} was made by {musician}.",
                  "{musician} performs {genre}."],
    "movie":    ["{movie} was directed by {director}.",
                  "{director} directed {movie}.",
                  "{movie} is a {genre} film."],
    "tool":     ["A {tool} is used on {target}.",
                  "{tool} acts on {target}.",
                  "A {tool} is a {action} tool."],
    "drink":    ["{drink} comes from {origin}.",
                  "The origin of {drink} is {origin}.",
                  "{drink} is a {type_} drink."],
    "computer": ["{computer} uses {chip}.",
                  "{computer} runs on {chip}.",
                  "{computer} is a {form}."],
}


def build_10domain_corpus(seed: int = 0):
    """Each domain produces sentences where the primary feature uses
    HAS_PRIMARY and the category uses HAS_CATEGORY (universal relations)."""
    rng = random.Random(seed)
    pairs = []
    for dname, (data, ent_key, prim_key, cat_key) in DOMAINS.items():
        templates = TEMPLATES_PER_DOMAIN[dname]
        for entity, primary, category in data:
            for tpl in templates:
                # Decide if this template is a primary or category sentence
                # by checking which keys it uses
                ctx = {
                    ent_key: entity,
                    prim_key: primary,
                    cat_key: category,
                }
                # Also alias 'class_' and 'type_' to standard keys for templates
                ctx["class_"] = category
                ctx["type_"] = category
                sentence = tpl.format(**ctx)
                # Build triple based on whether template references primary or category
                if "{" + prim_key + "}" in tpl or prim_key in tpl:
                    # check if primary or category by template content
                    if any(prim_marker in tpl for prim_marker in
                            ["{capital}", "{habitat}", "{origin}", "{work}",
                             "{equipment}", "{album}", "{director}", "{target}",
                             "{chip}"]):
                        triple = (entity, "HAS_PRIMARY", primary)
                    else:
                        triple = (entity, "HAS_CATEGORY", category)
                else:
                    triple = (entity, "HAS_CATEGORY", category)
                pairs.append((sentence, [triple]))
    rng.shuffle(pairs)
    return pairs


# Helper data
PRIMARY_PAIRS_BY_DOMAIN = {
    d: [(e[0], e[1]) for e in data[:-1]]   # exclude last (heldout)
    for d, (data, _, _, _) in DOMAINS.items()
}
HELDOUT = {d: (data[-1][0], data[-1][1])
             for d, (data, _, _, _) in DOMAINS.items()}
PRIMARIES_BY_DOMAIN = {
    d: list({e[1] for e in data})
    for d, (data, _, _, _) in DOMAINS.items()
}


if __name__ == "__main__":
    pairs = build_10domain_corpus()
    print(f"10-domain corpus: {len(pairs)} sentences across {len(DOMAINS)} domains")
    print(f"Domains: {list(DOMAINS.keys())}")
    print(f"\nSample sentences:")
    for s, ts in pairs[:10]:
        print(f"  {s}  --  {ts[0]}")
