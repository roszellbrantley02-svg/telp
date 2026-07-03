"""
lattice/corpus_enriched.py - corpus where capitals also have their own facts.

In corpus_tagged.py, capitals only appear as objects of HAS_CAPITAL.
Hold out HAS_CAPITAL for a country and its capital becomes isolated
in the knowledge graph — no information can flow to/from it.

This version adds facts ABOUT capitals: their continent, what they're
famous for, the language spoken there, etc. Now capitals are real
nodes in the graph with their own neighborhoods. Message passing can
exploit shared structure (e.g., Berlin and Germany both connect to
Europe via IN_CONTINENT).

Templates and country facts are augmented with capital-specific
template sentences.
"""
from __future__ import annotations

import random
from lattice.corpus import COUNTRIES


# Each capital gets a few independent facts.
# Hand-curated minimal facts — only what's true and useful as graph signal.
CAPITAL_FACTS = {
    "Washington":     ("North America", "English",    "politics"),
    "Paris":          ("Europe",        "French",     "fashion"),
    "Tokyo":          ("Asia",          "Japanese",   "technology"),
    "London":         ("Europe",        "English",    "history"),
    "Rome":           ("Europe",        "Italian",    "history"),
    "Madrid":         ("Europe",        "Spanish",    "art"),
    "Beijing":        ("Asia",          "Mandarin",   "history"),
    "Moscow":         ("Europe",        "Russian",    "architecture"),
    "Brasilia":       ("South America", "Portuguese", "architecture"),
    "Ottawa":         ("North America", "English",    "politics"),
    "Canberra":       ("Oceania",       "English",    "politics"),
    "Mexico City":    ("North America", "Spanish",    "culture"),
    "Buenos Aires":   ("South America", "Spanish",    "tango"),
    "Seoul":          ("Asia",          "Korean",     "technology"),
    "Cairo":          ("Africa",        "Arabic",     "history"),
    "Athens":         ("Europe",        "Greek",      "history"),
    "Stockholm":      ("Europe",        "Swedish",    "design"),
    "Warsaw":         ("Europe",        "Polish",     "history"),
    "Ankara":         ("Asia",          "Turkish",    "history"),
    "Amsterdam":      ("Europe",        "Dutch",      "canals"),
    "Cape Town":      ("Africa",        "English",    "tourism"),
    "Hanoi":          ("Asia",          "Vietnamese", "culture"),
    "Santiago":       ("South America", "Spanish",    "mountains"),
    "Nairobi":        ("Africa",        "Swahili",    "wildlife"),
    "Wellington":     ("Oceania",       "English",    "landscapes"),
    # Heldout
    "Berlin":         ("Europe",        "German",     "history"),
    "Delhi":          ("Asia",          "Hindi",      "history"),
    "Bangkok":        ("Asia",          "Thai",       "temples"),
    "Lisbon":         ("Europe",        "Portuguese", "history"),
    "Oslo":           ("Europe",        "Norwegian",  "landscapes"),
    "Lima":           ("South America", "Spanish",    "history"),
    "Abuja":          ("Africa",        "English",    "politics"),
}


CAPITAL_TEMPLATES = [
    ("{capital} is located in {continent}.",
        [("capital", "IN_CONTINENT", "continent")]),
    ("People in {capital} primarily speak {language}.",
        [("capital", "SPEAKS", "language")]),
    ("{capital} is famous for its {feature}.",
        [("capital", "FAMOUS_FOR", "feature")]),
    ("Visitors to {capital} enjoy its {feature}.",
        [("capital", "FAMOUS_FOR", "feature")]),
    ("{capital} is a major city in {continent}.",
        [("capital", "IN_CONTINENT", "continent")]),
]


def build_enriched_corpus(seed: int = 0):
    """Return list of (sentence, list_of_triples)."""
    from lattice.corpus_tagged import build_tagged_corpus
    pairs = build_tagged_corpus(seed=seed)

    rng = random.Random(seed)
    for capital, (continent, language, feature) in CAPITAL_FACTS.items():
        ctx = {
            "capital":   capital,
            "continent": continent,
            "language":  language,
            "feature":   feature,
        }
        tpls = list(CAPITAL_TEMPLATES)
        rng.shuffle(tpls)
        for tpl_str, triple_specs in tpls:
            sentence = tpl_str.format(**ctx)
            triples = []
            for s_field, rel, o_field in triple_specs:
                triples.append((ctx[s_field], rel, ctx[o_field]))
            pairs.append((sentence, triples))

    rng.shuffle(pairs)
    return pairs


def main():
    pairs = build_enriched_corpus()
    print(f"Enriched corpus: {len(pairs)} sentences")
    # Count triples
    total_triples = sum(len(t) for _, t in pairs)
    print(f"Total triples: {total_triples}")
    # Sample
    print("\nSample capital-specific sentences:")
    cap_examples = [p for p in pairs if any(
        t[0] in CAPITAL_FACTS and t[1] != "HAS_CAPITAL_INV" for t in p[1]
    )][:5]
    for s, ts in cap_examples:
        print(f"  S: {s}")
        for t in ts:
            print(f"     {t}")


if __name__ == "__main__":
    main()
