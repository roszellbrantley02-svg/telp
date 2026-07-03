"""
lattice/corpus.py - synthetic country fact corpus for Random Indexing.

Random Indexing learns semantic geometry from co-occurrence. For words
to develop meaningful similarity, they need to appear in many varied
contexts. A flat statement like "Germany's capital is Berlin" gives one
weak signal; we need that pattern showing up across many sentence
templates with different surrounding vocabulary.

This module generates a country-fact corpus by combining:
  - Country facts: continent, capital, language, currency, neighbors
  - ~12 sentence templates per country
  - Templates use varied vocabulary so the contexts aren't repetitive

Total output: ~25 countries × ~12 templates = ~300 sentences. Small
compared to real LLM training corpora, but enough to test whether the
PRINCIPLE of co-occurrence-based semantic learning works in HDC.

For honesty: the corpus DOES include sentences like "X is the capital
of Y", because that's what real Wikipedia-style country pages contain.
The test is whether co-occurrence learning generalizes — given that
Germany appears in "Germany is in Europe" and Berlin appears in
"Berlin is a major German city", do their accumulated context vectors
end up similar?
"""
from __future__ import annotations

import random


# Country facts — used by templates
COUNTRIES = [
    # (country, capital, continent, language, currency, neighbor1, neighbor2, feature)
    ("USA",            "Washington",   "North America", "English",    "dollar",   "Canada",        "Mexico",        "freedom"),
    ("France",         "Paris",        "Europe",        "French",     "euro",     "Germany",       "Spain",         "wine"),
    ("Japan",          "Tokyo",        "Asia",          "Japanese",   "yen",      "Korea",         "China",         "technology"),
    ("United Kingdom", "London",       "Europe",        "English",    "pound",    "Ireland",       "France",        "history"),
    ("Italy",          "Rome",         "Europe",        "Italian",    "euro",     "France",        "Switzerland",   "art"),
    ("Spain",          "Madrid",       "Europe",        "Spanish",    "euro",     "Portugal",      "France",        "beaches"),
    ("China",          "Beijing",      "Asia",          "Mandarin",   "yuan",     "Russia",        "India",         "history"),
    ("Russia",         "Moscow",       "Europe",        "Russian",    "ruble",    "China",         "Finland",       "winters"),
    ("Brazil",         "Brasilia",     "South America", "Portuguese", "real",     "Argentina",     "Peru",          "rainforests"),
    ("Canada",         "Ottawa",       "North America", "English",    "dollar",   "USA",           "Greenland",     "mountains"),
    ("Australia",      "Canberra",     "Oceania",       "English",    "dollar",   "Indonesia",     "New Zealand",   "wildlife"),
    ("Mexico",         "Mexico City",  "North America", "Spanish",    "peso",     "USA",           "Guatemala",     "cuisine"),
    ("Argentina",      "Buenos Aires", "South America", "Spanish",    "peso",     "Brazil",        "Chile",         "tango"),
    ("South Korea",    "Seoul",        "Asia",          "Korean",     "won",      "North Korea",   "Japan",         "innovation"),
    ("Egypt",          "Cairo",        "Africa",        "Arabic",     "pound",    "Libya",         "Sudan",         "pyramids"),
    ("Greece",         "Athens",       "Europe",        "Greek",      "euro",     "Turkey",        "Albania",       "islands"),
    ("Sweden",         "Stockholm",    "Europe",        "Swedish",    "krona",    "Norway",        "Finland",       "design"),
    ("Poland",         "Warsaw",       "Europe",        "Polish",     "zloty",    "Germany",       "Ukraine",       "castles"),
    ("Turkey",         "Ankara",       "Asia",          "Turkish",    "lira",     "Greece",        "Syria",         "bazaars"),
    ("Netherlands",    "Amsterdam",    "Europe",        "Dutch",      "euro",     "Germany",       "Belgium",       "canals"),
    ("South Africa",   "Cape Town",    "Africa",        "English",    "rand",     "Namibia",       "Botswana",      "wildlife"),
    ("Vietnam",        "Hanoi",        "Asia",          "Vietnamese", "dong",     "China",         "Laos",          "cuisine"),
    ("Chile",          "Santiago",     "South America", "Spanish",    "peso",     "Argentina",     "Bolivia",       "mountains"),
    ("Kenya",          "Nairobi",      "Africa",        "Swahili",    "shilling", "Ethiopia",      "Tanzania",      "safaris"),
    ("New Zealand",    "Wellington",   "Oceania",       "English",    "dollar",   "Australia",     "Fiji",          "landscapes"),
    # ── Held-out countries (used in test, but still appear in the corpus
    # so the encoder has semantic vectors for them) ──
    ("Germany",        "Berlin",       "Europe",        "German",     "euro",     "France",        "Poland",        "engineering"),
    ("India",          "Delhi",        "Asia",          "Hindi",      "rupee",    "Pakistan",      "China",         "diversity"),
    ("Thailand",       "Bangkok",      "Asia",          "Thai",       "baht",     "Myanmar",       "Laos",          "temples"),
    ("Portugal",       "Lisbon",       "Europe",        "Portuguese", "euro",     "Spain",         "Morocco",       "beaches"),
    ("Norway",         "Oslo",         "Europe",        "Norwegian",  "krone",    "Sweden",        "Finland",       "fjords"),
    ("Peru",           "Lima",         "South America", "Spanish",    "sol",      "Chile",         "Brazil",        "ruins"),
    ("Nigeria",        "Abuja",        "Africa",        "English",    "naira",    "Cameroon",      "Benin",         "music"),
]


# Templates — each takes the country fact tuple and produces a sentence
# The variety in vocabulary is what gives Random Indexing room to learn
# semantic associations.
TEMPLATES = [
    # Direct capital statements (~3-4 per country)
    "{capital} is the capital of {country}.",
    "The capital city of {country} is {capital}.",
    "{country}'s capital is {capital}, a major urban center.",
    # Geography
    "{country} is a country located in {continent}.",
    "Travelers exploring {continent} often visit {country}.",
    "{country} shares borders with {neighbor1} and {neighbor2}.",
    "Maps of {continent} show {country} between {neighbor1} and {neighbor2}.",
    # Language and culture
    "People in {country} primarily speak {language}.",
    "The {language} language is widely spoken in {country}.",
    "{country} is famous for its {feature}.",
    "Tourists visit {country} to see its {feature}.",
    # Economy
    "The currency used in {country} is the {currency}.",
    "Goods in {country} are priced in {currency}.",
    # Capital + country together (multiple framings for co-occurrence)
    "Many residents of {country} live near {capital}.",
    "Flights to {country} usually arrive at {capital}.",
    "{capital} hosts the government of {country}.",
    "Visitors from abroad often start their {country} trip in {capital}.",
    "{capital} is the largest city in {country}.",
]


def build_corpus(seed: int = 0) -> list[str]:
    """Return a list of sentence strings."""
    rng = random.Random(seed)
    sentences = []
    for fact in COUNTRIES:
        country, capital, continent, language, currency, n1, n2, feature = fact
        ctx = {
            "country":   country,
            "capital":   capital,
            "continent": continent,
            "language":  language,
            "currency":  currency,
            "neighbor1": n1,
            "neighbor2": n2,
            "feature":   feature,
        }
        # Shuffle template order per country so adjacency varies
        templates = list(TEMPLATES)
        rng.shuffle(templates)
        for tpl in templates:
            sentences.append(tpl.format(**ctx))
    # Final shuffle so countries are interleaved (better for context windows)
    rng.shuffle(sentences)
    return sentences


def main():
    """Print the corpus for inspection."""
    corp = build_corpus()
    print(f"Generated {len(corp)} sentences from {len(COUNTRIES)} countries")
    print(f"\nFirst 10 sentences:")
    for s in corp[:10]:
        print(f"  {s}")


if __name__ == "__main__":
    main()
