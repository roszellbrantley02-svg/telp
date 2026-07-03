"""
lattice/triple_extractor.py - rule-based (subject, relation, object) extraction.

No neural net. No LLM. Just regex patterns over English sentences.

For each sentence, try each pattern. If it matches, extract the named
groups into a triple. Returns all triples that fire.

This is intentionally a STATELESS, RULE-BASED parser. The point is to
show how much you can recover from text using pure pattern matching —
no learned models anywhere in the pipeline.

Patterns target the kinds of factual sentences typical of country
articles, biographies, scientific abstracts, etc. — formulaic English
where templates are common in the wild.

To extend to new domains, add new patterns. The system has no
"understanding" — only pattern coverage.
"""
from __future__ import annotations

import re
from typing import Iterable


# ─── Vocabulary lists (for resolving entities in extracted patterns) ──


KNOWN_CONTINENTS = {
    "Europe", "Asia", "Africa", "North America", "South America",
    "Oceania", "Antarctica"
}

KNOWN_CURRENCIES = {
    "dollar", "euro", "yen", "pound", "yuan", "ruble", "real",
    "peso", "won", "krona", "zloty", "lira", "rand", "dong",
    "shilling", "rupee", "baht", "krone", "sol", "naira"
}

KNOWN_LANGUAGES = {
    "English", "French", "Japanese", "Italian", "Spanish", "Mandarin",
    "Russian", "Portuguese", "Korean", "Arabic", "Greek", "Swedish",
    "Polish", "Turkish", "Dutch", "Vietnamese", "Swahili", "German",
    "Hindi", "Thai", "Norwegian"
}


# ─── Patterns ─────────────────────────────────────────────────────


# Each pattern: (compiled regex, function(match) -> list[(s, r, o)])

def _entity_or_phrase(text: str) -> str:
    """Strip articles + capitalize properly. Leaves multi-word names intact."""
    return text.strip().rstrip('.').rstrip(',')


def _emit_capital(m):
    cap = _entity_or_phrase(m.group("capital"))
    country = _entity_or_phrase(m.group("country"))
    return [
        (country, "HAS_CAPITAL", cap),
        (cap, "HAS_CAPITAL_INV", country),
    ]


def _emit_continent(m):
    country = _entity_or_phrase(m.group("country"))
    cont = _entity_or_phrase(m.group("continent"))
    return [(country, "IN_CONTINENT", cont)]


def _emit_borders(m):
    country = _entity_or_phrase(m.group("country"))
    n1 = _entity_or_phrase(m.group("n1"))
    n2 = _entity_or_phrase(m.group("n2"))
    return [
        (country, "BORDERS", n1),
        (country, "BORDERS", n2),
    ]


def _emit_speaks(m):
    country = _entity_or_phrase(m.group("country"))
    lang = _entity_or_phrase(m.group("language"))
    return [(country, "SPEAKS", lang)]


def _emit_famous(m):
    country = _entity_or_phrase(m.group("country"))
    feature = _entity_or_phrase(m.group("feature"))
    return [(country, "FAMOUS_FOR", feature)]


def _emit_currency(m):
    country = _entity_or_phrase(m.group("country"))
    curr = _entity_or_phrase(m.group("currency"))
    return [(country, "USES_CURRENCY", curr)]


# A multi-word entity is one or more capitalized words, optionally hyphenated
_ENTITY = r"(?:[A-Z][a-zA-Z]+(?:\s+(?:of\s+)?[A-Z][a-zA-Z]+)*)"
_LOWERWORD = r"[a-z][a-zA-Z]*"


PATTERNS = [
    # Capital patterns
    (re.compile(
        rf"(?P<capital>{_ENTITY})\s+is\s+the\s+capital\s+of\s+(?P<country>{_ENTITY})"
    ), _emit_capital),
    (re.compile(
        rf"(?:The\s+)?capital\s+(?:city\s+)?of\s+(?P<country>{_ENTITY})\s+is\s+(?P<capital>{_ENTITY})"
    ), _emit_capital),
    (re.compile(
        rf"(?P<country>{_ENTITY})'s\s+capital\s+is\s+(?P<capital>{_ENTITY})"
    ), _emit_capital),
    (re.compile(
        rf"(?P<capital>{_ENTITY})\s+hosts\s+the\s+government\s+of\s+(?P<country>{_ENTITY})"
    ), _emit_capital),
    (re.compile(
        rf"(?P<capital>{_ENTITY})\s+is\s+the\s+largest\s+city\s+in\s+(?P<country>{_ENTITY})"
    ), _emit_capital),
    (re.compile(
        rf"residents\s+of\s+(?P<country>{_ENTITY})\s+live\s+near\s+(?P<capital>{_ENTITY})"
    ), _emit_capital),
    (re.compile(
        rf"[Ff]lights\s+to\s+(?P<country>{_ENTITY})\s+(?:usually\s+)?arrive\s+at\s+(?P<capital>{_ENTITY})"
    ), _emit_capital),
    (re.compile(
        rf"start\s+their\s+(?P<country>{_ENTITY})\s+trip\s+in\s+(?P<capital>{_ENTITY})"
    ), _emit_capital),
    # Continent patterns
    (re.compile(
        rf"(?P<country>{_ENTITY})\s+is\s+a\s+country\s+located\s+in\s+(?P<continent>{_ENTITY})"
    ), _emit_continent),
    (re.compile(
        rf"exploring\s+(?P<continent>{_ENTITY})\s+often\s+visit\s+(?P<country>{_ENTITY})"
    ), _emit_continent),
    (re.compile(
        rf"Maps\s+of\s+(?P<continent>{_ENTITY})\s+show\s+(?P<country>{_ENTITY})"
    ), _emit_continent),
    # Borders
    (re.compile(
        rf"(?P<country>{_ENTITY})\s+shares\s+borders\s+with\s+(?P<n1>{_ENTITY})\s+and\s+(?P<n2>{_ENTITY})"
    ), _emit_borders),
    # Language
    (re.compile(
        rf"[Pp]eople\s+in\s+(?P<country>{_ENTITY})\s+(?:primarily\s+)?speak\s+(?P<language>{_ENTITY})"
    ), _emit_speaks),
    (re.compile(
        rf"The\s+(?P<language>{_ENTITY})\s+language\s+is\s+widely\s+spoken\s+in\s+(?P<country>{_ENTITY})"
    ), _emit_speaks),
    # Famous-for
    (re.compile(
        rf"(?P<country>{_ENTITY})\s+is\s+famous\s+for\s+its\s+(?P<feature>{_LOWERWORD})"
    ), _emit_famous),
    (re.compile(
        rf"Tourists\s+visit\s+(?P<country>{_ENTITY})\s+to\s+see\s+its\s+(?P<feature>{_LOWERWORD})"
    ), _emit_famous),
    # Currency
    (re.compile(
        rf"currency\s+used\s+in\s+(?P<country>{_ENTITY})\s+is\s+the\s+(?P<currency>{_LOWERWORD})"
    ), _emit_currency),
    (re.compile(
        rf"Goods\s+in\s+(?P<country>{_ENTITY})\s+are\s+priced\s+in\s+(?P<currency>{_LOWERWORD})"
    ), _emit_currency),
]


# ─── Main API ─────────────────────────────────────────────────────


def extract_triples(sentence: str) -> list[tuple[str, str, str]]:
    """Apply all patterns to a single sentence, return all triples that fire."""
    triples = []
    for pattern, emitter in PATTERNS:
        for m in pattern.finditer(sentence):
            triples.extend(emitter(m))
    # Deduplicate while preserving order
    seen = set()
    out = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def extract_all(sentences: Iterable[str]) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """For a list of sentences, return [(sentence, triples), ...]."""
    return [(s, extract_triples(s)) for s in sentences]


# ─── Self-test ────────────────────────────────────────────────────


def main():
    """Test extractor on a few sample sentences."""
    test_sentences = [
        "Berlin is the capital of Germany.",
        "The capital of France is Paris.",
        "Japan's capital is Tokyo.",
        "Germany is a country located in Europe.",
        "Spain shares borders with Portugal and France.",
        "People in Italy primarily speak Italian.",
        "The Japanese language is widely spoken in Japan.",
        "Egypt is famous for its pyramids.",
        "The currency used in Sweden is the krona.",
        "The Eiffel Tower was built in 1889.",   # should NOT match country patterns
        "Random sentence with no extractable triples.",
    ]
    print("Testing triple extractor on sample sentences:\n")
    for s in test_sentences:
        triples = extract_triples(s)
        print(f"  IN:  {s}")
        if triples:
            for t in triples:
                print(f"  OUT: {t}")
        else:
            print(f"  OUT: (no triples extracted)")
        print()


if __name__ == "__main__":
    main()
