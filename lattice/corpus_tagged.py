"""
lattice/corpus_tagged.py - corpus with per-sentence (subject, relation, object) tags.

Same generator as corpus.py, but every sentence is paired with the
list of relational triples it expresses. This is what lets relation-
bound encoding work — we know which words played which roles.

In a real-world setting, these triples would come from dependency
parsing (or smaller LMs used purely as parsers, which is a much
weaker dependency than using an LLM for semantics).
"""
from __future__ import annotations

import random
from lattice.corpus import COUNTRIES


# Each entry: (template_string, list_of_triple_specs)
# A triple_spec is (subject_field, relation_name, object_field).
# Fields refer to the per-country fact dict keys.
TAGGED_TEMPLATES = [
    # Capital-relation templates
    ("{capital} is the capital of {country}.",
        [("capital", "HAS_CAPITAL_INV", "country"),
         ("country", "HAS_CAPITAL", "capital")]),
    ("The capital city of {country} is {capital}.",
        [("country", "HAS_CAPITAL", "capital"),
         ("capital", "HAS_CAPITAL_INV", "country")]),
    ("{country}'s capital is {capital}, a major urban center.",
        [("country", "HAS_CAPITAL", "capital")]),
    ("{capital} hosts the government of {country}.",
        [("capital", "HAS_CAPITAL_INV", "country")]),
    ("{capital} is the largest city in {country}.",
        [("capital", "HAS_CAPITAL_INV", "country")]),
    # Geography
    ("{country} is a country located in {continent}.",
        [("country", "IN_CONTINENT", "continent")]),
    ("Travelers exploring {continent} often visit {country}.",
        [("country", "IN_CONTINENT", "continent")]),
    ("{country} shares borders with {neighbor1} and {neighbor2}.",
        [("country", "BORDERS", "neighbor1"),
         ("country", "BORDERS", "neighbor2")]),
    ("Maps of {continent} show {country} between {neighbor1} and {neighbor2}.",
        [("country", "IN_CONTINENT", "continent"),
         ("country", "BORDERS", "neighbor1"),
         ("country", "BORDERS", "neighbor2")]),
    # Language
    ("People in {country} primarily speak {language}.",
        [("country", "SPEAKS", "language")]),
    ("The {language} language is widely spoken in {country}.",
        [("country", "SPEAKS", "language")]),
    # Feature
    ("{country} is famous for its {feature}.",
        [("country", "FAMOUS_FOR", "feature")]),
    ("Tourists visit {country} to see its {feature}.",
        [("country", "FAMOUS_FOR", "feature")]),
    # Currency
    ("The currency used in {country} is the {currency}.",
        [("country", "USES_CURRENCY", "currency")]),
    ("Goods in {country} are priced in {currency}.",
        [("country", "USES_CURRENCY", "currency")]),
    # Capital+country together
    ("Many residents of {country} live near {capital}.",
        [("country", "HAS_CAPITAL", "capital")]),
    ("Flights to {country} usually arrive at {capital}.",
        [("country", "HAS_CAPITAL", "capital")]),
    ("Visitors from abroad often start their {country} trip in {capital}.",
        [("country", "HAS_CAPITAL", "capital")]),
]


def build_tagged_corpus(seed: int = 0) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Return list of (sentence_text, triples) pairs.

    triples is a list of (subject_word, relation_name, object_word).
    """
    rng = random.Random(seed)
    pairs = []
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
        tpls = list(TAGGED_TEMPLATES)
        rng.shuffle(tpls)
        for tpl_str, triple_specs in tpls:
            sentence = tpl_str.format(**ctx)
            triples = []
            for s_field, rel, o_field in triple_specs:
                subj = ctx[s_field]
                obj  = ctx[o_field]
                triples.append((subj, rel, obj))
            pairs.append((sentence, triples))
    rng.shuffle(pairs)
    return pairs


def main():
    pairs = build_tagged_corpus()
    print(f"Generated {len(pairs)} tagged sentences.\n")
    print("Sample:")
    for s, ts in pairs[:5]:
        print(f"  S: {s}")
        for t in ts:
            print(f"     triple: {t}")
        print()


if __name__ == "__main__":
    main()
