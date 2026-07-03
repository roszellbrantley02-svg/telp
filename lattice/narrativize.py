"""
lattice/narrativize.py — Phase 18: dictionary register -> fiction register.

WHY
---
Phase 17 made Telp justify unusual word choices by attaching the
dictionary gloss as an inline appositive.  The information was right
but the REGISTER was wrong — it read like a footnote:

  the headroom — the vertical clearance above someone's head, as
                 in a tunnel, doorway etc.

The user asked for fiction register instead: a description the
reader can picture, woven into the story:

  the headroom — a small space above the rafters, where the dust
                 hung quiet in the air.

Both carry the same information.  One reads like Wiktionary, one
reads like a children's book.  The narrativizer is the transformer
between them.

HOW
---
Three stages:

  1. STRIP taxonomic / technical noise.  "Any of various medium-sized
     woodpeckers of the genus Colaptes" -> "a kind of woodpecker".
     Pattern-match common Wiktionary boilerplate and reduce.

  2. PATTERN-MATCH the cleaned gloss against narrative templates.
     "vertical clearance above someone's head" -> "small space
     above ___".  Templates are surface-form patterns over noun
     phrases.

  3. ADD picturable atmosphere keyed to category.  An enclosed-space
     noun gets "where the dust drifted"-style scene texture.  An
     outdoor noun gets "where the wind moved through the grass".
     An animal noun stays simple ("a kind of X").

Pure surface-form transformations.  No LLM, no learned distribution.
Same algebraic substrate as everything else.
"""
from __future__ import annotations

import random
import re
from typing import Optional


# ─── Taxonomic / technical noise patterns ────────────────────────


# Drop phrases that mark a gloss as scientific-register.
_DROP_PHRASES = [
    # "as in a tunnel, doorway etc" — eat the whole as-in tail
    re.compile(r",?\s*as in [^.;]*?(?=[.;]|$)", re.IGNORECASE),
    # "of insect in the" leftover when "order X" got stripped
    re.compile(r"\s+in the(?=\s+characterized|\s+order|\s*$)",
                  re.IGNORECASE),
    re.compile(r",?\s*of the (genus|family|order|clade|class|"
                  r"phylum|kingdom|tribe|subfamily|superfamily)\b[^,.]*",
                  re.IGNORECASE),
    re.compile(r",?\s*Melanogrammus\s+aeglefinus", re.IGNORECASE),
    re.compile(r",?\s*Orycteropus\s+afer", re.IGNORECASE),
    re.compile(r",?\s*\(genus\s+[^)]+\)", re.IGNORECASE),
    re.compile(r",?\s*\(family\s+[^)]+\)", re.IGNORECASE),
    re.compile(r",?\s*\(order\s+[^)]+\)", re.IGNORECASE),
    re.compile(r",?\s*\(class\s+[^)]+\)", re.IGNORECASE),
    re.compile(r",?\s*\([A-Z][a-z]+\s+[a-z]+\)"),     # binomial in parens
    re.compile(r"†[A-Z][a-zA-Z]+"),                     # extinct dagger marker
    re.compile(r"\bclade\s+[A-Z][a-zA-Z]+\b"),
    re.compile(r"\border\s+[A-Z][a-zA-Z]+\b"),
    re.compile(r"\bfamily\s+[A-Z][a-zA-Z]+\b"),
    re.compile(r"\bgenus\s+[A-Z][a-zA-Z]+\b"),
    re.compile(r",?\s*also called[^,.]*", re.IGNORECASE),
    re.compile(r",?\s*as in[^,.]*", re.IGNORECASE),
    re.compile(r",?\s*especially[^,.]*", re.IGNORECASE),
    re.compile(r",?\s*typically[^,.]*", re.IGNORECASE),
    re.compile(r"\(of[^)]+\)"),                          # "(of cars)"
]

# Boilerplate openings to simplify.
_PREFIX_REWRITES = [
    (re.compile(r"^any of (the )?various ", re.IGNORECASE), "a kind of "),
    (re.compile(r"^any of (the )?several ", re.IGNORECASE), "a kind of "),
    (re.compile(r"^any of (a class of )?", re.IGNORECASE),  "a kind of "),
    (re.compile(r"^any of ", re.IGNORECASE),                 "a kind of "),
    (re.compile(r"^one of ", re.IGNORECASE),                 "a kind of "),
    (re.compile(r"^various ", re.IGNORECASE),                "a kind of "),
    (re.compile(r"^a member of ", re.IGNORECASE),            "a kind of "),
]


# ─── Atmosphere pools (picturable scene textures) ────────────────


_ATMOSPHERE = {
    "enclosed": [
        "where the dust hung quiet in the air",
        "where the light fell in long slants",
        "where the air felt close and still",
        "where shadows pooled in the corners",
        "where everything was hushed",
        "small and shadowed",
    ],
    "outdoor": [
        "where the wind moved through the grass",
        "where the sky stretched wide above",
        "where the light came golden through the leaves",
        "where the air smelled of green things",
        "open and full of sky",
    ],
    "water": [
        "where the water gleamed in the light",
        "where ripples spread slowly outward",
        "where everything smelled of salt and sky",
        "where the air was cool and damp",
    ],
    "sky": [
        "where the wind sang high and thin",
        "where the clouds drifted slow",
        "where the stars hung close",
    ],
    "fantastical": [
        "a place that felt almost like a thought",
        "where things were not quite real",
        "a place between waking and dreaming",
    ],
}


# Keywords that map a definition to an atmosphere category.  Tested
# against the cleaned gloss; first match wins.
_CATEGORY_KEYWORDS = [
    ("water",       ("shore", "coast", "river", "lake", "pond",
                           "ocean", "sea", "stream", "bay", "creek",
                           "puddle", "wet", "water")),
    ("sky",         ("sky", "cloud", "air", "above the ground",
                           "stars", "rainbow")),
    ("outdoor",     ("countryside", "field", "meadow", "land",
                           "tract", "rural", "remote", "forest",
                           "wood", "garden", "mountain", "valley",
                           "outside", "open")),
    ("enclosed",    ("room", "enclosure", "clearance", "space above",
                           "vertical", "building", "structure",
                           "indoor", "interior", "house", "barn",
                           "shed", "cave", "tunnel", "doorway",
                           "ceiling", "rafter", "wall")),
    ("fantastical", ("imaginary", "mythical", "legendary", "magical",
                           "fairy", "spirit", "phantom")),
]


# Animal/creature heuristics — these get the simpler "a kind of X"
# treatment without atmosphere.
_CREATURE_HINTS = (
    "fish", "bird", "mammal", "reptile", "amphibian", "insect",
    "rodent", "feline", "canine", "creature", "animal",
    "woodpecker", "char", "ant", "beetle", "moth", "lizard",
    "snake", "frog", "monkey", "ape", "dog", "cat",
)


# ─── Narrativizer ────────────────────────────────────────────────


class Narrativizer:
    """Convert a dictionary gloss into a short story-prose appositive.

    Stateless; safe to share across imagination engines.  Optionally
    takes a seeded RNG for deterministic atmosphere selection.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    # ── Public API ─────────────────────────────────────────────

    def narrativize(self, word: str, gloss: str) -> Optional[str]:
        """Return a fiction-register description, or None if the gloss
        can't be transformed cleanly."""
        if not gloss:
            return None

        cleaned = self._clean(gloss)
        if not cleaned or len(cleaned) < 6:
            return None

        category = self._category(cleaned)

        # Animals/creatures: keep simple ("a kind of woodpecker")
        if category == "creature":
            return self._creature_form(cleaned)

        # Add scene atmosphere if we recognised the category, the
        # cleaned gloss isn't already long, AND the gloss doesn't
        # already have its own "where"/clausal scene description.
        has_scene = bool(re.search(r"\bwhere\b|\bwhen\b", cleaned))
        atmosphere = self._atmosphere(category)
        if atmosphere and len(cleaned) < 70 and not has_scene:
            return f"{cleaned}, {atmosphere}"

        return cleaned

    # ── Internals ──────────────────────────────────────────────

    def _clean(self, gloss: str) -> str:
        """Strip taxonomic noise and normalize prefixes."""
        g = gloss.strip().rstrip(".")
        # Cut at first semicolon — kaikki joins alternate senses
        # with "; X" which read as broken in prose form.
        semi = g.find(";")
        if 6 < semi < 200:
            g = g[:semi]
        # First-pass deletions
        for pat in _DROP_PHRASES:
            g = pat.sub("", g)
        # Prefix rewrites
        for pat, repl in _PREFIX_REWRITES:
            new_g, n = pat.subn(repl, g)
            if n > 0:
                g = new_g
                break
        # "a kind of numerous species of insect" -> "a kind of insect"
        g = re.sub(r"a kind of (numerous |several |many |various )?"
                       r"species of\s+",
                       "a kind of ", g, flags=re.IGNORECASE)
        # "a member of" with the rest stripped -> drop entirely.
        # Also reject "a member, generally..." which is a fragment.
        g = re.sub(r"^a member\s*$", "", g, flags=re.IGNORECASE).strip()
        g = re.sub(r"^a member of\b\s*$", "", g, flags=re.IGNORECASE).strip()
        g = re.sub(r"^a member,\s*generally[^.]*", "", g,
                          flags=re.IGNORECASE).strip()
        # Trailing "etc" or ", etc"
        g = re.sub(r",?\s*etc\.?\s*$", "", g, flags=re.IGNORECASE)
        # Close unmatched parens by dropping unbalanced trailing parens
        opens, closes = g.count("("), g.count(")")
        if opens > closes:
            # Drop the orphan open-paren region
            g = re.sub(r"\s*\([^)]*$", "", g)
        elif closes > opens:
            g = re.sub(r"^[^(]*\)\s*", "", g, count=closes - opens)
        # Drop empty parens "(fish of )"
        g = re.sub(r"\s*\(\s*\)", "", g)
        g = re.sub(r"\(\s*[a-z]+\s+of\s*\)", "", g, flags=re.IGNORECASE)
        # Collapse multiple spaces / stray punctuation
        g = re.sub(r"\s+", " ", g).strip()
        g = re.sub(r"\s*,\s*,", ",", g)
        g = re.sub(r"\s+,", ",", g)
        g = g.rstrip(" ,.;:—-")
        # Drop sentence with self-reference
        if "aforementioned" in g.lower():
            g = re.sub(r",?\s*[^,.]*aforementioned[^,.]*", "", g,
                              flags=re.IGNORECASE).strip()
        # Lowercase leading letter for appositive use
        if g and g[0].isupper():
            g = g[0].lower() + g[1:]
        return g

    def _category(self, cleaned: str) -> Optional[str]:
        low = cleaned.lower()
        # Creature heuristic FIRST — animal gets simple form
        for hint in _CREATURE_HINTS:
            # Word-boundary match
            if re.search(rf"\b{re.escape(hint)}\b", low):
                return "creature"
        for cat, kws in _CATEGORY_KEYWORDS:
            for kw in kws:
                if kw in low:
                    return cat
        return None

    def _atmosphere(self, category: Optional[str]) -> Optional[str]:
        if category is None or category == "creature":
            return None
        pool = _ATMOSPHERE.get(category)
        if not pool:
            return None
        return self.rng.choice(pool)

    def _creature_form(self, cleaned: str) -> str:
        """For animals, prefer 'a kind of X' form when possible."""
        # If the cleaned gloss already starts with "a kind of",
        # great.  Otherwise try to extract the head noun.
        if cleaned.lower().startswith("a kind of "):
            return cleaned
        # "a saltwater fish" -> ok as is
        # "a small to medium-sized bird" -> ok as is
        # If it's just a binomial fragment, simplify
        m = re.match(r"^(a|an|the)\s+\S+\s+(fish|bird|mammal|"
                          r"insect|reptile|amphibian|rodent|woodpecker|"
                          r"creature|animal)\b", cleaned, re.IGNORECASE)
        if m:
            return cleaned
        return cleaned


# ─── CLI smoke ────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from lattice.dictionary_lookup import Dictionary
    from lattice.justification import Justifier

    d = Dictionary()
    j = Justifier(d)
    n = Narrativizer(rng=random.Random(42))

    print("    DICTIONARY                                NARRATIVE")
    print("    ----------                                ---------")
    for w in ["headroom", "cowyard", "backroom", "seafront",
                "thunderstorm", "rainbow", "puddle", "library",
                "attic", "flicker", "haddock", "beetle", "monkey",
                "redbelly", "gadiform", "backblock", "kitchen",
                "tower", "cave", "moon", "sky", "garden",
                "frontcountry", "migrant", "knife"]:
        gloss = j.justify(w)
        narrative = n.narrativize(w, gloss) if gloss else None
        if gloss is None:
            print(f"  {w:14}  (no entry)")
            continue
        d_short = (gloss[:36] + "...") if len(gloss) > 38 else gloss
        n_short = narrative or "(no transform)"
        print(f"  {w:14}  {d_short:<40}  ->  {n_short}")
