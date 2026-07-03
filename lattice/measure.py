"""
lattice/measure.py - numeric magnitudes from remembered prose.

"Which is bigger, Jupiter or Earth?" is a COMPUTATION, not a retrieval:
pull a number for each entity in the SAME dimension (diameter vs diameter,
never diameter vs height), normalize units to SI, compare, show both
numbers. Two evidence paths, tried in order:

  1. RELATIONAL: one remembered sentence names both entities with a
     "N times" relation ("Jupiter's diameter is 11 times that of Earth")
     - the comparison is already stated; cite it.
  2. TWO-VALUE: find "<entity> ... <number> <unit>" near a dimension word
     for each entity separately, normalize, compare.

Every answer carries its evidence rows - same law as everything else.
"""
from __future__ import annotations

import re

# unit -> factor to the dimension's SI base
_LENGTH_UNITS = {
    "km": 1000.0, "kilometre": 1000.0, "kilometres": 1000.0,
    "kilometer": 1000.0, "kilometers": 1000.0,
    "m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0,
    "cm": 0.01, "centimetres": 0.01, "centimeters": 0.01,
    "mm": 0.001, "mi": 1609.34, "mile": 1609.34, "miles": 1609.34,
    "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
}
_MASS_UNITS = {
    "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
    "g": 0.001, "gram": 0.001, "grams": 0.001,
    "tonne": 1000.0, "tonnes": 1000.0, "ton": 907.18, "tons": 907.18,
    "lb": 0.4536, "lbs": 0.4536, "pound": 0.4536, "pounds": 0.4536,
}
_SPEED_UNITS = {
    "km/h": 1.0, "kph": 1.0, "mph": 1.609, "m/s": 3.6,
}

_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}

# dimension classes tried in priority order per question verb; a class
# only wins when BOTH entities yield a value in it
_DIMENSIONS = {
    "size":   [("diameter", _LENGTH_UNITS), ("radius", _LENGTH_UNITS),
               ("height", _LENGTH_UNITS), ("length", _LENGTH_UNITS),
               ("mass", _MASS_UNITS), ("weigh", _MASS_UNITS)],
    "mass":   [("mass", _MASS_UNITS), ("weigh", _MASS_UNITS)],
    "height": [("height", _LENGTH_UNITS), ("tall", _LENGTH_UNITS),
               ("elevation", _LENGTH_UNITS)],
    "length": [("length", _LENGTH_UNITS), ("long", _LENGTH_UNITS)],
    "speed":  [("speed", _SPEED_UNITS), ("velocity", _SPEED_UNITS)],
}

_VERB_DIM = {
    "bigger": "size", "larger": "size", "smaller": "size",
    "heavier": "mass", "lighter": "mass",
    "taller": "height", "shorter": "height",
    "longer": "length", "faster": "speed", "slower": "speed",
}
_WANT_MIN = {"smaller", "lighter", "shorter", "slower"}


def _num_unit_re(units: dict) -> re.Pattern:
    alts = sorted((re.escape(u) for u in units), key=len, reverse=True)
    return re.compile(
        r"(\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion|trillion)?\s*"
        r"(" + "|".join(alts) + r")\b", re.I)


def _rows_for(agent, entity: str) -> list[str]:
    """Rows that NAME the entity (prefix match, knowledge sources only)."""
    key = entity.lower()
    key = key[:5] if len(key) > 5 else key
    out = []
    for t, s in zip(agent.lattice._texts, agent.lattice._sources):
        if s.startswith(("user_msg", "agent_response", "conversation_turn",
                         "image:", "video:")):
            continue
        if key in t.lower():
            out.append(t)
    return out


def find_magnitude(agent, entity: str, dim_word: str, units: dict):
    """Best (value_in_SI, raw_text, evidence_row) for entity in one
    dimension, or None. The number must share a sentence with both the
    entity and the dimension word."""
    pat = _num_unit_re(units)
    ent_key = entity.lower()
    ent_key = ent_key[:5] if len(ent_key) > 5 else ent_key
    for row in _rows_for(agent, entity):
        anchored = row.lower().startswith(ent_key)
        for sent in re.split(r"(?<=[.!?])\s+", row):
            sl = sent.lower()
            d_pos = sl.find(dim_word)
            if d_pos < 0:
                continue
            # the entity must OWN the dimension: named before the
            # dimension word (or the row is anchored "Entity: ...").
            # "Mars has a diameter of 6,779 km, half of Earth's" must
            # never become Earth's diameter.
            e_pos = sl.find(ent_key)
            if not (anchored or (0 <= e_pos < d_pos)):
                continue
            m = pat.search(sent, d_pos)
            if not m:
                continue
            val = float(m.group(1).replace(",", ""))
            if m.group(2):
                val *= _SCALE[m.group(2).lower()]
            val *= units[m.group(3).lower()]
            return {"value": val, "raw": m.group(0), "evidence": row}
    return None


def _relational(agent, a: str, b: str):
    """A single remembered sentence relating both entities' SIZE by
    'its <dimension> is N times that of' - NOT any old 'N times' ("at a
    distance roughly 30 times the width of Earth" is about the DISTANCE,
    and must never make the Moon bigger than Earth)."""
    ka = a.lower()[:5] if len(a) > 5 else a.lower()
    kb = b.lower()[:5] if len(b) > 5 else b.lower()
    fractions = {"half": 0.5, "one-quarter": 0.25, "one quarter": 0.25,
                 "a quarter": 0.25, "one-third": 1 / 3, "one third": 1 / 3,
                 "one-fifth": 0.2, "one fifth": 0.2, "one-tenth": 0.1,
                 "one tenth": 0.1, "a tenth": 0.1, "twice": 2.0,
                 "double": 2.0}
    rel_re = re.compile(
        r"\b(?:diameter|radius|mass|size|width|height|length|volume)\s+is\s+"
        r"(?:about\s+|nearly\s+|roughly\s+|approximately\s+)?"
        r"(?:(\d[\d,]*(?:\.\d+)?)\s+times"           # "11 times"
        r"|(\d[\d,]*(?:\.\d+)?)\s*%\s+(?:that\s+)?of"  # "1.2% of"
        r"|(" + "|".join(re.escape(f) for f in fractions) + r"))\b")
    for t, s in zip(agent.lattice._texts, agent.lattice._sources):
        if s.startswith(("user_msg", "agent_response", "conversation_turn")):
            continue
        tl = t.lower()
        if ka not in tl or kb not in tl:
            continue
        m = rel_re.search(tl)
        if not m:
            continue
        # the subject owns the dimension (named before it); the other
        # entity must sit in the "that of Y" position, after the relation
        pa, pb = tl.find(ka), tl.find(kb)
        subj, other = (a, b) if pa < pb else (b, a)
        p_other = pb if subj == a else pa
        if not (min(pa, pb) < m.start() and p_other > m.end() - 10):
            continue
        if m.group(1):
            n = float(m.group(1).replace(",", ""))
        elif m.group(2):
            n = float(m.group(2).replace(",", "")) / 100.0
        else:
            n = fractions[m.group(3)]
        bigger = subj if n > 1 else other
        return {"n": n, "subject": subj, "bigger": bigger, "evidence": t}
    return None


def compare_magnitude(agent, a: str, b: str, verb: str) -> dict | None:
    """Deterministic comparison verdict, or None when evidence is missing.
    Returns {"winner", "loser", "body", "evidence": [rows]}."""
    dim = _VERB_DIM.get(verb.lower())
    if dim is None:
        return None
    want_min = verb.lower() in _WANT_MIN

    rel = _relational(agent, a, b)
    if rel is not None:
        big, small = rel["bigger"], (b if rel["bigger"] == a else a)
        winner = small if want_min else big
        return {"winner": winner, "loser": big if winner == small else small,
                "body": (f"{winner.capitalize()} - my memory states it "
                         f"directly: \"{rel['evidence'].strip()}\""),
                "evidence": [rel["evidence"]]}

    for dim_word, units in _DIMENSIONS[dim]:
        ma = find_magnitude(agent, a, dim_word, units)
        mb = find_magnitude(agent, b, dim_word, units)
        if ma is None or mb is None:
            continue
        big = a if ma["value"] > mb["value"] else b
        small = b if big == a else a
        winner = small if want_min else big
        return {"winner": winner, "loser": a if winner == b else b,
                "body": (f"{winner.capitalize()} - comparing {dim_word}: "
                         f"{a} is {ma['raw']} and {b} is {mb['raw']}."),
                "evidence": [ma["evidence"], mb["evidence"]]}
    return None
