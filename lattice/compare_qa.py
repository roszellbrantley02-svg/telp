"""
lattice/compare_qa.py - comparative reasoning over the claim store.

Single-value QA returns one fact.  Comparative queries fetch TWO
facts and compare them:

    "Was Einstein born before Bach?"        - compare born_year
    "Who is older, Einstein or Bach?"       - compare born_year, older=lower
    "Was Einstein born before 1900?"        - compare to literal year
    "Who was born first, X or Y?"           - born_year, earliest wins

All of these reduce to: pull (X, R, vX) and (Y, R, vY) — or
(X, R, vX) and a literal — then run a numeric/temporal comparison.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from lattice.structured_qa import StructuredQA


@dataclass
class CompareResult:
    kind:       str           # "before", "after", "older", "younger", etc
    a:          str
    b:          str
    a_value:    str | int | None
    b_value:    str | int | None
    winner:     str | None    # the entity that satisfies the comparison
    explain:    str
    confidence: float = 1.0


_NAME = r"(?P<{N}>[A-Z][\w-]+(?:\s+[A-Z][\w-]+){{0,3}})"


# Patterns that take TWO entities and a comparison verb.
_TWO_ENTITY_PATTERNS = [
    # "Was X born before/after Y?"
    {
        "pattern": re.compile(
            r"\bwas\s+" + _NAME.format(N="a") +
            r"\s+born\s+(?P<op>before|after|in\s+the\s+same\s+year\s+as)\s+" +
            _NAME.format(N="b") + r"\??",
            re.I),
        "kind": "born_compare",
    },
    # "Who is older / younger, X or Y?"
    {
        "pattern": re.compile(
            r"\bwho\s+is\s+(?P<op>older|younger),?\s+" +
            _NAME.format(N="a") + r"\s+or\s+" + _NAME.format(N="b") + r"\??",
            re.I),
        "kind": "age_compare",
    },
    # "Who was born first/earlier, X or Y?"
    {
        "pattern": re.compile(
            r"\bwho\s+was\s+born\s+(?P<op>first|earlier|later|last),?\s+" +
            _NAME.format(N="a") + r"\s+or\s+" + _NAME.format(N="b") + r"\??",
            re.I),
        "kind": "born_order",
    },
    # "Did X live before Y?"
    {
        "pattern": re.compile(
            r"\bdid\s+" + _NAME.format(N="a") +
            r"\s+live\s+(?P<op>before|after)\s+" + _NAME.format(N="b") + r"\??",
            re.I),
        "kind": "lived_compare",
    },
]


# Patterns that take ONE entity + a literal year/number.
_ENTITY_LITERAL_PATTERNS = [
    # "Was X born before 1800?"
    {
        "pattern": re.compile(
            r"\bwas\s+" + _NAME.format(N="a") +
            r"\s+born\s+(?P<op>before|after)\s+(?P<year>\d{3,4})\??",
            re.I),
        "kind": "born_vs_year",
    },
]


class CompareQA:
    def __init__(self, structured: StructuredQA):
        self.structured = structured

    def _lookup_year(self, entity: str, verb: str) -> int | None:
        """Get the year (int) for (entity, verb=born_year/died_year, ?)."""
        e = entity.lower()
        e_surname = e.split()[-1]
        for s, v, o in self.structured.claim_triple:
            if v != verb:
                continue
            s_lo = s.lower()
            if e in s_lo or s_lo in e or e_surname == s_lo.split()[-1]:
                try:
                    return int(o)
                except ValueError:
                    continue
        return None

    def answer(self, query: str) -> CompareResult | None:
        # Two-entity patterns
        for shape in _TWO_ENTITY_PATTERNS:
            m = shape["pattern"].search(query)
            if not m:
                continue
            return self._handle_two_entity(m, shape["kind"])
        for shape in _ENTITY_LITERAL_PATTERNS:
            m = shape["pattern"].search(query)
            if not m:
                continue
            return self._handle_entity_literal(m, shape["kind"])
        return None

    def _handle_two_entity(self, m: re.Match, kind: str) -> CompareResult | None:
        a = m.group("a").strip()
        b = m.group("b").strip()
        op = m.group("op").lower()

        # All current kinds compare born_year.  Future kinds (height,
        # founding year, etc) would route differently.
        ya = self._lookup_year(a, "born_year")
        yb = self._lookup_year(b, "born_year")
        if ya is None or yb is None:
            missing = []
            if ya is None: missing.append(a)
            if yb is None: missing.append(b)
            return CompareResult(
                kind=kind, a=a, b=b, a_value=ya, b_value=yb,
                winner=None,
                explain=f"I don't have birth years for: {', '.join(missing)}",
                confidence=0.0,
            )

        if kind in ("born_compare", "lived_compare"):
            if op == "before":
                ok = ya < yb
                winner = a if ok else b
                explain = (f"{a} was born in {ya}; {b} was born in {yb}. "
                             f"{a} was born {'BEFORE' if ok else 'AFTER'} {b}.")
            elif op == "after":
                ok = ya > yb
                winner = a if ok else b
                explain = (f"{a} was born in {ya}; {b} was born in {yb}. "
                             f"{a} was born {'AFTER' if ok else 'BEFORE'} {b}.")
            else:  # same year
                ok = ya == yb
                winner = a if ok else None
                explain = (f"{a} was born in {ya}; {b} was born in {yb}. "
                             f"{'Same year.' if ok else 'Different years.'}")
            return CompareResult(kind=kind, a=a, b=b, a_value=ya, b_value=yb,
                                     winner=winner if ok else None,
                                     explain=explain)

        if kind == "age_compare":
            # Older = earlier birth year.
            older = a if ya < yb else b
            younger = b if older == a else a
            winner = older if op == "older" else younger
            explain = (f"{a} was born in {ya}; {b} was born in {yb}. "
                         f"{winner} is {op}.")
            return CompareResult(kind=kind, a=a, b=b, a_value=ya, b_value=yb,
                                     winner=winner, explain=explain)

        if kind == "born_order":
            first = a if ya < yb else b
            last = b if first == a else a
            if op in ("first", "earlier"):
                winner = first
            else:
                winner = last
            explain = (f"{a} was born in {ya}; {b} was born in {yb}. "
                         f"{winner} was born {op}.")
            return CompareResult(kind=kind, a=a, b=b, a_value=ya, b_value=yb,
                                     winner=winner, explain=explain)

        return None

    def _handle_entity_literal(self, m: re.Match,
                                  kind: str) -> CompareResult | None:
        a = m.group("a").strip()
        op = m.group("op").lower()
        year_lit = int(m.group("year"))
        ya = self._lookup_year(a, "born_year")
        if ya is None:
            return CompareResult(
                kind=kind, a=a, b=str(year_lit),
                a_value=None, b_value=year_lit, winner=None,
                explain=f"I don't have a birth year for {a}.",
                confidence=0.0,
            )
        if op == "before":
            ok = ya < year_lit
        else:
            ok = ya > year_lit
        return CompareResult(
            kind=kind, a=a, b=str(year_lit), a_value=ya, b_value=year_lit,
            winner=a if ok else None,
            explain=(f"{a} was born in {ya}. "
                       f"{'Yes' if ok else 'No'} — {a} was born "
                       f"{op} {year_lit}."),
        )
