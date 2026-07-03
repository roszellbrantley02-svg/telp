"""
lattice/aggregate_qa.py - enumerative / set-returning queries.

Single-fact QA returns one answer.  Aggregate queries return a SET:

    "List all the composers"               -> all S with (S, is, composer)
    "Which scientists were German?"        -> all S with (S, is, scientist*) ∩ (S, born, Germany)
    "How many countries do you know?"      -> count of distinct subjects with capital claims
    "Who was born in Germany?"             -> all S with (S, born, Germany)
    "What did Einstein develop?"           -> all O with (Einstein, developed, O)

The KG already has the data — this module just provides a query path
that enumerates over it instead of finding a single best match.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from lattice.structured_qa import StructuredQA, _PROFESSION_LIST


@dataclass
class AggregateResult:
    pattern: str
    filter_desc: str        # human-readable filter, e.g. "scientists born in Germany"
    items: list[str]        # the matched entities / objects
    count: int

    def format(self, max_items: int = 12) -> str:
        # COUNT-only result (e.g. "how many facts") - items is empty
        # but count carries the answer.
        if not self.items and self.count > 0:
            return f"I know {self.count} {self.filter_desc}."
        if not self.items:
            return f"I don't have any {self.filter_desc} in my claims."
        n = len(self.items)
        if n == 1:
            return f"I know 1 {self.filter_desc}: {self.items[0]}."
        if n <= max_items:
            joined = ", ".join(self.items)
            return f"I know {n} {self.filter_desc}: {joined}."
        head = ", ".join(self.items[:max_items])
        return (f"I know {n} {self.filter_desc}. First {max_items}: "
                  f"{head}, ...")


# Recognized profession words for "list all X" / "which X" queries.
_PROFESSION_SET = set(_PROFESSION_LIST.split("|"))


# Patterns the aggregate module recognises.
_PATTERNS = [
    # "List all the composers" / "Show me all writers"
    {
        "name": "list_profession",
        "pattern": re.compile(
            r"\b(?:list|show|tell\s+me\s+about|name|who\s+are)\s+(?:all\s+)?"
            r"(?:the\s+)?(?P<prof>[\w-]+?)s?\??$",
            re.I,
        ),
    },
    # "Which scientists were German?" / "Which composers came from Poland?"
    # Allow optional "born in" between verb and country - so the
    # regex can capture "Which scientists were born in England?" too.
    {
        "name": "which_profession_from_country",
        "pattern": re.compile(
            r"\bwhich\s+(?P<prof>[\w-]+?)s?\s+"
            r"(?:were|are|came\s+from|came|lived\s+in|are\s+from|"
            r"were\s+born\s+in|are\s+born\s+in)\s+"
            # (?-i:...) opts out of case-insensitivity so [A-Z]
            # actually requires uppercase first letter.
            r"(?P<country>(?-i:[A-Z])[\w-]+(?:\s+(?-i:[A-Z])[\w-]+)*)\??$",
            re.I,
        ),
    },
    # "Who was born in Germany?"
    {
        "name": "who_born_in",
        "pattern": re.compile(
            r"\bwho\s+(?:was|were)\s+born\s+in\s+"
            r"(?P<country>(?-i:[A-Z])[\w-]+(?:\s+(?-i:[A-Z])[\w-]+)*)\??$",
            re.I,
        ),
    },
    # "How many composers / scientists / countries / capitals?"
    {
        "name": "how_many",
        "pattern": re.compile(
            r"\bhow\s+many\s+(?P<thing>[\w-]+?)s?\s+"
            r"(?:do\s+you\s+(?:know|have)|are\s+there)\??$",
            re.I,
        ),
    },
]


class AggregateQA:
    """Set-returning queries over the structured claim store."""

    def __init__(self, structured: StructuredQA):
        self.structured = structured

    def _claims(self):
        return self.structured.claim_triple

    # ── Public entrypoint ──────────────────────────────────

    def answer(self, query: str) -> AggregateResult | None:
        for shape in _PATTERNS:
            m = shape["pattern"].search(query)
            if not m:
                continue
            name = shape["name"]
            if name == "list_profession":
                return self._list_profession(m.group("prof"))
            if name == "which_profession_from_country":
                return self._which_profession_from_country(
                    m.group("prof"), m.group("country"))
            if name == "who_born_in":
                return self._who_born_in(m.group("country"))
            if name == "how_many":
                return self._how_many(m.group("thing"))
        return None

    # ── Individual handlers ────────────────────────────────

    def _list_profession(self, prof_raw: str) -> AggregateResult | None:
        prof = prof_raw.lower().rstrip("s")    # composers -> composer
        if prof not in _PROFESSION_SET and prof + "s" not in _PROFESSION_SET:
            return None
        items = sorted({
            s for s, v, o in self._claims()
            if v == "is" and prof in o.lower()
        })
        return AggregateResult(
            pattern="list_profession",
            filter_desc=f"{prof}s I have claims about",
            items=items,
            count=len(items),
        )

    def _which_profession_from_country(self, prof_raw: str,
                                          country: str) -> AggregateResult:
        prof = prof_raw.lower().rstrip("s")
        # Demonym-aware: "German" should also match country "Germany".
        country_keys = {country.lower()}
        # Add the country->demonym reverse map from structured_qa.
        try:
            from lattice.structured_qa import _NATIONALITY_TO_COUNTRY
            for adj, ctry in _NATIONALITY_TO_COUNTRY.items():
                if ctry.lower() == country.lower():
                    country_keys.add(adj)
                if adj == country.lower():
                    country_keys.add(ctry.lower())
        except Exception:
            pass

        born_in: dict[str, str] = {}    # subj -> matched country form
        for s, v, o in self._claims():
            if v == "born" and o.lower() in country_keys:
                born_in[s] = o

        profession_of: dict[str, set] = {}
        for s, v, o in self._claims():
            if v == "is":
                profession_of.setdefault(s, set()).add(o.lower())

        # Match by SURNAME too, since Wikipedia bios use full names.
        def surname(s: str) -> str:
            return s.split()[-1].lower() if s else ""

        born_surnames = {surname(s): s for s in born_in}

        matched: list[str] = []
        for subj, profs in profession_of.items():
            if prof in profs or any(prof in p for p in profs):
                # Match the same subject directly, OR by surname.
                if subj in born_in:
                    matched.append(subj)
                elif surname(subj) in born_surnames:
                    matched.append(subj)
        matched.sort()
        return AggregateResult(
            pattern="which_profession_from_country",
            filter_desc=f"{prof}s born in {country}",
            items=matched,
            count=len(matched),
        )

    def _who_born_in(self, country: str) -> AggregateResult:
        country_keys = {country.lower()}
        try:
            from lattice.structured_qa import _NATIONALITY_TO_COUNTRY
            for adj, ctry in _NATIONALITY_TO_COUNTRY.items():
                if ctry.lower() == country.lower():
                    country_keys.add(adj)
        except Exception:
            pass
        items = sorted({
            s for s, v, o in self._claims()
            if v == "born" and o.lower() in country_keys
        })
        return AggregateResult(
            pattern="who_born_in",
            filter_desc=f"people born in {country}",
            items=items,
            count=len(items),
        )

    # Common irregular plurals + words ending in -ies / -es / -y where
    # the naive rstrip("s") would mangle.
    _PLURAL_STEM = {
        "countries": "country", "cities": "city", "facts": "fact",
        "claims": "claim", "nations": "nation", "capitals": "capital",
    }

    def _how_many(self, thing_raw: str) -> AggregateResult:
        raw = thing_raw.lower()
        # The regex strips a trailing 's', so we also try raw+'s' in
        # the plural-stem map ("countrie" -> "countries" -> "country").
        thing = (self._PLURAL_STEM.get(raw)
                   or self._PLURAL_STEM.get(raw + "s")
                   or raw.rstrip("s"))
        # Maps natural-language thing -> what to count
        if thing in _PROFESSION_SET or thing + "s" in _PROFESSION_SET:
            items = sorted({
                s for s, v, o in self._claims()
                if v == "is" and thing in o.lower()
            })
            return AggregateResult(
                pattern="how_many",
                filter_desc=f"{thing}s",
                items=items,
                count=len(items),
            )
        if thing in {"country", "countries", "nation", "nations"}:
            items = sorted({
                s for s, v, o in self._claims() if v == "capital"
            })
            return AggregateResult(
                pattern="how_many",
                filter_desc="countries (anything with a capital claim)",
                items=items,
                count=len(items),
            )
        if thing in {"capital", "capitals", "city", "cities"}:
            items = sorted({
                o for s, v, o in self._claims() if v == "capital"
            })
            return AggregateResult(
                pattern="how_many",
                filter_desc="capital cities",
                items=items,
                count=len(items),
            )
        if thing in {"fact", "facts", "claim", "claims"}:
            return AggregateResult(
                pattern="how_many",
                filter_desc="structured claims total",
                items=[],
                count=len(self._claims()),
            )
        # Generic fallback: count items with this word as object.
        items = sorted({
            s for s, v, o in self._claims() if thing in o.lower()
        })
        return AggregateResult(
            pattern="how_many",
            filter_desc=f"entries matching '{thing}'",
            items=items,
            count=len(items),
        )
