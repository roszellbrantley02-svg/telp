"""autopilot/forward_chain.py — neurosymbolic forward-chaining
inference over the lattice's stored facts.

Given a question that requires CHAINED facts (e.g., "how old was X
when Y happened?"), this module:

  1. Extracts the entities + properties asked about.
  2. Pulls each entity's facts from the lattice via targeted query.
  3. Runs a small set of inference rules (date arithmetic, transitive
     "X is a Y", "Y was at Z" → "X was at Z", etc.).
  4. Returns a derived answer + the chain of facts that produced it.

This is the experimental neurosymbolic capstone — we're using HDC
retrieval as the FACT STORE and a deterministic rule engine as the
REASONER on top.  For chains of length 1-2 it works well; beyond
that it stretches the lattice's coverage.

Supported inferences (v1):

  ── Age at event ──
  Q: "How old was Einstein when general relativity was published?"
  Step 1: pull "Einstein born" → 1879
  Step 2: pull "general relativity published" → 1915
  Step 3: derived: 1915 - 1879 = 36 years old.

  ── Time between events ──
  Q: "How many years between WW1 and WW2?"
  Steps: pull WW1 end (1918) + WW2 start (1939) → 21 years.

  ── Lifespan ──
  Q: "How long did X live?"
  Steps: pull X died + X born → subtract.
"""
from __future__ import annotations

import re
from typing import Optional


# ─── Inference patterns ───────────────────────────────────────────


# (regex, handler_name) — order matters; first match wins.
_INFERENCE_PATTERNS = [
    # "How old was X when Y happened?"
    (re.compile(
        r"how\s+old\s+was\s+(.+?)\s+when\s+(.+?)\s*\??$",
        re.IGNORECASE | re.DOTALL,
    ), "age_at_event"),

    # "How long did X live?"
    (re.compile(
        r"how\s+long\s+did\s+(.+?)\s+live\s*\??$",
        re.IGNORECASE,
    ), "lifespan"),

    # "How many years between X and Y?"
    (re.compile(
        r"how\s+many\s+years\s+(?:are\s+)?between\s+(.+?)\s+and\s+(.+?)\s*\??$",
        re.IGNORECASE | re.DOTALL,
    ), "years_between"),

    # "When did X die?"  → not a chain, but goes through fact lookup
    (re.compile(
        r"when\s+did\s+(.+?)\s+(?:die|pass\s+away)\s*\??$",
        re.IGNORECASE,
    ), "year_died"),

    # "When was X born?"
    (re.compile(
        r"when\s+was\s+(.+?)\s+born\s*\??$",
        re.IGNORECASE,
    ), "year_born"),
]


# ─── Fact lookup primitives ──────────────────────────────────────


def _lookup_year(agent, entity: str, kind: str) -> Optional[int]:
    """Find the year associated with (entity, kind) by querying the
    lattice and parsing the highest-confidence match.

    kind in {"born", "died", "published", "happened", "ended", "started"}.
    """
    queries = {
        "born":      f"{entity} born",
        "died":      f"{entity} died",
        "published": f"{entity} published year",
        "happened":  f"{entity} year",
        "ended":     f"{entity} ended year",
        "started":   f"{entity} started year",
    }
    q = queries.get(kind, f"{entity} year")

    try:
        hits = agent.lattice.query(q, k=20)
    except Exception:
        return None

    # Score-weighted year voting: parse a year from each high-sim hit,
    # tally votes weighted by similarity.
    entity_words = set(re.findall(r"\b[a-z]{3,}\b", entity.lower()))
    year_score: dict[int, float] = {}

    for h in hits:
        sim = float(h.get("similarity") or 0.0)
        if sim < 0.50:
            continue
        txt = h.get("text") or ""
        toks = set(re.findall(r"\b[a-z]{3,}\b", txt.lower()))
        if not (entity_words & toks):
            continue

        # Find years that appear NEAR the kind keyword.
        kind_patterns = {
            "born":      r"born\s+(?:on\s+\w+\s+\d+,?\s*)?(?:in\s+)?(\d{4})",
            "died":      r"(?:died|passed\s+away)\s+(?:on\s+\w+\s+\d+,?\s*)?(?:in\s+)?(\d{4})",
            "published": r"published\s+(?:in\s+)?(\d{4})",
            "happened":  r"\b(?:in\s+)?(\d{4})\b",
            "ended":     r"ended\s+(?:in\s+)?(\d{4})",
            "started":   r"(?:began|started)\s+(?:in\s+)?(\d{4})",
        }
        pat = kind_patterns.get(kind, r"\b(\d{4})\b")
        for m in re.finditer(pat, txt, re.IGNORECASE):
            year = int(m.group(1))
            if 1000 <= year <= 2100:
                year_score[year] = year_score.get(year, 0.0) + sim

    if not year_score:
        return None
    return max(year_score, key=year_score.get)


def _normalize_entity(entity: str) -> str:
    """Strip leading articles and trim."""
    e = entity.strip()
    e = re.sub(r"^(?:the\s+|a\s+|an\s+)", "", e, flags=re.IGNORECASE)
    return e.strip()


# ─── Inference handlers ──────────────────────────────────────────


def _handler_age_at_event(agent, person: str, event: str) -> Optional[dict]:
    p = _normalize_entity(person)
    e = _normalize_entity(event)
    born = _lookup_year(agent, p, "born")
    # For the event, try published / happened.
    event_year = (_lookup_year(agent, e, "published")
                    or _lookup_year(agent, e, "happened"))
    if born is None or event_year is None:
        return None
    age = event_year - born
    return {
        "answer": (f"{p.title()} was {age} years old when "
                       f"{e} was published in {event_year}."),
        "chain": [
            f"{p} born → {born}",
            f"{e} → {event_year}",
            f"age = {event_year} - {born} = {age}",
        ],
    }


def _handler_lifespan(agent, person: str, _unused=None) -> Optional[dict]:
    p = _normalize_entity(person)
    born = _lookup_year(agent, p, "born")
    died = _lookup_year(agent, p, "died")
    if born is None or died is None:
        return None
    return {
        "answer": (f"{p.title()} lived {died - born} years "
                       f"({born}-{died})."),
        "chain": [
            f"{p} born → {born}",
            f"{p} died → {died}",
            f"lifespan = {died} - {born} = {died - born}",
        ],
    }


def _handler_years_between(agent, a: str, b: str) -> Optional[dict]:
    ya = (_lookup_year(agent, a, "happened")
            or _lookup_year(agent, a, "ended")
            or _lookup_year(agent, a, "started"))
    yb = (_lookup_year(agent, b, "happened")
            or _lookup_year(agent, b, "started")
            or _lookup_year(agent, b, "ended"))
    if ya is None or yb is None:
        return None
    diff = abs(yb - ya)
    return {
        "answer": (f"{diff} years between {a.strip()} ({ya}) and "
                       f"{b.strip()} ({yb})."),
        "chain": [
            f"{a.strip()} → {ya}",
            f"{b.strip()} → {yb}",
            f"diff = |{yb} - {ya}| = {diff}",
        ],
    }


def _handler_year_born(agent, person: str, _unused=None) -> Optional[dict]:
    p = _normalize_entity(person)
    y = _lookup_year(agent, p, "born")
    if y is None:
        return None
    return {"answer": f"{p.title()} was born in {y}.",
              "chain": [f"{p} born → {y}"]}


def _handler_year_died(agent, person: str, _unused=None) -> Optional[dict]:
    p = _normalize_entity(person)
    y = _lookup_year(agent, p, "died")
    if y is None:
        return None
    return {"answer": f"{p.title()} died in {y}.",
              "chain": [f"{p} died → {y}"]}


_HANDLERS = {
    "age_at_event":  _handler_age_at_event,
    "lifespan":      _handler_lifespan,
    "years_between": _handler_years_between,
    "year_born":     _handler_year_born,
    "year_died":     _handler_year_died,
}


# ─── Public entry ────────────────────────────────────────────────


def try_forward_chain(msg: str, agent) -> Optional[dict]:
    """Try to answer `msg` via forward-chaining inference over the
    lattice.

    Returns {answer: str, chain: list[str]} on success, None otherwise.
    """
    if not msg or agent is None:
        return None
    for pattern, handler_name in _INFERENCE_PATTERNS:
        m = pattern.search(msg)
        if not m:
            continue
        handler = _HANDLERS.get(handler_name)
        if handler is None:
            continue
        groups = m.groups()
        try:
            if len(groups) == 1:
                result = handler(agent, groups[0])
            else:
                result = handler(agent, *groups)
        except Exception:
            result = None
        if result is not None:
            return {
                "handler": handler_name,
                **result,
            }
    return None


def _self_test():
    """No-op without a loaded agent — just check that pattern matching
    works."""
    cases = [
        ("How old was Einstein when general relativity was published?",
            "age_at_event"),
        ("How long did Leonardo da Vinci live?",
            "lifespan"),
        ("How many years between World War 1 and World War 2?",
            "years_between"),
        ("When was Einstein born?",
            "year_born"),
        ("When did Einstein die?",
            "year_died"),
    ]
    for q, expected in cases:
        for pattern, handler_name in _INFERENCE_PATTERNS:
            if pattern.search(q):
                mark = "OK " if handler_name == expected else "BAD"
                print(f"  [{mark}] {q!r:<70} → {handler_name}")
                break
        else:
            print(f"  [BAD] {q!r:<70} → NO MATCH")


if __name__ == "__main__":
    _self_test()
