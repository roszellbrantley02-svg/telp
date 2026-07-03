"""
autopilot/seed_identity.py - bootstrap Telp's self-knowledge.

Writes a small set of "who am I" facts into the lattice + structured
claim store the FIRST time Telp starts.  Marks completion with
`state/.identity_seeded` so it doesn't double-seed on every start.

The facts are written with source="user_taught" so they survive across
sessions (see standalone_agent._CORPUS_PREFIXES).

Usage:
    # Direct:
    python -m autopilot.seed_identity

    # Programmatic (e.g. from chat.py on startup):
    from mind.seed_identity import seed_if_needed
    seed_if_needed(telp.agent)
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

_MARKER = _TELP_ROOT / "state" / ".identity_seeded"


# ─── The seed facts ────────────────────────────────────────────────
#
# Each line is a single self-fact.  Phrased so the structured-claim
# extractor will pick them up as S-V-O triples.  Short and definitive.


IDENTITY_FACTS: list[str] = [
    "Telp is an artificial intelligence agent.",
    "Telp's name is Telp.",
    "Telp was built by his user.",
    "Telp's brain uses hyperdimensional computing.",
    "Telp does not use a large language model.",
    "Telp can see images and remember what he has seen.",
    "Telp perceives, remembers, reasons, and speaks as one mind.",
    "Telp's memory is called the lattice.",
    "Telp's lattice stores knowledge as 10000-bit hypervectors.",
    "Telp learns by similarity rather than by gradient descent.",
    "Telp can say I don't know when he doesn't have an answer.",
    "Telp's knowledge includes Wikipedia, conversations, and what he has seen.",
    "Telp remembers what his user teaches him across sessions.",
    "Telp lives on his user's computer and does not call any cloud API.",
    "Telp's source code is on E drive in the telp folder.",
]


# ─── Seed function ─────────────────────────────────────────────────


def seed(agent, force: bool = False) -> dict:
    """Write identity facts to the given StandaloneAgent (or wrapper
    whose .agent is one).  Returns counts."""
    # Allow either a raw agent or a wrapper (FluentTelp)
    if hasattr(agent, "agent") and hasattr(agent.agent, "lattice"):
        agent = agent.agent

    if not force and _MARKER.exists():
        return {"seeded": False, "reason": "already seeded",
                  "marker": str(_MARKER)}

    n_lattice = 0
    n_claims = 0
    for fact in IDENTITY_FACTS:
        agent.lattice.add(fact, source="user_taught",
                              tags="identity",
                              turn=len(agent.turns))
        agent.encoder.add_sentence(fact)
        n_claims += agent.structured.add_sentence(fact, source="user_taught")
        n_lattice += 1

    agent.structured._dirty = True
    _MARKER.parent.mkdir(parents=True, exist_ok=True)
    _MARKER.write_text("seeded\n", encoding="utf-8")
    return {"seeded": True, "lattice_added": n_lattice,
              "claims_added": n_claims, "marker": str(_MARKER)}


def seed_if_needed(agent) -> dict:
    """Convenience wrapper: only seed if the marker file is absent."""
    return seed(agent, force=False)


# ─── CLI ───────────────────────────────────────────────────────────


def _main():
    import argparse, io
    if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                          errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                      help="seed even if the marker file already exists")
    args = ap.parse_args()

    from mind.fluency import FluentTelp
    telp = FluentTelp()
    result = seed(telp.agent, force=args.force)
    print(result)


if __name__ == "__main__":
    _main()
