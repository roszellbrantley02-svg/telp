"""
autopilot/seed_identity.py - bootstrap Telp's self-knowledge.

Writes a small set of "who am I" facts into the lattice + structured
claim store the FIRST time Telp starts.  Marks completion with
`state/.identity_seeded` so it doesn't double-seed on every start.

The facts are written with source="user_taught" so they survive across
sessions (see standalone_agent._CORPUS_PREFIXES).

Usage:
    # Direct:
    python -m mind.seed_identity

    # Programmatic (e.g. from chat.py on startup):
    from mind.seed_identity import seed_if_needed
    seed_if_needed(telp.agent)
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.paths import state_path  # noqa: E402
_MARKER = state_path(".identity_seeded")


# ─── The seed facts ────────────────────────────────────────────────
#
# Each line is a single self-fact.  Phrased so the structured-claim
# extractor will pick them up as S-V-O triples.  Short and definitive.


IDENTITY_FACTS: list[str] = [
    "Telp is an artificial intelligence agent.",
    "Telp's name is Telp.",
    "Telp was built by his user.",
    "Telp finds answers by comparing the meaning of his memories.",
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
]


# Facts earlier versions seeded - one machine-specific, one overclaiming.
RETIRED_IDENTITY_FACTS = [
    "Telp's source code is on E drive in the telp folder.",
    "Telp's brain uses hyperdimensional computing.",
]


def _migrate_old_seed(agent) -> int:
    """Bring a memory seeded by an earlier version up to date: drop retired
    facts, add new ones, and relabel identity facts that were filed as
    "user_taught" (provenance then told the user "you told me")."""
    con = agent.lattice._con
    relabeled = con.execute("UPDATE memories SET source='identity' "
                            "WHERE tags='identity' AND source='user_taught'"
                            ).rowcount
    con.commit()
    if relabeled:
        agent.lattice._reload_from_disk()
    retired = [mid for mid, t in zip(agent.lattice._ids, agent.lattice._texts)
               if t in RETIRED_IDENTITY_FACTS]
    agent.lattice.delete_ids(retired)        # also reloads from disk
    have = set(agent.lattice._texts)
    n = 0
    for fact in IDENTITY_FACTS:
        if fact not in have:
            agent.lattice.add(fact, source="identity", tags="identity",
                              turn=len(agent.turns))
            n += 1
    if relabeled or retired or n:
        agent._rebuild_structured_qa()
    return n


# ─── Seed function ─────────────────────────────────────────────────


def seed(agent, force: bool = False) -> dict:
    """Write identity facts to the given StandaloneAgent (or wrapper
    whose .agent is one).  Returns counts."""
    # Allow either a raw agent or a wrapper (FluentTelp)
    if hasattr(agent, "agent") and hasattr(agent.agent, "lattice"):
        agent = agent.agent

    # seeded = this memory already holds the identity facts (a global
    # marker file alone said "seeded" even for a brand-new memory file)
    already = agent.lattice._con.execute(
        "SELECT 1 FROM memories WHERE tags='identity' LIMIT 1").fetchone()
    if not force and already:
        n_new = _migrate_old_seed(agent)
        return {"seeded": False, "reason": "already seeded",
                  "updated": n_new, "marker": str(_MARKER)}

    n_lattice = 0
    n_claims = 0
    for fact in IDENTITY_FACTS:
        # source "identity", not "user_taught": these are built in, and
        # provenance must not tell the user "you told me" about them
        agent.lattice.add(fact, source="identity",
                              tags="identity",
                              turn=len(agent.turns))
        agent.encoder.add_sentence(fact)
        n_claims += agent.structured.add_sentence(fact, source="identity")
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
