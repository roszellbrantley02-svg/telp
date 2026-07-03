"""
lattice/role_typing.py — Phase 17a: observed role-filler distributions.

WHY
---
A dictionary tells you what a word IS.  It does not tell you how
the word BEHAVES in stories.  "Headroom" is a kind of room, says
the dictionary.  But no story ever has a character walk to the
headroom — because headroom is the space ABOVE your head, not a
place you go.

This module learns "how words behave" by observing every frame
Telp has actually read.  For each (role, action) pair, it records
what fillers have appeared there:

  GOAL after walked: {forest: 8, hill: 4, pond: 3, river: 2,
                       house: 2, ...}

The output is NOT used as a filter — that would suppress creativity.
It is used as a NOVELTY DETECTOR: when Telp wants to use "headroom"
as the GOAL of a walked frame, novelty() returns ~1.0, and the
imagination engine knows to attach a JUSTIFICATION clause so the
unusual choice earns its place in the story.

DESIGN
------
Build once from lattice.line_events.  Query many times during
generation.  Counters are small (per-role × per-action histograms).
"""
from __future__ import annotations

from collections import Counter
from typing import Optional


# Roles where we care about novelty detection.  Agent and action
# are picked by the imagination engine's own logic; we don't second-
# guess those.  The interesting roles to ground in observed data are
# the slots the protagonist interacts WITH.
_TRACKED_ROLES = (
    "patient", "goal", "location", "attribute", "direction",
    "source", "instrument", "recipient",
)


class RoleFillerTable:
    """Observed distribution of fillers per (role, action) and per role.

    Built once at engine init from a `line_events` dict.  Provides
    novelty() for any candidate filler, scaled to [0.0, 1.0]:

      0.0  — extremely typical (this filler dominates the slot)
      0.4  — has been seen here at least once, modest frequency
      0.7  — seen in this role for OTHER actions, but not this one
      1.0  — never seen in this role at all (= fully novel)
    """

    def __init__(self, line_events: Optional[dict] = None):
        # role -> Counter of fillers
        self.role_filler: dict[str, Counter] = {}
        # (role, action) -> Counter of fillers
        self.role_action_filler: dict[tuple[str, str], Counter] = {}
        # action -> Counter of fillers (any role) — used as soft prior
        self.action_filler: dict[str, Counter] = {}
        if line_events:
            self.observe_lattice(line_events)

    # ── Observation ────────────────────────────────────────────

    def observe_lattice(self, line_events: dict) -> None:
        """Walk every parsed frame, recording its role->filler bindings."""
        for raw, frame in line_events.items():
            if not isinstance(frame, dict):
                continue
            action = (frame.get("action") or "").lower().strip()
            for role in _TRACKED_ROLES:
                filler = frame.get(role)
                if not filler:
                    continue
                f = str(filler).lower().strip()
                if not f:
                    continue
                self.role_filler.setdefault(role, Counter())[f] += 1
                if action:
                    self.role_action_filler.setdefault(
                        (role, action), Counter())[f] += 1
                    self.action_filler.setdefault(
                        action, Counter())[f] += 1

    def observe_frame(self, frame: dict) -> None:
        """Incrementally observe one parsed frame."""
        self.observe_lattice({frame.get("_raw", ""): frame})

    # ── Querying ───────────────────────────────────────────────

    def novelty(self, role: str, action: str, filler: str) -> float:
        """Return novelty score in [0.0, 1.0].

        0.0 = filler dominates this (role, action) slot.
        1.0 = filler has never appeared in this role.
        """
        f = filler.lower().strip()
        action = (action or "").lower().strip()

        # Strongest signal: how often does this filler appear in
        # THIS exact (role, action) slot?
        if action:
            ra_counter = self.role_action_filler.get((role, action))
            if ra_counter and f in ra_counter:
                count = ra_counter[f]
                total = sum(ra_counter.values())
                # If a filler dominates (e.g. forest is 8/30 GOALs of
                # walked), novelty ~= 1.0 - 5*0.27 ~= 0 (very typical).
                # If it's a one-off (1/30), novelty ~= 0.83.
                freq = count / max(total, 1)
                return max(0.0, min(1.0, 1.0 - freq * 5.0))

        # Soft signal: has the filler been seen in this role for ANY action?
        r_counter = self.role_filler.get(role)
        if r_counter and f in r_counter:
            return 0.7   # known role-filler, just not this verb

        # Never seen in this role at all.
        return 1.0

    def typical_fillers(self, role: str, action: Optional[str] = None,
                                top_n: int = 10) -> list[tuple[str, int]]:
        """Most common fillers for a (role, action) slot."""
        if action:
            c = self.role_action_filler.get((role, action.lower()))
            if c:
                return c.most_common(top_n)
        c = self.role_filler.get(role)
        return c.most_common(top_n) if c else []

    # ── Diagnostics ────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "roles":                 sorted(self.role_filler.keys()),
            "n_role_action_pairs":   len(self.role_action_filler),
            "n_role_fillers_total":  sum(len(c) for c in
                                                  self.role_filler.values()),
            "n_actions":             len(self.action_filler),
        }


# ─── CLI smoke ───────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lattice.reading_lattice import ReadingLattice, LATTICE_PATH

    if not Path(LATTICE_PATH).exists():
        print(f"No lattice at {LATTICE_PATH}", file=sys.stderr)
        sys.exit(1)
    lat = ReadingLattice.load(LATTICE_PATH)
    le = getattr(lat, "line_events", {}) or {}
    print(f"Lattice has {len(le)} stored frames")
    table = RoleFillerTable(le)
    print(f"Stats: {table.stats()}")
    print()
    for role, action in [("goal", "walked"), ("goal", "ran"),
                                  ("location", "saw"), ("patient", "saw"),
                                  ("attribute", "felt"),
                                  ("attribute", "was")]:
        top = table.typical_fillers(role, action, top_n=8)
        print(f"  TOP for {role}/{action}: {top}")
    print()
    print(f"Novelty checks:")
    for role, action, filler in [
        ("goal", "walked", "forest"),
        ("goal", "walked", "hill"),
        ("goal", "walked", "headroom"),
        ("goal", "walked", "library"),
        ("location", "saw",  "morning"),
        ("attribute", "felt", "lonely"),
        ("attribute", "felt", "philosophical"),
    ]:
        n = table.novelty(role, action, filler)
        print(f"  novelty({role}, {action}, {filler!r}) = {n:.2f}")
