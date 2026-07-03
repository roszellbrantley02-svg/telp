"""
lattice/hierarchical.py - HDC encoding of multi-level taxonomies.

Each entity in a taxonomy has a chain of ancestors:
    labrador -> dog -> mammal -> animal -> physical -> thing

We encode it as a bundle of role-bound ancestor identifiers:
    labrador_hv = bundle(
        bind(LEVEL_0, thing_id),
        bind(LEVEL_1, physical_id),
        bind(LEVEL_2, animal_id),
        bind(LEVEL_3, mammal_id),
        bind(LEVEL_4, dog_id),
        bind(LEVEL_5, labrador_id),
    )

Each level has its own role vector. Each named concept (thing,
physical, animal, ...) has its own random id vector.

Querying IS-A:
    "Is labrador a mammal?"
    target = labrador_hv XOR LEVEL_3_role
    cleanup against all LEVEL_3 concepts -> mammal (if correct)

Querying class membership:
    "Find all mammals."
    For each stored entity:
        target = entity_hv XOR LEVEL_3_role
        if cleanup -> mammal, then entity is a mammal.

Bundle capacity matters here: with 6 levels bundled, that's near the
classical 7-item ceiling. Real test of whether HDC can hold deep
taxonomic structure.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, bundle, bind, hamming_distance


class HDCTaxonomy:
    """Encode and query a multi-level concept hierarchy via HDC."""

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self.concept_ids: dict[str, np.ndarray] = {}      # concept name -> id vec
        self.level_roles: dict[int, np.ndarray] = {}      # level int -> role vec
        # entity_hv stored by leaf name
        self.entities: dict[str, np.ndarray] = {}
        # ancestor chain bookkeeping for inspection
        self.ancestors: dict[str, list[str]] = {}
        # which concepts are at which level (for cleanup queries)
        self.concepts_at_level: dict[int, list[str]] = {}

    def _concept_id(self, name: str) -> np.ndarray:
        if name not in self.concept_ids:
            self.concept_ids[name] = self._rng.integers(
                0, 2, size=D, dtype=np.int8
            )
        return self.concept_ids[name]

    def _level_role(self, level: int) -> np.ndarray:
        if level not in self.level_roles:
            self.level_roles[level] = self._rng.integers(
                0, 2, size=D, dtype=np.int8
            )
        return self.level_roles[level]

    def add_entity(self, ancestor_chain: list[str]) -> None:
        """ancestor_chain[0]=root concept, ...[-1]=leaf (this entity).

        Each ancestor at level i contributes bind(LEVEL_i, ancestor_id)
        to the entity's bundled hypervector.
        """
        parts = []
        for level, concept in enumerate(ancestor_chain):
            parts.append(bind(self._level_role(level), self._concept_id(concept)))
            # Track which concepts live at which level
            self.concepts_at_level.setdefault(level, [])
            if concept not in self.concepts_at_level[level]:
                self.concepts_at_level[level].append(concept)
        leaf = ancestor_chain[-1]
        self.entities[leaf] = bundle(parts)
        self.ancestors[leaf] = list(ancestor_chain)

    # ─── Queries ────────────────────────────────────────────

    def is_a(self, entity: str, level: int) -> tuple[str, int]:
        """Return the cleanup match for what the entity is at given level.

        Example: is_a('labrador', level=3) -> ('mammal', distance)
        """
        if entity not in self.entities:
            raise KeyError(entity)
        ent_hv = self.entities[entity]
        role = self._level_role(level)
        target = np.bitwise_xor(ent_hv, role)
        # Cleanup against all known concepts at that level
        candidates = self.concepts_at_level.get(level, [])
        if not candidates:
            return None, None
        best = None
        best_d = None
        for c in candidates:
            d = hamming_distance(target, self._concept_id(c))
            if best_d is None or d < best_d:
                best_d = d
                best = c
        return best, best_d

    def members_of(self, concept: str, level: int) -> list[str]:
        """Find all stored entities whose level-i ancestor cleans up to `concept`."""
        members = []
        for ent_name in self.entities:
            ancestor, _ = self.is_a(ent_name, level)
            if ancestor == concept:
                members.append(ent_name)
        return members

    def similarity(self, a: str, b: str) -> int:
        """Hamming distance between two entities' bundled hypervectors."""
        return hamming_distance(self.entities[a], self.entities[b])
