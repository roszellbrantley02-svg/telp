"""
lattice/knowledge_graph.py - HRR-style knowledge base in pure HDC.

Tony Plate's 1994 idea, applied: store relational facts as bound triples,
bundle them all into a single 'knowledge' hypervector, query by unbinding.

A fact is a triple (subject, relation, object). Example:
    ("Germany", "HAS_CAPITAL", "Berlin")

In HDC:
    fact_hv = HAS_CAPITAL ⊗ Germany ⊗ Berlin
    knowledge = bundle of all fact_hvs

To query "what is Germany's capital?":
    query_hv = knowledge ⊗ HAS_CAPITAL ⊗ Germany
    answer = nearest entity in cleanup memory(query_hv)

The algebra works because XOR is self-inverting:
    knowledge ⊗ HAS_CAPITAL ⊗ Germany
    = (bundle of all facts) ⊗ HAS_CAPITAL ⊗ Germany
    ≈ HAS_CAPITAL ⊗ Germany ⊗ Berlin ⊗ HAS_CAPITAL ⊗ Germany  (+ noise from other facts)
    = Berlin (+ noise that the cleanup memory denoises)

The "+ noise" is the cross-talk from all other facts. Cleanup memory
fixes it as long as we don't store more facts than the bundle capacity
allows (typically ~7-15 for binary 10k-dim vectors before signal degrades).

This is the BUILDING BLOCK Plate envisioned for relational reasoning in
vector symbolic architectures. It does NOT learn — it stores. The
'generalization' question is separate.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import (
    D, bundle, bind, hamming_distance, random_vec
)


class HDCKnowledgeBase:
    """HRR-style knowledge graph in HDC.

    Each entity (subject or object) is a random hypervector.
    Each relation is also a random hypervector (functionally a "role").
    A fact (s, r, o) becomes the bound vector r ⊗ s ⊗ o.
    The knowledge is the bundle of all facts.

    Query (s, r) -> recover ~o by XOR-unbinding the knowledge.
    """

    def __init__(self, seed: int = 314159):
        self._rng = np.random.default_rng(seed)
        self.entities: dict[str, np.ndarray] = {}
        self.relations: dict[str, np.ndarray] = {}
        self.facts: list[tuple[str, str, str]] = []
        self._fact_hvs: list[np.ndarray] = []
        self.knowledge: np.ndarray | None = None

    # ─── Atomic vectors ──────────────────────────────────────

    def _entity_vec(self, name: str) -> np.ndarray:
        if name not in self.entities:
            self.entities[name] = self._rng.integers(0, 2, size=D, dtype=np.int8)
        return self.entities[name]

    def _relation_vec(self, name: str) -> np.ndarray:
        if name not in self.relations:
            self.relations[name] = self._rng.integers(0, 2, size=D, dtype=np.int8)
        return self.relations[name]

    def set_entity_vector(self, name: str, hv: np.ndarray) -> None:
        """Override a random entity vector with one from an external encoder.

        Useful if you want to use RI-derived semantic vectors for entities
        (so retrieval can also work via similarity, not only exact recall).
        """
        assert hv.shape == (D,) and hv.dtype == np.int8
        self.entities[name] = hv.copy()

    # ─── Fact storage ────────────────────────────────────────

    def add_fact(self, subject: str, relation: str, object_: str) -> None:
        """Bind (relation ⊗ subject ⊗ object) and bundle into knowledge."""
        s = self._entity_vec(subject)
        r = self._relation_vec(relation)
        o = self._entity_vec(object_)
        fact_hv = bind(bind(r, s), o)
        self.facts.append((subject, relation, object_))
        self._fact_hvs.append(fact_hv)
        # Re-bundle so knowledge reflects all stored facts
        self.knowledge = bundle(self._fact_hvs)

    # ─── Queries ─────────────────────────────────────────────

    def query(self, subject: str, relation: str,
                top_k: int = 1, restrict_to: list[str] | None = None
                ) -> list[tuple[str, int]]:
        """Find ~o such that (subject, relation, o) was stored.

        Returns top_k (name, hamming_distance) tuples sorted by closeness.
        If restrict_to is given, only entities in that list are candidates.
        """
        if self.knowledge is None:
            return []
        s = self._entity_vec(subject)
        r = self._relation_vec(relation)
        # query_hv = knowledge ⊗ r ⊗ s   ≈   o   (+ cross-talk noise)
        query_hv = bind(bind(self.knowledge, r), s)

        candidates = restrict_to if restrict_to is not None else list(self.entities.keys())
        scored = []
        for name in candidates:
            if name == subject:
                continue
            v = self._entity_vec(name)
            d = hamming_distance(query_hv, v)
            scored.append((name, d))
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]

    def query_with_vector(self, subject_vec: np.ndarray, relation: str,
                            cleanup_entities: list[str],
                            top_k: int = 1) -> list[tuple[str, int]]:
        """Query using an arbitrary subject hypervector (e.g., RI-derived).

        Useful when the subject's identity is captured as a semantic
        vector rather than a stored random vector. Tests whether
        relational retrieval generalizes to similar-but-unstored subjects.
        """
        if self.knowledge is None:
            return []
        r = self._relation_vec(relation)
        query_hv = bind(bind(self.knowledge, r), subject_vec)
        scored = []
        for name in cleanup_entities:
            v = self._entity_vec(name)
            d = hamming_distance(query_hv, v)
            scored.append((name, d))
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]

    # ─── Inspection ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.facts)

    def stats(self) -> dict:
        return {
            "n_facts": len(self.facts),
            "n_entities": len(self.entities),
            "n_relations": len(self.relations),
            "bundle_load": "n/a — binary HD has soft capacity ~7-15 items",
        }
