"""
lattice/hdc_gnn.py - HDC graph neural network with message passing.

Each entity in the knowledge graph has a state hypervector. Each round
of message passing:

  For every (s, r, o) triple, two messages flow:
    s receives:  bind(role_r, o.state)              "I'm related to o via r"
    o receives:  bind(role_r_inv, s.state)          "s is related to me via r"

  After collecting all incoming messages, each entity updates:
    new_state = bundle(current_state, *messages)

After K rounds, each entity's state has absorbed information from
K hops away in the graph. This is the HDC analog of message-passing
GNNs (GCN, R-GCN, etc.).

For knowledge-graph completion: train state vectors via message
passing, then query (s, r) by binding s.state with r_role and finding
the nearest entity. Generalization comes from the propagation — even
if (s, r, o) was held out, s's state has absorbed information from
neighbors that DO have similar relations, pulling s toward o-like
entities in HD space.

This is the next frontier past classical knowledge-graph embedding:
RBI is one round of message passing with explicit role binding;
multi-round GNN-HDC propagates further.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, bind, hamming_distance


class HDCGraphNN:
    """Multi-round message-passing GNN in HDC.

    States are accumulated as integer counters (signed bipolar) and
    binarized for queries.
    """

    def __init__(self, dim: int = D, seed: int = 42,
                  decay: float = 0.5):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.decay = decay   # how much weight to keep from prior rounds
        self.index_vectors: dict[str, np.ndarray] = {}     # original random ID
        self.relation_roles: dict[str, np.ndarray] = {}    # role per relation
        self.relation_roles_inv: dict[str, np.ndarray] = {}
        # State as signed int accumulator (resets at start of each round)
        self.state_counters: dict[str, np.ndarray] = {}
        self.triples: list[tuple[str, str, str]] = []

    # ─── Atomic vectors ─────────────────────────────────────

    def _index(self, name: str) -> np.ndarray:
        if name not in self.index_vectors:
            self.index_vectors[name] = self.rng.integers(
                0, 2, size=self.dim, dtype=np.int8
            )
            # initial state = index vector (signed)
            self.state_counters[name] = (
                self.index_vectors[name].astype(np.int32) * 2 - 1
            ) * 8   # small starting magnitude
        return self.index_vectors[name]

    def _role(self, rel: str, inverse: bool = False) -> np.ndarray:
        key = rel + ("_INV" if inverse else "")
        store = self.relation_roles_inv if inverse else self.relation_roles
        if key not in store:
            store[key] = self.rng.integers(0, 2, size=self.dim, dtype=np.int8)
        return store[key]

    # ─── Build graph ────────────────────────────────────────

    def add_triples(self, triples: list[tuple[str, str, str]]) -> None:
        for s, r, o in triples:
            self._index(s); self._index(o)
            self._role(r); self._role(r, inverse=True)
            self.triples.append((s, r, o))

    # ─── Message passing ────────────────────────────────────

    def get_binary_state(self, name: str) -> np.ndarray:
        c = self.state_counters[name]
        return (c > 0).astype(np.int8)

    def step(self) -> None:
        """One round of message passing.

        Each entity collects messages from its neighbors, bound with
        the relation role, then bundles them into its state counter.
        """
        new_counters: dict[str, np.ndarray] = {}
        for name in self.state_counters:
            # Decay prior state
            new_counters[name] = (self.state_counters[name].astype(np.float32) * self.decay).astype(np.int32)

        for s, r, o in self.triples:
            r_role = self._role(r)
            r_inv  = self._role(r, inverse=True)
            o_state = self.get_binary_state(o)
            s_state = self.get_binary_state(s)
            # Message s gets: bind(r_role, o.state)
            msg_to_s = np.bitwise_xor(r_role, o_state)
            new_counters[s] += (msg_to_s.astype(np.int32) * 2 - 1)
            # Message o gets: bind(r_inv, s.state)
            msg_to_o = np.bitwise_xor(r_inv, s_state)
            new_counters[o] += (msg_to_o.astype(np.int32) * 2 - 1)

        self.state_counters = new_counters

    def run_rounds(self, k: int) -> None:
        for _ in range(k):
            self.step()

    # ─── Inference ──────────────────────────────────────────

    def query(self, subject: str, relation: str,
                candidates: list[str], top_k: int = 1
                ) -> list[tuple[str, int]]:
        """Predict (subject, relation, ?) via state-bound query."""
        if subject not in self.state_counters:
            return []
        s_state = self.get_binary_state(subject)
        r_role = self._role(relation)
        target = np.bitwise_xor(s_state, r_role)
        scored = []
        for c in candidates:
            if c not in self.state_counters:
                continue
            c_state = self.get_binary_state(c)
            d = hamming_distance(target, c_state)
            scored.append((c, d))
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]
