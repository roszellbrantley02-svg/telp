"""
lattice/relation_indexing.py - HDC encoder with explicit relation roles.

The next-generation Random Indexing: instead of bag-of-neighbors
accumulation, accumulate role-bound relational evidence.

For each (subject, relation, object) triple in the corpus:
  Add  bind(RELATION_role, OBJECT_index_vector)  to SUBJECT's context.

After training, each word's context vector is a bundle of all its
participation in relations, each tagged with the appropriate role.

To query "subject's R-relation target":
  target = XOR(subject_context, R_role)   # unbind
  cleanup against candidate index vectors to recover the specific O

This is the HDC equivalent of structured knowledge representation
PLUS distributional learning. Unlike pure HRR, multiple objects for
the same subject+relation bundle gracefully (capacity ~ bundle size).
Unlike pure RI, pair-specific signal is preserved because the binding
is role-aware.

This is THE attack on the relational-pairing problem we identified
in the failed RI test. If this works, HDC can do what we couldn't
crack before — extract specific relational pairings from templated
text and generalize to held-out subjects whose facts appear in the
corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, hamming_distance


class RelationBoundEncoder:
    """HDC encoder that accumulates relation-bound evidence per word.

    Vectors:
      - index_vectors[word]:     random binary vector — the word's identity
      - role_vectors[relation]:  random binary vector — the relation's role
      - context_counters[word]:  int32 accumulator (signed contribution from each bound triple)

    Training (.add_triple): for each (S, R, O):
        bound = XOR(R_role, O_index)
        signed_bound = (bound * 2) - 1                  # to ±1
        context_counters[S] += signed_bound

    Binarized context: (counters > 0).astype(int8)

    Query (.query): for each candidate O,
        target = XOR(S_context_binarized, R_role)
        cleanup: nearest candidate O_index by Hamming distance.
    """

    def __init__(self, dim: int = D, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.index_vectors: dict[str, np.ndarray] = {}
        self.role_vectors: dict[str, np.ndarray] = {}
        self.context_counters: dict[str, np.ndarray] = {}
        self.triple_count = 0

    # ─── Atomic vectors (lazy-allocated) ──────────────────────

    def _index(self, word: str) -> np.ndarray:
        if word not in self.index_vectors:
            self.index_vectors[word] = self.rng.integers(
                0, 2, size=self.dim, dtype=np.int8
            )
            self.context_counters[word] = np.zeros(self.dim, dtype=np.int32)
        return self.index_vectors[word]

    def _role(self, relation: str) -> np.ndarray:
        if relation not in self.role_vectors:
            self.role_vectors[relation] = self.rng.integers(
                0, 2, size=self.dim, dtype=np.int8
            )
        return self.role_vectors[relation]

    # ─── Training ────────────────────────────────────────────

    def add_triple(self, subject: str, relation: str, object_: str) -> None:
        """Add (subject, relation, object) — only subject's context updates."""
        # ensure atomic vectors exist
        self._index(subject)
        r = self._role(relation)
        o = self._index(object_)
        # bind(R, O) — XOR
        bound = np.bitwise_xor(r, o)
        # Convert {0,1} to {-1,+1} for proper signed accumulation
        signed = (bound.astype(np.int32) * 2) - 1
        self.context_counters[subject] += signed
        self.triple_count += 1

    def train(self, tagged_corpus) -> None:
        """tagged_corpus: list of (sentence_text, list_of_(subject, relation, object))."""
        for _sentence, triples in tagged_corpus:
            for subj, rel, obj in triples:
                self.add_triple(subj, rel, obj)

    # ─── Inspection / encoding ──────────────────────────────

    def get_context(self, word: str) -> np.ndarray:
        """Binarized context vector for a word."""
        if word not in self.context_counters:
            return np.zeros(self.dim, dtype=np.int8)
        return (self.context_counters[word] > 0).astype(np.int8)

    def get_index(self, word: str) -> np.ndarray:
        """The word's identity index vector (for cleanup memory use)."""
        return self._index(word)

    # ─── Queries ────────────────────────────────────────────

    def query(self, subject: str, relation: str,
                candidates: list[str], top_k: int = 1
                ) -> list[tuple[str, int]]:
        """Find subject's relation-target via XOR-unbind + cleanup.

        candidates: the cleanup space (which words to consider as possible
                    objects). Must include the true answer for it to be
                    findable.
        """
        if subject not in self.context_counters:
            return []
        ctx = self.get_context(subject)
        r = self._role(relation)
        target = np.bitwise_xor(ctx, r)
        # Cleanup against candidate index vectors
        scored = []
        for c in candidates:
            c_idx = self._index(c)
            d = hamming_distance(target, c_idx)
            scored.append((c, d))
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]

    def stats(self) -> dict:
        return {
            "vocab":     len(self.index_vectors),
            "relations": len(self.role_vectors),
            "triples":   self.triple_count,
        }
