"""
lattice/cleanup_memory.py - snap a noisy HV to its nearest known atom.

WHY
---
Every HDC operation (bind, unbind, bundle) introduces NOISE.  After
a chain of operations, the result HV doesn't exactly match any
stored atom — it's a perturbed version of one.

Cleanup is the inverse: given a noisy HV, find the nearest known
atom in a vocabulary and return THAT atom's exact HV.  Without
cleanup, generation drifts further from valid atoms with every
step and the output collapses into noise.

This module provides:
  CleanupMemory(vocab) — holds (label, hv) pairs, snaps noisy HV
                          to nearest by Hamming distance

Layered usage:
  phoneme_cleanup = CleanupMemory.from_phonemes()
  syllable_cleanup = CleanupMemory()    # grows as syllables observed
  word_cleanup     = CleanupMemory()    # grows as words observed
  phrase_cleanup   = CleanupMemory()    # grows as phrases observed
  line_cleanup     = CleanupMemory()    # grows as lines observed

After Telp predicts a noisy "next-word HV", word_cleanup.snap(hv)
returns the nearest real word he's actually heard.  Without that
snap, the prediction is unusable text.
"""
from __future__ import annotations
from typing import Iterable, Optional

import numpy as np

from train.v5_hdc_prototype import D, hamming_distance


class CleanupMemory:
    """A growable atom-store: (label -> HV) with nearest-neighbor lookup."""

    def __init__(self):
        self._labels: list[str] = []
        # Stored as (n, D) int8 matrix for vectorized Hamming
        self._hvs: Optional[np.ndarray] = None
        # Set of labels already added (for dedup on .add)
        self._label_set: set[str] = set()

    # ── Add atoms ───────────────────────────────────────────────

    def add(self, label: str, hv: np.ndarray) -> None:
        """Add a new (label, hv) atom.  Idempotent on label."""
        if label in self._label_set:
            return
        self._label_set.add(label)
        self._labels.append(label)
        h = hv.astype(np.int8).reshape(1, -1)
        if self._hvs is None:
            self._hvs = h.copy()
        else:
            self._hvs = np.vstack([self._hvs, h])

    def add_many(self, pairs: Iterable[tuple[str, np.ndarray]]) -> int:
        n = 0
        for label, hv in pairs:
            if label not in self._label_set:
                self.add(label, hv)
                n += 1
        return n

    # ── Snap noisy HV to nearest atom ──────────────────────────

    def snap(self, query_hv: np.ndarray,
                  top_k: int = 1) -> list[tuple[str, float]]:
        """Return top-K nearest atoms as [(label, similarity), ...].

        Similarity is 1.0 for identical, ~0.5 for orthogonal.
        Empty if no atoms stored.
        """
        if self._hvs is None or len(self._labels) == 0:
            return []
        # Vectorized Hamming via XOR + popcount
        q = query_hv.astype(np.int8).reshape(1, -1)
        xor = np.bitwise_xor(self._hvs, q)
        dists = xor.sum(axis=1)
        sims = 1.0 - (dists / D)
        # Top-K by similarity (highest first)
        k = min(top_k, len(self._labels))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self._labels[i], float(sims[i])) for i in idx]

    def snap_one(self, query_hv: np.ndarray) -> Optional[tuple[str, float]]:
        """Return the single nearest (label, similarity), or None if empty."""
        top = self.snap(query_hv, top_k=1)
        return top[0] if top else None

    # ── Bookkeeping ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._labels)

    def labels(self) -> list[str]:
        return list(self._labels)

    def has(self, label: str) -> bool:
        return label in self._label_set

    def get_hv(self, label: str) -> Optional[np.ndarray]:
        """Retrieve the exact stored HV for a label."""
        if label not in self._label_set:
            return None
        i = self._labels.index(label)
        return self._hvs[i].copy()

    def stats(self) -> dict:
        return {
            "n_atoms": len(self._labels),
            "hv_dim":   D,
            "memory_bytes": (self._hvs.nbytes
                                       if self._hvs is not None else 0),
        }

    # ── Factories ──────────────────────────────────────────────

    @classmethod
    def from_phonemes(cls) -> "CleanupMemory":
        """Pre-populate with the 39 ARPABET phoneme atoms."""
        from lattice.phoneme_hdc import PHONEMES, phoneme_hv
        cm = cls()
        for p in PHONEMES.keys():
            cm.add(p, phoneme_hv(p))
        return cm


# --- CLI smoke test -------------------------------------------------


if __name__ == "__main__":
    from lattice.phoneme_hdc import phoneme_hv, compose_word
    from train.v5_hdc_prototype import bundle

    print("=== Cleanup memory smoke test ===\n")

    # 1. Phoneme cleanup: known phonemes snap to themselves
    cm = CleanupMemory.from_phonemes()
    print(f"Phoneme cleanup: {len(cm)} atoms stored\n")

    for p in ("B", "IY", "S", "NG"):
        hv = phoneme_hv(p)
        top = cm.snap(hv, top_k=3)
        print(f"  exact {p:3} -> {top}")

    # 2. Noisy HV (flip 10% of bits) still snaps back
    print()
    np.random.seed(42)
    for p in ("B", "IY", "S"):
        hv = phoneme_hv(p).copy()
        # Flip 10% of bits
        flip = np.random.choice(D, size=D // 10, replace=False)
        hv[flip] = 1 - hv[flip]
        top = cm.snap(hv, top_k=3)
        ok = top[0][0] == p
        print(f"  noisy {p:3} -> {top[0][0]} {'(correct)' if ok else '(WRONG)'}  "
                f"top3={top}")

    # 3. A bundle of two phonemes should snap to either one (ambiguous)
    print("\n  bundle(B, P) — should be similar to both")
    mixed = bundle([phoneme_hv("B"), phoneme_hv("P")])
    top = cm.snap(mixed, top_k=4)
    for name, sim in top:
        print(f"    {name}: {sim:.3f}")

    # 4. Word cleanup growing from observations
    print("\n=== Word cleanup growing from observations ===\n")
    word_cm = CleanupMemory()
    # Add a few words
    for word, phonemes in [
        ("bear",  ["B", "EH", "R"]),
        ("bare",  ["B", "EH", "R"]),    # same phonemes as bear
        ("wear",  ["W", "EH", "R"]),
        ("dog",   ["D", "AO", "G"]),
        ("cat",   ["K", "AE", "T"]),
        ("bat",   ["B", "AE", "T"]),
        ("hat",   ["HH", "AE", "T"]),
    ]:
        word_cm.add(word, compose_word(phonemes))
    print(f"word cleanup: {len(word_cm)} words stored\n")

    # Query with a noisy "bear" HV
    bear_hv = compose_word(["B", "EH", "R"]).copy()
    flip = np.random.choice(D, size=D // 8, replace=False)
    bear_hv[flip] = 1 - bear_hv[flip]
    top = word_cm.snap(bear_hv, top_k=3)
    print(f"  noisy bear -> top3: {top}")

    # Query with a novel "fear" → should be closest to bear/wear/bare
    fear_hv = compose_word(["F", "IH", "R"])
    top = word_cm.snap(fear_hv, top_k=3)
    print(f"  novel fear -> top3: {top}  (no F in vocab — sees nearest neighbors)")

    # Query with "rat" → should be closest to bat/cat/hat (same suffix)
    rat_hv = compose_word(["R", "AE", "T"])
    top = word_cm.snap(rat_hv, top_k=3)
    print(f"  novel rat  -> top3: {top}  (shares AE+T with bat/cat/hat)")
