"""
lattice/pure_encoder.py - text encoding with NO LLM dependency.

The other encoder (text_encoder.py) wraps MiniLM — a small distilled
BERT. Every test using it benefits from the LLM's semantic geometry
(Germany already lives near Berlin in MiniLM space because they
co-occur in training corpora).

This module strips the LLM out. Only HDC primitives:

  - Each character gets a random hypervector (fixed seed for determinism)
  - Word vector = bundle of position-permuted character vectors
  - Sentence vector = bundle of position-permuted word vectors

There is NO pre-trained semantic geometry. Two unrelated words with the
same length and similar letters might look similar; semantically
related words with different letters will look unrelated.

This is the honest test of what HDC alone can do.

References:
  Kanerva (1988) - Sparse Distributed Memory (the original text-encoding
                   approach uses random vectors + position binding)
  Joshi et al. (2016) - "Language Geometry Using Random Indexing"
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, bundle, hamming_distance


_DEFAULT_SEED = 1729


class PureHDCTextEncoder:
    """Text -> hypervector via pure HDC primitives, no LLM.

    All character vectors are random and deterministic (seeded).
    Position binding uses cyclic permutation (np.roll).
    Composition uses majority-vote bundling.
    """

    def __init__(self, seed: int = _DEFAULT_SEED, case_sensitive: bool = False):
        self.rng = np.random.default_rng(seed)
        self.case_sensitive = case_sensitive
        # character -> hypervector (lazy-init for unseen chars, but
        # deterministic per character via per-char seeded RNG)
        self._char_vecs: dict[str, np.ndarray] = {}
        self._seed = seed

    def _char_hv(self, c: str) -> np.ndarray:
        if not self.case_sensitive:
            c = c.lower()
        if c not in self._char_vecs:
            # Per-character deterministic seed so character vectors are
            # stable across runs and across encoders with the same seed
            local_seed = (self._seed * 1009 + ord(c)) % (2**32)
            local_rng = np.random.default_rng(local_seed)
            self._char_vecs[c] = local_rng.integers(
                0, 2, size=D, dtype=np.int8
            )
        return self._char_vecs[c]

    def encode_word(self, word: str) -> np.ndarray:
        """One word -> position-bound bundle of its character vectors."""
        if not word:
            return np.zeros(D, dtype=np.int8)
        positioned = []
        for i, ch in enumerate(word):
            v = self._char_hv(ch)
            positioned.append(np.roll(v, i))
        return bundle(positioned)

    def encode(self, text: str) -> np.ndarray:
        """Sentence / phrase -> position-bound bundle of word vectors."""
        if not text:
            return np.zeros(D, dtype=np.int8)
        words = text.split()
        if not words:
            return np.zeros(D, dtype=np.int8)
        positioned_words = []
        for i, w in enumerate(words):
            wv = self.encode_word(w)
            # Position-bind the word at the sentence level too
            positioned_words.append(np.roll(wv, i * 7))   # 7 = arbitrary shift offset
        return bundle(positioned_words)

    def encode_many(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts])


# ─── Sanity check / smoke test ─────────────────────────────────────


def _sanity_check():
    """Verify the pure encoder has reasonable properties."""
    print("Pure HDC encoder sanity check\n")
    enc = PureHDCTextEncoder()

    # 1. Determinism
    print("1. Determinism")
    a1 = enc.encode("hello world")
    a2 = enc.encode("hello world")
    print(f"   ham(same input twice) = {hamming_distance(a1, a2)}  (should be 0)")
    assert hamming_distance(a1, a2) == 0

    # 2. Different texts -> different vectors (no collisions)
    print("\n2. Distinctness")
    h_hi = enc.encode("hi there")
    h_bye = enc.encode("goodbye friend")
    d = hamming_distance(h_hi, h_bye)
    print(f"   ham('hi there', 'goodbye friend') = {d} ({d/D*100:.1f}%) — should be ~50%")

    # 3. Same letters, different order -> different vectors (position matters)
    print("\n3. Position binding works")
    h_abc = enc.encode_word("abc")
    h_cba = enc.encode_word("cba")
    d = hamming_distance(h_abc, h_cba)
    print(f"   ham('abc', 'cba') = {d} ({d/D*100:.1f}%) — should be substantial")

    # 4. Shared characters -> some similarity
    print("\n4. Shared characters create some similarity")
    h_cat = enc.encode_word("cat")
    h_car = enc.encode_word("car")
    h_xyz = enc.encode_word("xyz")
    d_cat_car = hamming_distance(h_cat, h_car)
    d_cat_xyz = hamming_distance(h_cat, h_xyz)
    print(f"   ham('cat', 'car') = {d_cat_car} ({d_cat_car/D*100:.1f}%)")
    print(f"   ham('cat', 'xyz') = {d_cat_xyz} ({d_cat_xyz/D*100:.1f}%)")
    print(f"   'cat'/'car' should be closer than 'cat'/'xyz': "
          f"{'YES' if d_cat_car < d_cat_xyz else 'NO'}")

    # 5. The big one: semantically related, different letters
    print("\n5. The hard test — semantic relation, no letter overlap")
    h_germany = enc.encode("Germany")
    h_berlin  = enc.encode("Berlin")
    h_xyz_word = enc.encode("xqrtmpz")
    d_g_b = hamming_distance(h_germany, h_berlin)
    d_g_xyz = hamming_distance(h_germany, h_xyz_word)
    print(f"   ham('Germany', 'Berlin')  = {d_g_b} ({d_g_b/D*100:.1f}%)")
    print(f"   ham('Germany', 'xqrtmpz') = {d_g_xyz} ({d_g_xyz/D*100:.1f}%)")
    print(f"   No semantic prior — both should be near 50%")


if __name__ == "__main__":
    _sanity_check()
