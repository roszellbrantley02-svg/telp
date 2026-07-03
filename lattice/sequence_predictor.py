"""
lattice/sequence_predictor.py - tiny HDC LLM (next-word prediction).

Pure HDC sequence model. For each n-gram in the training corpus:
  - Encode the prefix (last N words) as a position-bound hypervector
  - Store (prefix_hv, next_word) as a memory

At inference, given a new prefix:
  - Encode it
  - Find the K nearest stored prefixes by Hamming distance
  - Vote (distance-weighted) on the most likely next word

This is essentially an N-gram language model with HDC encoding instead
of exact-match frequency tables. Because HDC uses similarity, paraphrased
or partial prefixes still find their neighbors — a generalization
property that pure frequency-counting n-grams lack.

To generate: repeatedly predict the next word, append, predict again.

No gradient descent. No neural network. Just bundle + Hamming + vote.

References:
  Kanerva (1988) - Sparse Distributed Memory uses related sequence
                   encoding
  Recchia et al. (2015) - HDC for sentence-level modeling
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, bundle, hamming_distance


_TOKEN_RE = re.compile(r"[^a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.split(text.lower()) if t]


class HDCSequencePredictor:
    """N-gram LM in pure HDC.

    Prefix encoding: bundle of position-permuted word hypervectors.
    Each word at position i gets rolled by (i * shift_step) bit positions
    before being bundled. Different positions create different
    hypervectors, so "the cat sat" and "sat cat the" hash differently.
    """

    def __init__(self, n_gram: int = 3, dim: int = D, seed: int = 42,
                  shift_step: int = 137):
        self.n = n_gram
        self.dim = dim
        self.shift_step = shift_step
        self.rng = np.random.default_rng(seed)
        self.word_vecs: dict[str, np.ndarray] = {}
        self.memories: list[tuple[np.ndarray, str, tuple[str, ...]]] = []
        # Built after train()
        self._stack: np.ndarray | None = None
        self._next_words: list[str] | None = None

    # ─── Atomic vectors ──────────────────────────────────────

    def _word_vec(self, word: str) -> np.ndarray:
        if word not in self.word_vecs:
            self.word_vecs[word] = self.rng.integers(
                0, 2, size=self.dim, dtype=np.int8
            )
        return self.word_vecs[word]

    def _encode_prefix(self, words: list[str]) -> np.ndarray:
        """Position-bound bundle of the last N word vectors."""
        ctx = words[-self.n:]
        if not ctx:
            return np.zeros(self.dim, dtype=np.int8)
        positioned = []
        for i, w in enumerate(ctx):
            wv = self._word_vec(w)
            offset_into_window = self.n - len(ctx) + i  # right-aligned
            positioned.append(np.roll(wv, offset_into_window * self.shift_step))
        return bundle(positioned)

    # ─── Training ────────────────────────────────────────────

    def train(self, sentences: list[str]) -> None:
        """Walk every sentence, store (prefix, next_word) for each position."""
        for sent in sentences:
            tokens = tokenize(sent)
            for i in range(len(tokens) - 1):
                prefix = tokens[max(0, i - self.n + 1): i + 1]
                next_word = tokens[i + 1]
                prefix_hv = self._encode_prefix(prefix)
                self.memories.append((prefix_hv, next_word, tuple(prefix)))
        # Build batch-search arrays
        if self.memories:
            self._stack = np.stack([m[0] for m in self.memories])
            self._next_words = [m[1] for m in self.memories]

    # ─── Inference ───────────────────────────────────────────

    def predict(self, prefix_words: list[str], top_k: int = 12,
                  temperature: float = 0.0) -> dict:
        """Predict the next word given a prefix.

        Returns: {
            "next_word": str,
            "confidence": float in [0,1],
            "nearest_distance": int,
            "votes": dict[str, float],
            "top_neighbors": [(distance, prefix_tuple, next_word), ...],
        }
        """
        if self._stack is None:
            raise RuntimeError("Model not trained")
        q = self._encode_prefix(prefix_words)
        xor = np.bitwise_xor(self._stack, q[None, :])
        dists = xor.sum(axis=1)
        nearest = np.argsort(dists)[:top_k]
        # Distance-weighted votes
        votes: dict[str, float] = {}
        for j in nearest:
            w = self._next_words[j]
            weight = 1.0 / (1 + int(dists[j]) / 100)
            votes[w] = votes.get(w, 0.0) + weight
        if temperature > 0:
            # Sample from the vote distribution
            words = list(votes.keys())
            weights = np.array([votes[w] for w in words])
            weights = weights ** (1.0 / temperature)
            weights = weights / weights.sum()
            winner = self.rng.choice(words, p=weights)
        else:
            winner = max(votes, key=votes.get)
        total = sum(votes.values())
        confidence = votes[winner] / total if total > 0 else 0.0
        top_neighbors = [
            (int(dists[j]), self.memories[j][2], self.memories[j][1])
            for j in nearest[:5]
        ]
        return {
            "next_word": winner,
            "confidence": float(confidence),
            "nearest_distance": int(dists[nearest[0]]),
            "votes": votes,
            "top_neighbors": top_neighbors,
        }

    def generate(self, prompt: str, n_words: int = 12,
                   temperature: float = 0.0,
                   stop_words: set[str] = None) -> str:
        """Generate n_words continuation of prompt by iterating predict()."""
        stop_words = stop_words or set()
        tokens = tokenize(prompt)
        produced = []
        for _ in range(n_words):
            result = self.predict(tokens, temperature=temperature)
            nw = result["next_word"]
            produced.append(nw)
            tokens.append(nw)
            if nw in stop_words:
                break
        return " ".join(produced)

    # ─── Inspection ──────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "n_gram": self.n,
            "vocab": len(self.word_vecs),
            "memories": len(self.memories),
        }
