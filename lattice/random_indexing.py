"""
lattice/random_indexing.py - HDC learns semantic geometry from raw text.

Sahlgren-style Random Indexing. Each word gets a sparse random "index
vector" (the word's identity). For each occurrence of a word in the
corpus, accumulate the index vectors of words in its context window.
After training, words with similar contexts (similar meanings) have
similar accumulated "context vectors".

No gradient descent. No neural network. No LLM. Just sparse vectors,
addition, and majority-vote binarization.

References:
  Sahlgren, M. (2005) - "An Introduction to Random Indexing"
  Kanerva, P. et al. (2000) - "Random indexing of text samples for
    latent semantic analysis"

Predates Word2Vec by 8 years and works on the same principle of
distributional semantics: "you shall know a word by the company it
keeps" (Firth 1957). The HDC twist is the use of sparse ±1 index
vectors and final binarization, making outputs compatible with our
HDC algebra (XOR, bundle, distance).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, hamming_distance, bundle


_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return [t for t in _TOKEN_RE.split(text.lower()) if t]


class RandomIndexingEncoder:
    """HDC encoder that learns semantic geometry from text co-occurrence."""

    def __init__(self, dim: int = D, sparsity: int = 20,
                  window: int = 5, seed: int = 42):
        """
        dim:      hypervector dimension (default 10000)
        sparsity: number of non-zero entries per index vector
        window:   context window size (±N words around each token)
        seed:     RNG seed for determinism
        """
        self.dim = dim
        self.sparsity = sparsity
        self.window = window
        self.rng = np.random.default_rng(seed)
        # word -> sparse ±1 index vector
        self.index_vectors: dict[str, np.ndarray] = {}
        # word -> accumulated int32 context vector
        self.context_vectors: dict[str, np.ndarray] = {}
        # word -> number of times we've seen it
        self.token_counts: dict[str, int] = {}

    def _make_index_vec(self) -> np.ndarray:
        """Sparse vector: `sparsity` random non-zero entries (half +1, half -1)."""
        v = np.zeros(self.dim, dtype=np.int32)
        idx = self.rng.choice(self.dim, size=self.sparsity, replace=False)
        # Half +1, half -1
        half = self.sparsity // 2
        signs = np.concatenate([
            np.ones(half, dtype=np.int32),
            -np.ones(self.sparsity - half, dtype=np.int32)
        ])
        self.rng.shuffle(signs)
        v[idx] = signs
        return v

    def _ensure_word(self, word: str) -> None:
        if word not in self.index_vectors:
            self.index_vectors[word]  = self._make_index_vec()
            self.context_vectors[word] = np.zeros(self.dim, dtype=np.int32)
            self.token_counts[word] = 0

    def train(self, sentences: Iterable[str]) -> None:
        """Accumulate context vectors by walking through sentences."""
        for sentence in sentences:
            tokens = tokenize(sentence)
            # First pass: make sure every token has an index vector
            for t in tokens:
                self._ensure_word(t)
                self.token_counts[t] += 1
            # Second pass: accumulate context
            n = len(tokens)
            for i, w in enumerate(tokens):
                start = max(0, i - self.window)
                end   = min(n, i + self.window + 1)
                for j in range(start, end):
                    if i == j:
                        continue
                    self.context_vectors[w] += self.index_vectors[tokens[j]]

    # ─── Encoding ─────────────────────────────────────────────

    def encode_word(self, word: str) -> np.ndarray:
        """Binarize a single word's context vector into an HDC hypervector."""
        w = word.lower()
        if w not in self.context_vectors:
            # Unknown word - return all-zero (will land far from everything)
            return np.zeros(self.dim, dtype=np.int8)
        return (self.context_vectors[w] > 0).astype(np.int8)

    def encode(self, text: str) -> np.ndarray:
        """Encode a phrase by bundling its word hypervectors.

        Multi-word names like 'United Kingdom' or 'Buenos Aires' get
        bundled across their constituent words.
        """
        tokens = tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.int8)
        word_hvs = [self.encode_word(t) for t in tokens]
        # Drop all-zero (unknown) words from the bundle
        word_hvs = [v for v in word_hvs if v.sum() > 0]
        if not word_hvs:
            return np.zeros(self.dim, dtype=np.int8)
        return bundle(word_hvs)

    # ─── Inspection ──────────────────────────────────────────

    def vocab_size(self) -> int:
        return len(self.index_vectors)

    def top_neighbors(self, word: str, k: int = 5) -> list[tuple[str, int]]:
        """Return the k most similar words by Hamming distance on context vecs."""
        target = self.encode_word(word)
        if target.sum() == 0:
            return []
        scored = []
        for w in self.context_vectors:
            if w == word.lower():
                continue
            v = self.encode_word(w)
            if v.sum() == 0:
                continue
            d = hamming_distance(target, v)
            scored.append((w, d))
        scored.sort(key=lambda x: x[1])
        return scored[:k]
