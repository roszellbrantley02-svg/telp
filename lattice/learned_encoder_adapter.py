"""
lattice/learned_encoder_adapter.py - drop-in CorpusRIEncoder replacement
backed by GPU-trained DifferentiableTextEncoder vectors.

The rest of Telp's stack expects an encoder with this interface:
  * dim                       int
  * encode(text)              -> np.ndarray (D-bit binary)
  * encode_word(word)         -> np.ndarray (D-bit binary)
  * add_sentence(sentence)    -> None   (live vocab growth)
  * add_corpus(sentences)     -> None
  * index_vectors             dict[str, np.ndarray]
  * stats()                   -> dict

The differentiable encoder learns word vectors via GPU skip-gram, but
once trained, lookups are pure numpy — no torch dependency at inference.
add_sentence() on previously-unseen words falls back to a hash-derived
hypervector (same as the unknown-word path in DifferentiableTextEncoder).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.diff_text_encoder import (
    DifferentiableTextEncoder,
    _unknown_word_hv,
    tokenize,
)


def _content_tokens(s: str) -> list[str]:
    """Tokens of length >= 3 — drops most stopwords."""
    toks = tokenize(s)
    return [t for t in toks if len(t) >= 3]


class LearnedHDCEncoder:
    """Adapter exposing DifferentiableTextEncoder as a CorpusRIEncoder."""

    def __init__(self, diff_enc: DifferentiableTextEncoder):
        self._enc = diff_enc
        self.dim = diff_enc.dim
        # IndexVectors-style view so other code that touches
        # encoder.index_vectors keeps working.
        self.index_vectors: dict[str, np.ndarray] = {}
        for word, idx in diff_enc.vocab.items():
            self.index_vectors[word] = (
                diff_enc._bin_vectors[idx].numpy()
            )
        self.doc_freq: dict[str, int] = {}
        self.n_docs: int = 0
        # Mark live-learned words separately from frozen GPU vocab so
        # we know which need binary fallback vectors.
        self._live_added: dict[str, np.ndarray] = {}

    # ── Identity-style API the rest of the stack uses ─────────

    def encode_word(self, word: str) -> np.ndarray:
        return self._enc.encode_word(word)

    def encode(self, text: str) -> np.ndarray:
        return self._enc.encode(text)

    def add_sentence(self, sentence: str) -> None:
        """Live vocab growth.  Words new to the trained vocab get
        deterministic hash-derived hypervectors so they're at least
        reproducible and usable for retrieval until retrain time."""
        for t in tokenize(sentence):
            if (t not in self._enc.vocab and
                    t not in self._live_added):
                hv = _unknown_word_hv(t, self.dim)
                self._live_added[t] = hv
                self.index_vectors[t] = hv
        # DF counting (used by some downstream code)
        self.n_docs += 1
        for t in set(tokenize(sentence)):
            self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

    def add_corpus(self, sentences: list[str]) -> None:
        for s in sentences:
            self.add_sentence(s)

    def _idf_weight(self, word: str) -> float:
        """Compatibility shim mirroring CorpusRIEncoder._idf_weight.
        Rare words get higher weight so they dominate bundles."""
        import math
        if self.n_docs == 0:
            return 1.0
        df = self.doc_freq.get(word.lower(), 0)
        idf = math.log((self.n_docs + 1) / (df + 1)) + 1.0
        return max(0.5, min(idf, 6.0))

    def stats(self) -> dict:
        return {
            "vocab_size":         len(self.index_vectors),
            "vocab_with_context": len(self.index_vectors),
            "dim":                self.dim,
            "learned_words":      len(self._enc.vocab),
            "live_added":         len(self._live_added),
        }

    # ── Helpers some downstream code looks for ───────────────

    def encode_expanded(self, text: str, neighbors_per_word: int = 3,
                          per_neighbor_weight: float = 0.5) -> np.ndarray:
        """Compatibility shim — encodes the text plus nearest-neighbour
        words from the learned vocabulary, bundling them by majority
        vote.  Used by query expansion."""
        base_toks = _content_tokens(text) or tokenize(text)
        n_dim = self.dim
        acc = np.zeros(n_dim, dtype=np.int32)
        contributors = 0
        for t in base_toks:
            hv = self.encode_word(t)
            acc += hv.astype(np.int32)
            contributors += 1
            # Add neighbors
            nbrs = self._enc.nearest_neighbors(t, k=neighbors_per_word)
            for n_w, _sim in nbrs:
                hv2 = self.encode_word(n_w)
                # Weight neighbors lighter than the target
                if per_neighbor_weight >= 1.0:
                    acc += hv2.astype(np.int32)
                    contributors += 1
                else:
                    # Probabilistic inclusion proportional to weight
                    # — keeps the bundle arithmetic clean (integer).
                    if (np.random.default_rng().random()
                            < per_neighbor_weight):
                        acc += hv2.astype(np.int32)
                        contributors += 1
        if contributors == 0:
            return np.zeros(n_dim, dtype=np.int8)
        threshold = contributors / 2.0
        return (acc > threshold).astype(np.int8)
