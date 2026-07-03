"""
lattice/standalone_encoder.py - LLM-free semantic encoder.

Pure Random Indexing trained on the corpus as we ingest. No MiniLM,
no pre-trained embeddings — semantic similarity emerges from
distributional co-occurrence within the user's own corpus.

Architecture:
  - Each word gets a sparse random index vector (Sahlgren 1995/2005)
  - As sentences are ingested, each word's CONTEXT vector accumulates
    the index vectors of its neighbors within a window
  - Words appearing in similar contexts develop similar context vectors
  - Binary hypervector for a query = bundle of constituent words'
    binarized context vectors

This is the same RI technique we tested earlier — but now we run it
DYNAMICALLY on the live corpus instead of statically on a small
benchmark. The system gets better at semantic matching as it ingests
more text.

For multi-word queries, we bundle the constituent words' hypervectors.
For unknown words, we fall back to a character-level pure-HDC encoding.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, hamming_distance, bundle


_TOKEN_RE = re.compile(r"\b[\w-]+\b")


_FUNCTION_WORDS = {
    "the","a","an","is","was","were","are","be","been","being","has","have","had",
    "of","in","on","at","by","to","from","with","for","into","onto","through",
    "and","or","but","not","nor","yet","this","that","these","those","such",
    "it","its","they","their","them","he","she","his","her","him","i","we","us",
    "our","you","your","my","mine","as","than","then","also","both",
    "which","who","whom","whose","what","where","when","why","how","do","does","did",
    "if","because","while","during","since","before","after","until",
    "so","very","more","most","much","many","some","any","all","each","every",
    "between","among","over","under","above","below","near","s",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _FUNCTION_WORDS]


class CorpusRIEncoder:
    """Random Indexing encoder that trains live on ingested text.

    Drop-in replacement for the MiniLM-based TextEncoder, with the
    encode() method returning a D-dim binary hypervector.
    """

    def __init__(self, dim: int = D, sparsity: int = 20, window: int = 5,
                  seed: int = 42):
        self.dim = dim
        self.sparsity = sparsity
        self.window = window
        self.rng = np.random.default_rng(seed)
        self.index_vectors: dict[str, np.ndarray] = {}
        self.context_counters: dict[str, np.ndarray] = {}
        # Char vectors for unknown-word fallback
        self.char_vectors: dict[str, np.ndarray] = {}
        # Document frequency: how many sentences each word appeared in.
        # Used for IDF weighting at encode-time so rare distinctive words
        # (proper nouns) dominate the bundle over common content words.
        self.doc_freq: dict[str, int] = {}
        self.n_docs: int = 0

    # ─── Random index vectors ───────────────────────────────

    def _make_index(self) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.int32)
        idx = self.rng.choice(self.dim, size=self.sparsity, replace=False)
        half = self.sparsity // 2
        signs = np.concatenate([
            np.ones(half, dtype=np.int32),
            -np.ones(self.sparsity - half, dtype=np.int32)
        ])
        self.rng.shuffle(signs)
        v[idx] = signs
        return v

    def _ensure_word(self, w: str):
        if w not in self.index_vectors:
            self.index_vectors[w] = self._make_index()
            self.context_counters[w] = np.zeros(self.dim, dtype=np.int32)

    def _char_vec(self, c: str) -> np.ndarray:
        if c not in self.char_vectors:
            # SHA-seeded, NOT hash(): Python hash() is salted per process, which
            # made unknown-word encodings different noise every restart (the
            # hash-determinism bug, fixed here 2026-07-02 for the OOV path too).
            import hashlib
            local_seed = int.from_bytes(
                hashlib.sha1(c.encode("utf-8")).digest()[:4], "little")
            local_rng = np.random.default_rng(local_seed)
            self.char_vectors[c] = local_rng.integers(0, 2, size=self.dim, dtype=np.int8)
        return self.char_vectors[c]

    # ─── Training ────────────────────────────────────────────

    def add_sentence(self, sentence: str):
        """Accumulate context vectors based on co-occurrence in this sentence."""
        tokens = tokenize(sentence)
        for t in tokens:
            self._ensure_word(t)
        n = len(tokens)
        for i, w in enumerate(tokens):
            start = max(0, i - self.window)
            end = min(n, i + self.window + 1)
            for j in range(start, end):
                if i == j: continue
                self.context_counters[w] += self.index_vectors[tokens[j]]
        # Document-frequency update — counts unique words per sentence.
        self.n_docs += 1
        for t in set(tokens):
            self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

    def _idf_weight(self, word: str) -> float:
        """Smoothed inverse document frequency in [1, ~6].  Rare words
        get a larger weight so they dominate the bundle.  Unknown words
        get the max weight (we treat them as maximally distinctive)."""
        if self.n_docs == 0:
            return 1.0
        df = self.doc_freq.get(word, 0)
        # +1 smoothing; cap so a single rare word can't blow up the sum.
        import math
        idf = math.log((self.n_docs + 1) / (df + 1)) + 1.0
        return max(0.5, min(idf, 6.0))

    def add_corpus(self, sentences: list[str]):
        for s in sentences:
            self.add_sentence(s)

    # ─── Encoding (drop-in for MiniLM TextEncoder.encode) ──

    def _raw_word_vector(self, word: str) -> np.ndarray:
        """Return the raw (signed int32) semantic vector for a word.
        Used internally when bundling — keeps the bundle in dense space
        until the final sign-threshold.
        """
        w = word.lower()
        if w in self.context_counters:
            v = self.context_counters[w]
            if v.any():
                return v.astype(np.int32)
            # Word seen but no accumulated context — use its index vec.
            return self.index_vectors[w].astype(np.int32)
        # Unknown: bundle binary char vecs and re-cast to signed space.
        cv = self._encode_unknown_word(w)
        # Map {0,1} -> {-1,+1} so it contributes additively.
        return (cv.astype(np.int32) * 2 - 1)

    def encode_word(self, word: str) -> np.ndarray:
        v = self._raw_word_vector(word)
        return (v > 0).astype(np.int8)

    def _encode_unknown_word(self, word: str) -> np.ndarray:
        if not word:
            return np.zeros(self.dim, dtype=np.int8)
        pieces = []
        for i, c in enumerate(word):
            cv = self._char_vec(c)
            pieces.append(np.roll(cv, i * 7))
        return bundle(pieces)

    def encode(self, text: str) -> np.ndarray:
        """Encode a phrase/sentence into a D-bit binary hypervector.

        Strategy:
          1. Take content words (function words dropped).
          2. For each, look up its raw int32 context-vector.
          3. Normalize per-word to unit scale.
          4. Weight by IDF — rare proper nouns dominate.
          5. Threshold at the MEDIAN, not zero, so every document
             has exactly D/2 bits set.  This eliminates "magnet"
             documents whose short bundles accidentally match
             everything.
        """
        toks = content_tokens(text) or tokenize(text)
        if not toks:
            return np.zeros(self.dim, dtype=np.int8)
        acc = np.zeros(self.dim, dtype=np.int64)
        for t in toks:
            v = self._raw_word_vector(t).astype(np.int64)
            mx = int(np.abs(v).max())
            if mx == 0:
                continue
            idf = self._idf_weight(t)
            scale = int(1000 * idf)
            v = (v * scale) // mx
            acc += v
        # Force a balanced hypervector with exactly D/2 bits set, by
        # taking the top D/2 dimensions of `acc`.  Median-based
        # thresholding is unreliable when acc has many zero values
        # (short sentences) — argpartition gives a robust top-K split
        # that's invariant to document length.
        # Break ties deterministically by adding a tiny seeded jitter.
        if np.all(acc == 0):
            return np.zeros(self.dim, dtype=np.int8)
        jitter = self.rng.integers(0, 7, size=self.dim, dtype=np.int64)
        scored = acc * 7 + jitter
        top = np.argpartition(scored, self.dim // 2)[self.dim // 2:]
        out = np.zeros(self.dim, dtype=np.int8)
        out[top] = 1
        return out

    def stats(self) -> dict:
        return {
            "vocab_size":         len(self.index_vectors),
            "vocab_with_context": sum(1 for v in self.context_counters.values()
                                          if v.any()),
            "dim":                self.dim,
        }

    # ─── Query expansion ─────────────────────────────────────

    def nearest_words(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        """Return the k semantically nearest words to `word` by cosine
        similarity of context vectors.  Useful for query expansion.
        Empty list for unknown words.
        """
        w = word.lower()
        if w not in self.context_counters:
            return []
        target = self.context_counters[w].astype(np.float32)
        nt = np.linalg.norm(target)
        if nt == 0:
            return []
        target = target / nt
        scores: list[tuple[str, float]] = []
        for other, vec in self.context_counters.items():
            if other == w:
                continue
            v = vec.astype(np.float32)
            nv = np.linalg.norm(v)
            if nv == 0:
                continue
            sim = float(np.dot(target, v) / nv)
            scores.append((other, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def encode_expanded(self, text: str, neighbors_per_word: int = 3,
                          neighbor_weight: float = 0.5) -> np.ndarray:
        """Like encode() but, for each content word, also include its
        top-N nearest words at reduced weight.  Thickens thin queries
        so they match longer documents more reliably.
        """
        toks = content_tokens(text) or tokenize(text)
        if not toks:
            return np.zeros(self.dim, dtype=np.int8)
        acc = np.zeros(self.dim, dtype=np.int64)
        for t in toks:
            v = self._raw_word_vector(t).astype(np.int64)
            mx = int(np.abs(v).max())
            if mx == 0:
                continue
            idf = self._idf_weight(t)
            scale = int(1000 * idf)
            acc += (v * scale) // mx
            # Add neighbours at reduced weight (no IDF — they're
            # already filtered by similarity).
            for nbr, sim in self.nearest_words(t, k=neighbors_per_word):
                if sim < 0.15:
                    continue
                nv = self._raw_word_vector(nbr).astype(np.int64)
                nmx = int(np.abs(nv).max())
                if nmx == 0:
                    continue
                nbr_scale = int(1000 * sim * neighbor_weight)
                acc += (nv * nbr_scale) // nmx
        if np.all(acc == 0):
            return np.zeros(self.dim, dtype=np.int8)
        jitter = self.rng.integers(0, 7, size=self.dim, dtype=np.int64)
        scored = acc * 7 + jitter
        top = np.argpartition(scored, self.dim // 2)[self.dim // 2:]
        out = np.zeros(self.dim, dtype=np.int8)
        out[top] = 1
        return out
