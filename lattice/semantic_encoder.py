"""
lattice/semantic_encoder.py - THE encoder: meaning in, hypervectors out.

Replaces CorpusRIEncoder as Telp's text encoder. The RI encoder derived word
vectors from co-occurrence in whatever happened to be in the lattice - blind to
any phrasing it hadn't statistically seen ("raccoon" was noise). This encoder
maps text through a sentence-meaning model (MiniLM, local, ~80MB) and projects
into the SAME 10,000-bit hypervector space with a SHA-seeded random projection:

    text -> MiniLM 384-d meaning vector -> fixed LSH projection -> D-bit HV

Properties the mind gets from this:
  * "how far away is the moon?" ~= "the Moon orbits at an average distance..."
    - similarity by MEANING, zero shared words needed.
  * DETERMINISTIC FOREVER: model weights are frozen on disk and the projection
    is derived from a fixed SHA seed - unlike RI, the handwriting no longer
    depends on what the corpus contained or the order it was read. Encode the
    same text in any process, any year: same hypervector.
  * Same substrate: binary HVs, Hamming/dot similarity, bind/bundle all work.

Interface: drop-in for CorpusRIEncoder (the contract documented in
learned_encoder_adapter.py): dim, encode, encode_word, add_sentence, add_corpus,
index_vectors, stats, _idf_weight (corpus statistics kept for ranking bonuses).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

# portable model cache: TELP_MODEL_DIR > D:\hf_home (if D: exists) > ~/.cache/telp
_cache = os.environ.get("TELP_MODEL_DIR") or (
    r"D:\hf_home" if Path(r"D:\\").exists() else str(Path.home() / ".cache" / "telp"))
os.environ.setdefault("HF_HOME", _cache)

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SemanticEncoder:
    """MiniLM-backed drop-in replacement for CorpusRIEncoder."""

    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SEED = b"TELP_SEMANTIC_V1"

    def __init__(self, dim: int = 10000, device: str | None = None):
        self.dim = dim
        from sentence_transformers import SentenceTransformer
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self._model = SentenceTransformer(self.MODEL, device=device)
        emb_dim = self._model.get_sentence_embedding_dimension()
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha1(self.SEED).digest()[:8], "little"))
        self._proj = rng.standard_normal((emb_dim, dim)).astype(np.float32)
        # RI-compat surface: some legacy QA machinery touches these
        self.rng = np.random.default_rng(42)
        self.window = 5
        # corpus statistics for _idf_weight ranking bonuses (not for encoding)
        self.doc_freq: dict[str, int] = {}
        self.n_docs: int = 0
        # word-HV cache; exposed for code that reads encoder.index_vectors
        self.index_vectors: dict[str, np.ndarray] = {}
        self._sent_cache: dict[str, np.ndarray] = {}
        self._word_emb: dict[str, np.ndarray] = {}   # float word embeddings

    # ── encoding ─────────────────────────────────────────────────────
    def _embed(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(texts, convert_to_numpy=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def _to_hv(self, emb: np.ndarray) -> np.ndarray:
        return (emb @ self._proj > 0).astype(np.int8)

    def encode(self, text: str) -> np.ndarray:
        key = text.strip()[:512]
        hv = self._sent_cache.get(key)
        if hv is None:
            hv = self._to_hv(self._embed([key])[0])
            if len(self._sent_cache) < 20000:
                self._sent_cache[key] = hv
        return hv

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """(n, dim) int8 - one GPU pass, for re-encoding stores."""
        return self._to_hv(self._embed([t.strip()[:512] for t in texts]))

    def encode_word(self, word: str) -> np.ndarray:
        w = word.lower()
        if w not in self.index_vectors:
            self.index_vectors[w] = self._to_hv(self._embed([w])[0])
        return self.index_vectors[w]

    def encode_expanded(self, text: str, **_kw) -> np.ndarray:
        return self.encode(text)          # meaning already generalizes

    def focus_alignment(self, focus_words: list[str], texts: list[str],
                        reduce: str = "mean") -> list[float]:
        """Answer-selection signal: for each text, how well do its words cover
        the question's focus words, by word-level MEANING (cosine)? 'eat' aligns
        with 'eating'/'omnivorous' but not 'habitats'. reduce='mean' ranks
        candidates; reduce='min' tests COVERAGE (is any focus word left
        unanswered? -> facet-aware miss detection)."""
        toks_per = [list(dict.fromkeys(tokenize(t)))[:48] for t in texts]
        vocab = list(dict.fromkeys(
            [w for w in focus_words] + [w for ts in toks_per for w in ts]))
        need = [w for w in vocab if w not in self._word_emb]
        if need:
            for w, e in zip(need, self._embed(need)):
                self._word_emb[w] = e
        if not focus_words:
            return [0.0] * len(texts)
        F = np.stack([self._word_emb[w] for w in focus_words])
        out = []
        for ts in toks_per:
            if not ts:
                out.append(0.0)
                continue
            T = np.stack([self._word_emb[w] for w in ts])
            per_focus = (F @ T.T).max(axis=1)
            out.append(float(per_focus.min() if reduce == "min"
                             else per_focus.mean()))
        return out

    # ── corpus statistics (ranking only) ─────────────────────────────
    def add_sentence(self, sentence: str):
        toks = tokenize(sentence)
        self.n_docs += 1
        for t in set(toks):
            self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

    def add_corpus(self, sentences: list[str]):
        for s in sentences:
            self.add_sentence(s)

    def _idf_weight(self, word: str) -> float:
        if self.n_docs == 0:
            return 1.0
        df = self.doc_freq.get(word.lower(), 0)
        idf = math.log((self.n_docs + 1) / (df + 1)) + 1.0
        return max(0.5, min(idf, 6.0))

    def stats(self) -> dict:
        return {"vocab_size": len(self.doc_freq),
                "vocab_with_context": len(self.doc_freq),
                "dim": self.dim, "model": self.MODEL}
