"""
lattice/text_encoder.py - turn English into hypervectors.

The bridge between language and the math substrate. Pipeline:

  sentence (str)
    -> sentence-transformers MiniLM (384-dim dense float)
    -> random-hyperplane LSH projection to D-dim binary
    -> 10k-bit hypervector

Why this works
--------------
Random-hyperplane projection is a form of LSH that preserves
COSINE similarity in HAMMING distance:

    cos(emb_a, emb_b) approx 1 - 2 * ham(hv_a, hv_b) / D

So two semantically similar sentences (cos ~ 0.8) land at Hamming
distance ~10% of D. Two unrelated sentences (cos ~ 0) land at ~50% of D.
The hypervector inherits the meaning structure of the dense embedding,
but in the binary domain where HDC's algebra (XOR bind, majority bundle,
permute) is cheap and exact.

We freeze the projection matrix with a fixed seed so the encoder is
deterministic across runs.

References:
  Charikar (2002) - SimHash
  Kanerva (1988)  - Sparse Distributed Memory
  Reimers & Gurevych (2019) - sentence-transformers
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, hamming_distance  # noqa: E402


# ─── Encoder ───────────────────────────────────────────────────────


_MINILM_MODEL_NAME = "all-MiniLM-L6-v2"     # 384-dim, ~22M params, CPU-fast
_PROJECTION_SEED = 17                         # deterministic hyperplanes


class TextEncoder:
    """sentence -> dense embedding -> binary hypervector via LSH.

    Loads the MiniLM once at construction. The projection matrix W
    (shape D x embed_dim) is randomly generated with a fixed seed so
    every TextEncoder instance produces the same hypervectors.
    """

    def __init__(self, model_name: str = _MINILM_MODEL_NAME,
                  device: str = "cpu"):
        # Lazy import — sentence-transformers pulls in a lot
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.embed_dim = self.model.get_sentence_embedding_dimension()
        # Random hyperplanes for LSH projection
        rng = np.random.default_rng(_PROJECTION_SEED)
        # Gaussian works as well as uniform for this; gaussian is standard.
        self.W = rng.standard_normal((D, self.embed_dim)).astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        """One sentence -> one 10k-bit binary hypervector."""
        emb = self.model.encode(text, convert_to_numpy=True,
                                  normalize_embeddings=True)
        # emb shape: (embed_dim,)
        projection = self.W @ emb
        return (projection > 0).astype(np.int8)

    def encode_many(self, texts: Iterable[str]) -> np.ndarray:
        """Batch encode N sentences -> (N, D) binary array."""
        texts = list(texts)
        emb = self.model.encode(texts, convert_to_numpy=True,
                                  normalize_embeddings=True,
                                  show_progress_bar=False)
        # emb shape: (N, embed_dim)
        projections = emb @ self.W.T   # (N, D)
        return (projections > 0).astype(np.int8)


# ─── Convenience: cosine of two dense embeddings, ham of two hvs ───


def hv_distance_pct(a: np.ndarray, b: np.ndarray) -> float:
    """Hamming distance as fraction of D. 0 = identical, 0.5 = random,
    1.0 = exact opposite."""
    return hamming_distance(a, b) / D


def hv_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Maps Hamming back to a cosine-like score in [-1, 1]. 1 = identical."""
    return 1.0 - 2.0 * hv_distance_pct(a, b)
