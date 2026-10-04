"""
A stand-in for sentence_transformers: a hashed bag-of-words "sentence
model". Texts that share words get similar vectors, which is all routing
and retrieval tests need - no network, no 80MB MiniLM download, no torch.

    from tests.fake_minilm import install; install()
"""
from __future__ import annotations

import hashlib
import re
import sys
import types

import numpy as np

_DIM = 384
_word_vecs: dict[str, np.ndarray] = {}


def _stem(w: str) -> str:
    """Crude stemming so flows/flow, eggs/egg, boiling/boil land together -
    the morphological robustness a real sentence model has."""
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    return w[:6]


def _word_vec(w: str) -> np.ndarray:
    key = _stem(w)
    v = _word_vecs.get(key)
    if v is None:
        seed = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(_DIM).astype(np.float32)
        _word_vecs[key] = v
    return v


class FakeSentenceTransformer:
    def __init__(self, *_a, **_kw):
        pass

    def get_sentence_embedding_dimension(self) -> int:
        return _DIM

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        out = np.zeros((len(texts), _DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9']+", t.lower()):
                out[i] += _word_vec(w)
            n = float(np.linalg.norm(out[i]))
            if n:
                out[i] /= n
        return out[0] if single else out


def install() -> None:
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = mod
