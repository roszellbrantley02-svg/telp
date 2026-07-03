"""
lattice/image_gallery.py - text<->image binding store for HDC generation.

The gallery is a persistent collection of (caption, image_hv) pairs.
Two retrieval modes:

  1. Direct lookup
       Encode the query caption into a text hypervector.  Find the
       gallery entry whose caption hypervector is most similar.  Decode
       the bound image hypervector through the RGB codec.  Effectively
       a content-addressable image lookup.

  2. Algebraic composition  (the genuine HDC-native path)
       Each gallery entry stores the BINDING  text_hv * image_hv.
       Bundle of all bindings creates an associative memory.
       At generation time:
         query_text_hv = encoder.encode("red sunset over ocean")
         composed_image_hv = unbind(query_text_hv, gallery_bundle)
         decoded = codec.decode(composed_image_hv)

       This is the Plate HRR / Kanerva associative-memory mechanism.
       It composes images from the algebraic "between" of memories
       similar to the query — a genuine generation step.

For text encoding we lean on the existing DifferentiableTextEncoder /
CorpusRIEncoder via a small wrapper that just wants .encode(str)->hv.

For image encoding we use HDCImageCodecRGB.

This file has zero neural-net dependency.  All HDC algebra.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, bundle, bind
from lattice.image_vq_rgb import HDCImageCodecRGB


class HDCImageGallery:
    """Text<->image binding store + algebraic generation."""

    def __init__(self, codec: HDCImageCodecRGB, text_encoder):
        """
        codec:        an HDCImageCodecRGB with a built codebook
        text_encoder: any object with .encode(text)->np.ndarray(int8, D)
        """
        self.codec = codec
        self.text_encoder = text_encoder
        # Per-pair storage
        self.captions: list[str] = []
        self.image_paths: list[str] = []
        self.text_hvs: np.ndarray | None = None     # (N, D) int8
        self.image_hvs: np.ndarray | None = None    # (N, D) int8
        # Associative bundle: sum of (text_hv XOR image_hv) across pairs
        # Stored as int32 sum so we can incrementally extend; we sign-
        # threshold to int8 when we need to use it.
        self._assoc_acc: np.ndarray | None = None
        self._assoc_count: int = 0

    # ─── Add pairs ─────────────────────────────────────────────

    def add(self, caption: str, image_path: str) -> int:
        """Add one (caption, image) pair to the gallery."""
        t_hv = self.text_encoder.encode(caption)
        if t_hv.dtype != np.int8:
            t_hv = t_hv.astype(np.int8)
        try:
            i_hv = self.codec.encode(image_path)
        except Exception as e:
            print(f"[gallery] skip {image_path}: {e}")
            return -1
        if self.text_hvs is None:
            self.text_hvs  = t_hv[None, :].copy()
            self.image_hvs = i_hv[None, :].copy()
        else:
            self.text_hvs  = np.vstack([self.text_hvs,  t_hv[None, :]])
            self.image_hvs = np.vstack([self.image_hvs, i_hv[None, :]])
        # Update associative bundle: text_hv XOR image_hv summed
        binding = np.bitwise_xor(t_hv, i_hv).astype(np.int32)
        if self._assoc_acc is None:
            self._assoc_acc = binding.copy()
        else:
            self._assoc_acc += binding
        self._assoc_count += 1
        self.captions.append(caption)
        self.image_paths.append(image_path)
        return len(self.captions) - 1

    def add_many(self, pairs: list[tuple[str, str]],
                      verbose: bool = True) -> int:
        n_added = 0
        for i, (cap, path) in enumerate(pairs):
            if self.add(cap, path) >= 0:
                n_added += 1
            if verbose and (i + 1) % 100 == 0:
                print(f"[gallery]   added {n_added}/{i+1}", flush=True)
        return n_added

    # ─── Retrieval / generation ─────────────────────────────────

    def _assoc_bundle(self) -> np.ndarray:
        """Materialize the associative bundle as a binary hypervector
        via majority threshold."""
        if self._assoc_acc is None or self._assoc_count == 0:
            return np.zeros(D, dtype=np.int8)
        half = self._assoc_count / 2.0
        return (self._assoc_acc > half).astype(np.int8)

    def retrieve(self, caption: str, k: int = 5) -> list[dict]:
        """Direct nearest-caption lookup.  Returns top-k matches."""
        if self.text_hvs is None or len(self.captions) == 0:
            return []
        t_hv = self.text_encoder.encode(caption)
        if t_hv.dtype != np.int8:
            t_hv = t_hv.astype(np.int8)
        xor = np.bitwise_xor(self.text_hvs, t_hv[None, :])
        dists = xor.sum(axis=1)
        order = np.argsort(dists)[:k]
        return [{
            "caption":    self.captions[i],
            "image_path": self.image_paths[i],
            "distance":   int(dists[i]),
            "similarity": float(1.0 - 2.0 * dists[i] / D),
            "image_hv":   self.image_hvs[i],
        } for i in order]

    def generate(self, caption: str, blend_k: int = 3) -> np.ndarray:
        """Algebraic image generation.

        Two-stage process:
          A) Unbind the query caption from the associative bundle to
             get a coarse "image candidate".
          B) Blend with the top-k nearest stored captions' image
             hypervectors via bundle.

        Returns: a hypervector you can pass to codec.decode().
        """
        if self.text_hvs is None or len(self.captions) == 0:
            return np.zeros(D, dtype=np.int8)
        t_hv = self.text_encoder.encode(caption)
        if t_hv.dtype != np.int8:
            t_hv = t_hv.astype(np.int8)
        # Step A: unbind from associative bundle
        assoc = self._assoc_bundle()
        coarse = np.bitwise_xor(t_hv, assoc)
        # Step B: nearest captions, blend their image vectors
        top = self.retrieve(caption, k=blend_k)
        if not top:
            return coarse
        hvs_to_bundle = [coarse]
        for h in top:
            hvs_to_bundle.append(h["image_hv"])
        return bundle(hvs_to_bundle)

    # ─── Persistence ───────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "captions":     self.captions,
                "image_paths":  self.image_paths,
                "text_hvs":     self.text_hvs,
                "image_hvs":    self.image_hvs,
                "_assoc_acc":   self._assoc_acc,
                "_assoc_count": self._assoc_count,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path,
                codec: HDCImageCodecRGB,
                text_encoder) -> "HDCImageGallery":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        g = cls(codec, text_encoder)
        g.captions      = blob["captions"]
        g.image_paths   = blob["image_paths"]
        g.text_hvs      = blob["text_hvs"]
        g.image_hvs     = blob["image_hvs"]
        g._assoc_acc    = blob["_assoc_acc"]
        g._assoc_count  = blob["_assoc_count"]
        return g

    def stats(self) -> dict:
        return {
            "n_pairs":          len(self.captions),
            "image_size":       self.codec.image_size,
            "patch_size":       self.codec.patch_size,
            "codebook_size":    self.codec.codebook_size,
            "hv_bytes":         D // 8,
            "assoc_bundle_set": self._assoc_count,
        }
