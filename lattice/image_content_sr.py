"""
lattice/image_content_sr.py - content-aware HDC super-resolution.

v1 tried to learn low-res-patch -> high-res-patch directly (pixel-thinking).
Bicubic beat it because we were competing on bicubic's home turf.

v2 thinks differently:

    encode(patch) -> content_features_hv     (resolution-invariant)
    library:   (content_features_hv, high_res_patch) pairs from training
    at inference:
        for each low-res patch in input:
            features = extract resolution-invariant features
            look up matching high-res patch in library by feature similarity
            paste it

Features are designed to be (approximately) scale-invariant:
  * color statistics (mean RGB, std RGB)
  * edge orientation histogram (Sobel-based)
  * brightness / variance
  * dominant gradient direction

A 4x4 patch and an 8x8 patch of the same content should produce
similar feature hypervectors.  That means we can index the
high-res library by features extracted from low-res input —
the whole point.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, bundle, bind


# ─── Feature extractors (resolution-invariant) ─────────────────────


def _patch_features(patch: np.ndarray) -> dict:
    """Extract scale-invariant features from an RGB patch (H, W, 3).
    Returns dict of feature -> quantized bin index.
    """
    patch = patch.astype(np.float32)
    H, W, _ = patch.shape
    # Color: mean R, G, B (quantized to 16 levels each)
    mean_rgb = patch.mean(axis=(0, 1))                  # (3,)
    color_bins = np.clip((mean_rgb / 16).astype(np.int32), 0, 15)
    # Brightness: luminance (quantized to 16 levels)
    luma = 0.299*patch[:,:,0] + 0.587*patch[:,:,1] + 0.114*patch[:,:,2]
    bright_bin = int(np.clip(luma.mean() / 16, 0, 15))
    # Contrast: stdev of luma (16 levels)
    contrast_bin = int(np.clip(luma.std() / 8, 0, 15))
    # Edge orientation: Sobel gradients, quantize dominant angle
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    if H >= 3 and W >= 3:
        gx[1:-1, 1:-1] = (luma[1:-1, 2:] - luma[1:-1, :-2]) * 0.5
        gy[1:-1, 1:-1] = (luma[2:, 1:-1] - luma[:-2, 1:-1]) * 0.5
    mag = np.sqrt(gx**2 + gy**2)
    if mag.sum() > 1e-6:
        ang = np.arctan2(gy, gx)
        weights = mag.flatten()
        angles = ang.flatten()
        # Histogram-weight average wrapped: use sum of (cos, sin)*mag
        cx = (np.cos(angles) * weights).sum()
        cy = (np.sin(angles) * weights).sum()
        dom_ang = np.arctan2(cy, cx)         # in [-pi, pi]
        edge_bin = int(((dom_ang + np.pi) / (2 * np.pi) * 8)) % 8
        edge_mag_bin = int(np.clip(mag.mean() / 8, 0, 7))
    else:
        edge_bin = 0
        edge_mag_bin = 0
    return {
        "r_bin":      int(color_bins[0]),
        "g_bin":      int(color_bins[1]),
        "b_bin":      int(color_bins[2]),
        "bright_bin": bright_bin,
        "contrast_bin": contrast_bin,
        "edge_orient_bin": edge_bin,
        "edge_mag_bin":  edge_mag_bin,
    }


# ─── HDC feature -> hypervector ────────────────────────────────────


class FeatureVocab:
    """Manages randomly-assigned hypervectors per feature bin."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._cache: dict[str, np.ndarray] = {}

    def get(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = self.rng.integers(0, 2, size=D, dtype=np.int8)
        return self._cache[key]


def features_to_hv(feats: dict, vocab: FeatureVocab) -> np.ndarray:
    """Bind each (feature_name, bin_value) into a hypervector and bundle."""
    parts = []
    for name, value in feats.items():
        role = vocab.get(f"role:{name}")
        val_hv = vocab.get(f"{name}:{value}")
        parts.append(np.bitwise_xor(role, val_hv))
    return bundle(parts)


# ─── Content-aware library + super-resolution ──────────────────────


class HDCContentSR:
    """Content-feature indexed super-resolution.

    Training:
      For each high-res training image, for each high-res patch:
        feats = features extracted from that patch
        hv = features_to_hv(feats)
        store (hv, patch_pixels) in library

    Inference (upscale lo_img to target hi resolution):
      For each lo-res patch:
        feats = extract features (resolution-invariant)
        hv = features_to_hv(feats)
        query library for nearest hv -> high-res patch
        paste at the corresponding target position
    """

    def __init__(self, hi_size: int = 32, lo_size: int = 8,
                   hi_patch: int = 8, lo_patch: int = 2,
                   library_size: int = 8192,
                   seed: int = 42):
        assert hi_size % hi_patch == 0
        assert lo_size % lo_patch == 0
        n_hi = hi_size // hi_patch
        n_lo = lo_size // lo_patch
        assert n_hi == n_lo
        self.hi_size = hi_size
        self.lo_size = lo_size
        self.hi_patch = hi_patch
        self.lo_patch = lo_patch
        self.scale = hi_size // lo_size
        self.n_per_side = n_hi
        self.library_size = library_size
        self.vocab = FeatureVocab(seed=seed)
        # Library: (N, D) int8 hypervectors and aligned hi-res patches
        self.lib_hvs: np.ndarray | None = None
        self.lib_patches: np.ndarray | None = None    # (N, hp, hp, 3)

    # ── Patch helpers ──────────────────────────────────────────

    def _extract_patches(self, img: np.ndarray, ps: int) -> np.ndarray:
        H = img.shape[0]
        n = H // ps
        out = []
        for r in range(n):
            for c in range(n):
                out.append(img[r*ps:(r+1)*ps, c*ps:(c+1)*ps, :])
        return np.stack(out)

    def _patch_hv(self, patch: np.ndarray) -> np.ndarray:
        feats = _patch_features(patch)
        return features_to_hv(feats, self.vocab)

    # ── Training ───────────────────────────────────────────────

    def train(self, hi_paths: list, max_images: int = 5000,
                 verbose: bool = True) -> None:
        if verbose:
            print(f"[csr] building content-indexed library from up to "
                    f"{max_images:,} images", flush=True)
        all_hvs: list[np.ndarray] = []
        all_patches: list[np.ndarray] = []
        for i, p in enumerate(hi_paths[:max_images]):
            try:
                img = Image.open(p).convert("RGB").resize(
                    (self.hi_size, self.hi_size), Image.LANCZOS
                )
                hi = np.asarray(img, dtype=np.float32)
                hi_p = self._extract_patches(hi, self.hi_patch)
                for hp in hi_p:
                    all_hvs.append(self._patch_hv(hp))
                    all_patches.append(hp)
            except Exception as e:
                if verbose:
                    print(f"[csr] skip {p}: {e}")
            if verbose and (i + 1) % 500 == 0:
                print(f"[csr]   processed {i+1}/"
                        f"{min(len(hi_paths), max_images)}  "
                        f"({len(all_hvs):,} patches)", flush=True)
        if not all_hvs:
            raise RuntimeError("no training patches collected")
        # Subsample if library is too big
        if len(all_hvs) > self.library_size:
            idx = self.vocab.rng.choice(
                len(all_hvs), size=self.library_size, replace=False
            )
            self.lib_hvs = np.stack([all_hvs[j] for j in idx])
            self.lib_patches = np.stack([all_patches[j] for j in idx])
        else:
            self.lib_hvs = np.stack(all_hvs)
            self.lib_patches = np.stack(all_patches)
        if verbose:
            print(f"[csr] library size: {len(self.lib_hvs):,}", flush=True)

    # ── Inference ──────────────────────────────────────────────

    def _query_library(self, lo_patch: np.ndarray) -> np.ndarray:
        """Look up the high-res patch whose features best match lo_patch."""
        hv = self._patch_hv(lo_patch)
        # Hamming distance against all library entries
        xor = np.bitwise_xor(self.lib_hvs, hv[None, :])
        dists = xor.sum(axis=1)
        idx = int(np.argmin(dists))
        return self.lib_patches[idx]

    def upscale(self, lo_img: np.ndarray) -> np.ndarray:
        if self.lib_hvs is None:
            raise RuntimeError("library not trained")
        lo = lo_img.astype(np.float32)
        if lo.shape[0] != self.lo_size:
            lo = np.asarray(
                Image.fromarray(np.clip(lo, 0, 255).astype(np.uint8))
                     .resize((self.lo_size, self.lo_size), Image.LANCZOS),
                dtype=np.float32,
            )
        lo_p = self._extract_patches(lo, self.lo_patch)
        recon = np.zeros((self.hi_size, self.hi_size, 3), dtype=np.float32)
        hp = self.hi_patch
        for i, patch in enumerate(lo_p):
            best_hi = self._query_library(patch)
            r = i // self.n_per_side
            c = i %  self.n_per_side
            recon[r*hp:(r+1)*hp, c*hp:(c+1)*hp, :] = best_hi
        return np.clip(recon, 0, 255).astype(np.uint8)

    # ── Persistence ────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "hi_size":      self.hi_size,
                "lo_size":      self.lo_size,
                "hi_patch":     self.hi_patch,
                "lo_patch":     self.lo_patch,
                "library_size": self.library_size,
                "vocab_cache":  self.vocab._cache,
                "lib_hvs":      self.lib_hvs,
                "lib_patches":  self.lib_patches,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "HDCContentSR":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        c = cls(
            hi_size=blob["hi_size"], lo_size=blob["lo_size"],
            hi_patch=blob["hi_patch"], lo_patch=blob["lo_patch"],
            library_size=blob["library_size"],
        )
        c.vocab._cache = blob["vocab_cache"]
        c.lib_hvs      = blob["lib_hvs"]
        c.lib_patches  = blob["lib_patches"]
        return c
