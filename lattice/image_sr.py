"""
lattice/image_sr.py - HDC super-resolution via a paired-resolution codebook.

The setup:
  * For each training image, hold both versions:
      - hi:  the original (say 32x32 RGB)
      - lo:  a downsampled copy (say 8x8, 4x smaller)
  * Tile each into aligned patches:
      - lo_patch:  small (e.g. 2x2)
      - hi_patch:  large (e.g. 8x8)  — same physical region as lo_patch
  * K-means cluster the lo_patches → N atoms
  * For each cluster, average the aligned hi_patches → N "expected" hi atoms

At inference: split the low-res input into lo_patches, find each one's
nearest lo atom, paste the paired hi atom into the right spot. Boundaries
are blended via simple 1-pixel overlap averaging.

Pure HDC algebra — actually pure K-means + indexing.  No neural net.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))


def downsample_avg(rgb: np.ndarray, scale: int) -> np.ndarray:
    """Average-pool a (H, W, 3) image by `scale` per side.  Returns
    (H/scale, W/scale, 3)."""
    H, W, C = rgb.shape
    assert H % scale == 0 and W % scale == 0
    hh, ww = H // scale, W // scale
    return rgb.reshape(hh, scale, ww, scale, C).mean(axis=(1, 3))


class HDCSuperResCodec:
    """Paired-resolution codebook super-resolver."""

    def __init__(self,
                   hi_size: int = 32,
                   lo_size: int = 8,
                   hi_patch: int = 8,
                   lo_patch: int = 2,
                   codebook_size: int = 1024,
                   seed: int = 42):
        """
        hi_size:        high-res target side length
        lo_size:        low-res input side length (must divide hi_size)
        hi_patch:       high-res patch side
        lo_patch:       low-res patch side
        codebook_size:  number of paired atoms

        Must satisfy:
          hi_size // hi_patch == lo_size // lo_patch  (same patch grid)
          hi_size / lo_size == hi_patch / lo_patch   (same scale factor)
        """
        assert hi_size % hi_patch == 0
        assert lo_size % lo_patch == 0
        n_per_side_hi = hi_size // hi_patch
        n_per_side_lo = lo_size // lo_patch
        assert n_per_side_hi == n_per_side_lo, \
            f"patch grids don't match ({n_per_side_hi} vs {n_per_side_lo})"
        assert hi_size // lo_size == hi_patch // lo_patch, \
            "scale factors don't match"
        self.hi_size = hi_size
        self.lo_size = lo_size
        self.hi_patch = hi_patch
        self.lo_patch = lo_patch
        self.scale = hi_size // lo_size
        self.n_per_side = n_per_side_hi
        self.n_patches = self.n_per_side ** 2
        self.codebook_size = codebook_size
        self.rng = np.random.default_rng(seed)
        # Filled by train()
        self.lo_atoms: np.ndarray | None = None   # (K, lo_patch*lo_patch*3)
        self.hi_atoms: np.ndarray | None = None   # (K, hi_patch, hi_patch, 3)

    # ─── Patch extraction ──────────────────────────────────────

    def _extract_patches(self, img: np.ndarray, ps: int) -> np.ndarray:
        H = img.shape[0]
        n = H // ps
        patches = []
        for r in range(n):
            for c in range(n):
                patches.append(img[r*ps:(r+1)*ps, c*ps:(c+1)*ps, :])
        return np.stack(patches)

    # ─── Train the paired codebook ─────────────────────────────

    def train(self, hi_paths: list, max_images: int = 5_000,
                 verbose: bool = True) -> None:
        if verbose:
            print(f"[sr] training paired codebook "
                    f"({self.lo_patch}x{self.lo_patch} -> "
                    f"{self.hi_patch}x{self.hi_patch}, "
                    f"K={self.codebook_size}) on up to "
                    f"{max_images:,} images", flush=True)
        lo_list: list[np.ndarray] = []
        hi_list: list[np.ndarray] = []
        used = 0
        for i, p in enumerate(hi_paths[:max_images]):
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((self.hi_size, self.hi_size), Image.LANCZOS)
                hi = np.asarray(img, dtype=np.float32)
                lo = downsample_avg(hi, self.scale)
                lo_p = self._extract_patches(lo, self.lo_patch)
                hi_p = self._extract_patches(hi, self.hi_patch)
                # Aligned pairing — same patch index = same physical region
                lo_list.append(lo_p)
                hi_list.append(hi_p)
                used += 1
            except Exception as e:
                if verbose:
                    print(f"[sr] skip {p}: {e}")
            if verbose and (i + 1) % 500 == 0:
                print(f"[sr]   processed {i+1}/{min(len(hi_paths), max_images)}",
                        flush=True)
        if not lo_list:
            raise RuntimeError("no usable training images")
        lo_arr = np.concatenate(lo_list, axis=0)   # (N_patches, lp, lp, 3)
        hi_arr = np.concatenate(hi_list, axis=0)   # (N_patches, hp, hp, 3)
        lo_flat = lo_arr.reshape(len(lo_arr), -1)
        if verbose:
            print(f"[sr] clustering {len(lo_flat):,} low-res patches into "
                    f"{self.codebook_size} clusters ...", flush=True)
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(
            n_clusters=self.codebook_size,
            random_state=42,
            batch_size=1024,
            n_init=3,
            max_iter=100,
        )
        labels = km.fit_predict(lo_flat)
        # lo_atoms: centroids
        self.lo_atoms = km.cluster_centers_
        # hi_atoms: per-cluster average of aligned hi patches
        hi_atoms = np.zeros(
            (self.codebook_size, self.hi_patch, self.hi_patch, 3),
            dtype=np.float32
        )
        counts = np.zeros(self.codebook_size, dtype=np.int64)
        for idx in range(len(labels)):
            c = labels[idx]
            hi_atoms[c] += hi_arr[idx]
            counts[c] += 1
        # Avoid divide-by-zero for empty clusters
        counts = np.maximum(counts, 1)
        self.hi_atoms = hi_atoms / counts[:, None, None, None]
        if verbose:
            empty = int((counts == 1).sum())
            print(f"[sr] trained — {self.codebook_size} atoms "
                    f"({empty} were empty/singleton)", flush=True)

    # ─── Inference ─────────────────────────────────────────────

    def upscale(self, lo_img: np.ndarray) -> np.ndarray:
        """Upscale (lo_size, lo_size, 3) → (hi_size, hi_size, 3) uint8."""
        if self.lo_atoms is None:
            raise RuntimeError("codec not trained")
        if lo_img.ndim == 2:
            lo_img = np.stack([lo_img]*3, axis=-1)
        lo_img = lo_img.astype(np.float32)
        if lo_img.shape[0] != self.lo_size:
            lo_img = np.asarray(
                Image.fromarray(np.clip(lo_img, 0, 255).astype(np.uint8))
                     .resize((self.lo_size, self.lo_size), Image.LANCZOS),
                dtype=np.float32,
            )
        lo_p = self._extract_patches(lo_img, self.lo_patch)
        # Find nearest atom for each patch via L2 in flat space
        flat = lo_p.reshape(len(lo_p), -1)
        # squared distance B x K
        dists = (
            np.sum(flat**2, axis=1, keepdims=True)
            - 2 * flat @ self.lo_atoms.T
            + np.sum(self.lo_atoms**2, axis=1, keepdims=True).T
        )
        nearest = np.argmin(dists, axis=1)
        # Tile the paired hi atoms
        recon = np.zeros((self.hi_size, self.hi_size, 3), dtype=np.float32)
        hp = self.hi_patch
        for i, atom_idx in enumerate(nearest):
            r = i // self.n_per_side
            c = i %  self.n_per_side
            recon[r*hp:(r+1)*hp, c*hp:(c+1)*hp, :] = self.hi_atoms[atom_idx]
        return np.clip(recon, 0, 255).astype(np.uint8)

    # ─── Persistence ───────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "hi_size":   self.hi_size,
                "lo_size":   self.lo_size,
                "hi_patch":  self.hi_patch,
                "lo_patch":  self.lo_patch,
                "codebook_size": self.codebook_size,
                "lo_atoms":  self.lo_atoms,
                "hi_atoms":  self.hi_atoms,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "HDCSuperResCodec":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        c = cls(
            hi_size=blob["hi_size"], lo_size=blob["lo_size"],
            hi_patch=blob["hi_patch"], lo_patch=blob["lo_patch"],
            codebook_size=blob["codebook_size"],
        )
        c.lo_atoms = blob["lo_atoms"]
        c.hi_atoms = blob["hi_atoms"]
        return c
