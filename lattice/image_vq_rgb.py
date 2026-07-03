"""
lattice/image_vq_rgb.py - color HDC image codec.

Extends the grayscale HDCImageCodec idea to RGB.  Each patch is an
8x8x3 = 192-dimensional vector (R,G,B per pixel).  A shared K-means
codebook of N visual atoms is sampled from training images.  An image
becomes a hypervector via:

    image_hv = bundle_i( bind(position_role_i, atom_hv[nearest(patch_i)]) )

Decode reverses: unbind each position to recover the atom_hv, cleanup
to the nearest codebook entry, paste the RGB pixels back into place.

Pure HDC algebra.  No neural network.  K-means is the only learning
step — a 1960s algorithm, not gradient-trained.

This is the foundation for text-to-image generation via the HDC
image gallery (see image_gallery.py).
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


class HDCImageCodecRGB:
    """RGB version of the patch-codebook HDC image codec."""

    def __init__(self, image_size: int = 64, patch_size: int = 8,
                   codebook_size: int = 1024, seed: int = 42):
        """
        image_size:    target square size (e.g. 64 -> 64x64 RGB)
        patch_size:    side length of each patch
        codebook_size: number of visual atoms in the shared codebook
        """
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_per_side = image_size // patch_size
        self.n_patches = self.n_per_side ** 2
        self.codebook_size = codebook_size
        self.rng = np.random.default_rng(seed)
        # Codebook of (codebook_size, patch_size*patch_size*3) RGB patches
        self.codebook_pixels: np.ndarray | None = None
        # Random hypervector per atom
        self.codebook_hvs: np.ndarray | None = None
        # Position roles, one per patch position
        self.position_roles = np.stack([
            self.rng.integers(0, 2, size=D, dtype=np.int8)
            for _ in range(self.n_patches)
        ])

    # ─── Image -> arrays ───────────────────────────────────────

    def _to_rgb_array(self, image_path) -> np.ndarray:
        """Load image as (image_size, image_size, 3) uint8."""
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        return np.asarray(img, dtype=np.float32)

    def _extract_patches(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: (H, W, 3) -> (n_patches, patch_size, patch_size, 3)"""
        ps = self.patch_size
        patches = []
        for r in range(self.n_per_side):
            for c in range(self.n_per_side):
                patches.append(rgb[r*ps:(r+1)*ps, c*ps:(c+1)*ps, :])
        return np.stack(patches)

    # ─── Codebook construction (K-means, CPU) ──────────────────

    def build_codebook(self, training_paths: list,
                            sample_patches: int = 50_000,
                            verbose: bool = True) -> None:
        if verbose:
            print(f"[codec] sampling patches from {len(training_paths)} images "
                    f"(target {sample_patches:,}) ...", flush=True)
        all_patches: list[np.ndarray] = []
        per_image_budget = max(8, sample_patches // max(1, len(training_paths)))
        for i, p in enumerate(training_paths):
            try:
                rgb = self._to_rgb_array(p)
                patches = self._extract_patches(rgb)
                # Take every patch up to budget — they're already small
                take = min(per_image_budget, len(patches))
                idx = self.rng.choice(len(patches), size=take, replace=False)
                all_patches.extend(patches[idx])
            except Exception as e:
                if verbose:
                    print(f"[codec] skip {p}: {e}")
            if verbose and (i + 1) % 200 == 0:
                print(f"[codec]   sampled from {i+1}/{len(training_paths)} "
                        f"({len(all_patches):,} patches so far)", flush=True)
        if not all_patches:
            raise RuntimeError("No patches sampled — bad image paths?")
        all_patches = np.stack(all_patches)
        if len(all_patches) > sample_patches:
            idx = self.rng.choice(len(all_patches), size=sample_patches,
                                       replace=False)
            all_patches = all_patches[idx]
        if verbose:
            print(f"[codec] clustering {len(all_patches):,} patches into "
                    f"{self.codebook_size} atoms ...", flush=True)
        from sklearn.cluster import MiniBatchKMeans
        flat = all_patches.reshape(len(all_patches), -1)
        km = MiniBatchKMeans(
            n_clusters=self.codebook_size,
            random_state=42,
            batch_size=512,
            n_init=3,
            max_iter=100,
            verbose=0,
        )
        km.fit(flat)
        self.codebook_pixels = km.cluster_centers_.reshape(
            self.codebook_size, self.patch_size, self.patch_size, 3
        )
        # Random hypervector per atom
        self.codebook_hvs = np.stack([
            self.rng.integers(0, 2, size=D, dtype=np.int8)
            for _ in range(self.codebook_size)
        ])
        if verbose:
            print(f"[codec] codebook ready: {self.codebook_size} RGB atoms",
                    flush=True)

    # ─── Encode / decode ───────────────────────────────────────

    def _nearest_atom_idx(self, patch: np.ndarray) -> int:
        flat = patch.flatten()
        dists = np.sum(
            (self.codebook_pixels.reshape(self.codebook_size, -1) - flat) ** 2,
            axis=1
        )
        return int(np.argmin(dists))

    def encode_array(self, rgb: np.ndarray) -> np.ndarray:
        """RGB array -> hypervector."""
        if self.codebook_pixels is None:
            raise RuntimeError("Codebook not built.")
        patches = self._extract_patches(rgb)
        positional_atoms = []
        for i, patch in enumerate(patches):
            atom_idx = self._nearest_atom_idx(patch)
            atom_hv = self.codebook_hvs[atom_idx]
            positional_atoms.append(
                np.bitwise_xor(self.position_roles[i], atom_hv)
            )
        return bundle(positional_atoms)

    def encode(self, image_path) -> np.ndarray:
        return self.encode_array(self._to_rgb_array(image_path))

    def decode(self, image_hv: np.ndarray) -> np.ndarray:
        """Hypervector -> reconstructed RGB image (H, W, 3) uint8."""
        recon = np.zeros((self.image_size, self.image_size, 3),
                            dtype=np.float32)
        ps = self.patch_size
        for i in range(self.n_patches):
            noisy_atom = np.bitwise_xor(image_hv, self.position_roles[i])
            xor = np.bitwise_xor(self.codebook_hvs, noisy_atom[None, :])
            dists = xor.sum(axis=1)
            atom_idx = int(np.argmin(dists))
            r = i // self.n_per_side
            c = i %  self.n_per_side
            recon[r*ps:(r+1)*ps, c*ps:(c+1)*ps, :] = self.codebook_pixels[atom_idx]
        return np.clip(recon, 0, 255).astype(np.uint8)

    # ─── Round-trip helper ─────────────────────────────────────

    def reconstruct(self, image_path, out_path) -> dict:
        rgb = self._to_rgb_array(image_path)
        hv = self.encode_array(rgb)
        recon = self.decode(hv)
        Image.fromarray(recon, mode="RGB").save(out_path)
        mse = float(np.mean((rgb - recon.astype(np.float32)) ** 2))
        psnr = float(10 * np.log10(255**2 / mse)) if mse > 0 else float("inf")
        return {
            "input":  str(image_path),
            "output": str(out_path),
            "hv_bytes": D // 8,
            "mse":    mse,
            "psnr":   psnr,
        }

    # ─── Persistence ───────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "image_size":     self.image_size,
                "patch_size":     self.patch_size,
                "codebook_size":  self.codebook_size,
                "codebook_pixels": self.codebook_pixels,
                "codebook_hvs":   self.codebook_hvs,
                "position_roles": self.position_roles,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "HDCImageCodecRGB":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        codec = cls(
            image_size=blob["image_size"],
            patch_size=blob["patch_size"],
            codebook_size=blob["codebook_size"],
        )
        codec.codebook_pixels = blob["codebook_pixels"]
        codec.codebook_hvs   = blob["codebook_hvs"]
        codec.position_roles = blob["position_roles"]
        return codec
