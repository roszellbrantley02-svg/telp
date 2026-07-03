"""
lattice/image_vq.py - HDC image codec via shared patch codebook.

The trick: information theory allows reconstruction when sender and
receiver share a codebook. JPEG uses the DCT basis. VQ-VAE learns a
discrete codebook. Here we build a codebook of image patches sampled
from training data, and use HDC to ADDRESS into it.

Pipeline (encode):
  image -> tile into N patches at fixed positions
  for each patch: find nearest codebook atom (cleanup memory)
  image_hv = bundle of bind(position_role_i, atom_hv_for_patch_i)

Pipeline (decode):
  for each position i:
    unbind position_role_i from image_hv -> noisy atom_hv
    cleanup against codebook -> exact atom_hv
    look up the patch pixels for that atom
    paste patch at position i
  return reconstructed image

The per-image storage is just the 10k-bit hypervector (1.25KB). The
codebook (~64KB for 256 patches at 8x8 grayscale) is amortized — load
it once, encode/decode any number of images.

Reconstruction is LOSSY (limited by patch granularity + codebook size)
but it's REAL reconstruction. Visibly the same image. Recognizable.
And the information is genuinely transmitted via the hypervector,
in the same sense JPEG transmits via DCT coefficients.

This is the answer to "can HDC store and reconstruct images" — yes,
in exactly the way modern codecs do, with HDC as the addressing layer.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from PIL import Image

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, bundle, bind, hamming_distance


class HDCImageCodec:
    """VQ-style image codec with HDC as the addressing layer."""

    def __init__(self, image_size: int = 64, patch_size: int = 8,
                  codebook_size: int = 256, seed: int = 42):
        """
        image_size:   target image size (square, e.g. 64 -> 64x64 grayscale)
        patch_size:   each patch is patch_size x patch_size
        codebook_size: number of patches in the shared codebook
        """
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_per_side = image_size // patch_size
        self.n_patches = self.n_per_side ** 2
        self.codebook_size = codebook_size
        self.rng = np.random.default_rng(seed)
        # Codebook: (codebook_size, patch_size*patch_size) pixel arrays
        self.codebook_pixels: np.ndarray | None = None
        # Codebook atoms (hypervectors), one per entry
        self.codebook_hvs: np.ndarray | None = None
        # Position roles, one per patch position
        self.position_roles = np.stack([
            self.rng.integers(0, 2, size=D, dtype=np.int8)
            for _ in range(self.n_patches)
        ])

    # ─── Codebook construction ──────────────────────────────

    def _to_grayscale_array(self, image_path) -> np.ndarray:
        """Load image as image_size x image_size grayscale float32 in [0, 255]."""
        img = Image.open(image_path).convert("L")
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        return np.asarray(img, dtype=np.float32)

    def _extract_patches(self, gray: np.ndarray) -> np.ndarray:
        """gray: (image_size, image_size) -> (n_patches, patch_size, patch_size)"""
        ps = self.patch_size
        patches = []
        for r in range(self.n_per_side):
            for c in range(self.n_per_side):
                patches.append(gray[r*ps:(r+1)*ps, c*ps:(c+1)*ps])
        return np.stack(patches)

    def build_codebook(self, training_paths: list, sample_patches: int = 5000) -> None:
        """Sample patches from training images, cluster (k-means style) into
        codebook_size representative atoms, generate hypervectors."""
        print(f"  sampling {sample_patches} patches from {len(training_paths)} images ...")
        all_patches = []
        for p in training_paths:
            gray = self._to_grayscale_array(p)
            patches = self._extract_patches(gray)
            all_patches.extend(patches)
        all_patches = np.stack(all_patches)
        # Random subsample
        if len(all_patches) > sample_patches:
            idx = self.rng.choice(len(all_patches), size=sample_patches, replace=False)
            all_patches = all_patches[idx]

        # Simple k-means clustering to get codebook_size atoms
        print(f"  clustering {len(all_patches)} patches into {self.codebook_size} atoms ...")
        from sklearn.cluster import MiniBatchKMeans
        flat = all_patches.reshape(len(all_patches), -1)
        km = MiniBatchKMeans(n_clusters=self.codebook_size,
                              random_state=42,
                              batch_size=256,
                              n_init=3,
                              max_iter=50)
        km.fit(flat)
        # codebook_pixels[i] is the centroid of cluster i
        self.codebook_pixels = km.cluster_centers_.reshape(
            self.codebook_size, self.patch_size, self.patch_size
        )
        # Random hypervector per atom
        self.codebook_hvs = np.stack([
            self.rng.integers(0, 2, size=D, dtype=np.int8)
            for _ in range(self.codebook_size)
        ])
        print(f"  codebook ready: {self.codebook_size} atoms")

    def _nearest_atom_idx(self, patch: np.ndarray) -> int:
        """Find the codebook index for a given patch (Euclidean)."""
        flat = patch.flatten()
        dists = np.sum((self.codebook_pixels.reshape(self.codebook_size, -1) - flat) ** 2,
                         axis=1)
        return int(np.argmin(dists))

    # ─── Encode / decode ────────────────────────────────────

    def encode(self, image_path) -> np.ndarray:
        """Image -> 10k-bit hypervector."""
        if self.codebook_pixels is None:
            raise RuntimeError("Codebook not built. Call build_codebook first.")
        gray = self._to_grayscale_array(image_path)
        patches = self._extract_patches(gray)
        # For each position, find nearest atom, bind with position role
        positional_atoms = []
        for i, patch in enumerate(patches):
            atom_idx = self._nearest_atom_idx(patch)
            atom_hv = self.codebook_hvs[atom_idx]
            positional_atoms.append(np.bitwise_xor(self.position_roles[i], atom_hv))
        return bundle(positional_atoms)

    def decode(self, image_hv: np.ndarray) -> np.ndarray:
        """Hypervector -> reconstructed image_size x image_size grayscale."""
        recon = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        ps = self.patch_size
        recovered_indices = []
        for i in range(self.n_patches):
            # Unbind position role
            noisy_atom = np.bitwise_xor(image_hv, self.position_roles[i])
            # Cleanup: nearest codebook atom by Hamming
            xor = np.bitwise_xor(self.codebook_hvs, noisy_atom[None, :])
            dists = xor.sum(axis=1)
            atom_idx = int(np.argmin(dists))
            recovered_indices.append(atom_idx)
            # Paste the atom's pixels into position i
            r = i // self.n_per_side
            c = i %  self.n_per_side
            recon[r*ps:(r+1)*ps, c*ps:(c+1)*ps] = self.codebook_pixels[atom_idx]
        return recon

    def reconstruct(self, image_path, out_path) -> dict:
        """Full round trip: encode, decode, save to disk."""
        gray = self._to_grayscale_array(image_path)
        image_hv = self.encode(image_path)
        recon = self.decode(image_hv)
        recon_clipped = np.clip(recon, 0, 255).astype(np.uint8)
        Image.fromarray(recon_clipped, mode="L").save(out_path)
        # Compute PSNR
        mse = float(np.mean((gray - recon) ** 2))
        psnr = float(10 * np.log10(255**2 / mse)) if mse > 0 else float("inf")
        return {
            "input_path":  str(image_path),
            "output_path": str(out_path),
            "hypervector_bytes": D // 8,
            "mse":         mse,
            "psnr":        psnr,
        }
