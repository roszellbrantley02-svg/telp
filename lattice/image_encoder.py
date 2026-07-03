"""
lattice/image_encoder.py - encode images into HDC hypervectors.

Same pattern as text_encoder.py: take a high-dimensional feature
vector and binarize via random-hyperplane LSH into a 10k-bit
hypervector that preserves visual similarity as Hamming distance.

Pipeline:
  PIL image
    -> resize to small square (default 64x64)
    -> convert to grayscale
    -> flatten to 4096-dim float vector
    -> normalize
    -> random-hyperplane projection to D=10000
    -> binarize (positive bits become 1)

Visual similarity preserved: images that look alike in pixel space
get small Hamming distance hypervectors.

For semantic similarity (different objects but same "concept"), you'd
want CLIP/SigLIP features instead of raw pixels. We start with the
pixel version because it's dependency-free and demonstrates the
principle. CLIP can be a drop-in replacement at the feature step.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, hamming_distance


class ImageEncoder:
    """Image -> 10k-bit HDC hypervector via raw-pixel LSH."""

    def __init__(self, size: int = 64, seed: int = 17):
        self.size = size
        self.feat_dim = size * size   # grayscale flat
        # Deterministic random hyperplanes
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((D, self.feat_dim)).astype(np.float32)

    def _to_features(self, image_path: str | Path) -> np.ndarray:
        """PIL load -> small grayscale -> flat float vector."""
        from PIL import Image
        img = Image.open(image_path).convert("L")   # grayscale
        img = img.resize((self.size, self.size), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32).flatten()
        # Normalize to [-1, 1] range
        arr = (arr / 127.5) - 1.0
        return arr

    def encode(self, image_path: str | Path) -> np.ndarray:
        feat = self._to_features(image_path)
        projection = self.W @ feat
        return (projection > 0).astype(np.int8)

    def encode_many(self, paths: Iterable[str | Path]) -> np.ndarray:
        feats = np.stack([self._to_features(p) for p in paths])
        projs = feats @ self.W.T
        return (projs > 0).astype(np.int8)


# ─── Smoke test ───────────────────────────────────────────────────


def _sanity():
    print("Image encoder sanity check\n")

    # Generate a couple of synthetic test images
    from PIL import Image, ImageDraw
    import tempfile
    tmp = Path(tempfile.gettempdir())

    # Image 1: black square on white
    im1 = Image.new("RGB", (128, 128), "white")
    d = ImageDraw.Draw(im1)
    d.rectangle([32, 32, 96, 96], fill="black")
    p1 = tmp / "hdc_test_im1.png"
    im1.save(p1)

    # Image 2: same square, slightly shifted
    im2 = Image.new("RGB", (128, 128), "white")
    d = ImageDraw.Draw(im2)
    d.rectangle([40, 40, 104, 104], fill="black")
    p2 = tmp / "hdc_test_im2.png"
    im2.save(p2)

    # Image 3: completely different — black circle
    im3 = Image.new("RGB", (128, 128), "white")
    d = ImageDraw.Draw(im3)
    d.ellipse([20, 20, 108, 108], fill="black")
    p3 = tmp / "hdc_test_im3.png"
    im3.save(p3)

    # Image 4: text/noise
    im4 = Image.new("RGB", (128, 128), "white")
    d = ImageDraw.Draw(im4)
    for x in range(0, 128, 8):
        for y in range(0, 128, 8):
            d.point((x, y), fill="black")
    p4 = tmp / "hdc_test_im4.png"
    im4.save(p4)

    enc = ImageEncoder()
    h1 = enc.encode(p1)
    h2 = enc.encode(p2)
    h3 = enc.encode(p3)
    h4 = enc.encode(p4)

    print(f"  ham(square,            shifted square) = {hamming_distance(h1, h2)}  "
          f"({hamming_distance(h1,h2)/D*100:.1f}%)")
    print(f"  ham(square,            black circle)   = {hamming_distance(h1, h3)}  "
          f"({hamming_distance(h1,h3)/D*100:.1f}%)")
    print(f"  ham(square,            dot grid)       = {hamming_distance(h1, h4)}  "
          f"({hamming_distance(h1,h4)/D*100:.1f}%)")
    print(f"  ham(black circle,      dot grid)       = {hamming_distance(h3, h4)}  "
          f"({hamming_distance(h3,h4)/D*100:.1f}%)")
    print()
    print(f"  Square vs shifted-square should be CLOSEST.")
    closest = "OK" if (hamming_distance(h1, h2) < hamming_distance(h1, h3)
                          and hamming_distance(h1, h2) < hamming_distance(h1, h4)) else "MISS"
    print(f"  Result: [{closest}]")


if __name__ == "__main__":
    _sanity()
