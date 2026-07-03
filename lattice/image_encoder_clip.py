"""
lattice/image_encoder_clip.py - CLIP-based semantic image encoder for HDC.

Replaces the raw-pixel encoder. Uses OpenAI's CLIP (already cached
in your HF cache) to produce semantic features, then LSH-binarizes
to a 10k-bit HDC hypervector.

Pipeline:
  PIL image
    -> CLIP vision encoder -> 512-dim semantic feature
    -> random-hyperplane projection to D=10000
    -> binarize

The hypervector captures WHAT'S IN THE IMAGE, not just its pixel
layout. So AAPL daily charts cluster with other AAPL charts because
they share visual content (the ticker label, price ranges), not just
the candlestick layout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from train.v5_hdc_prototype import D, hamming_distance


_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


class CLIPImageEncoder:
    """Image -> CLIP features -> binarize to 10k-bit hypervector."""

    def __init__(self, device: str = "cpu", seed: int = 17):
        self.device = device
        self.seed = seed
        # Lazy-load CLIP — heavy
        self.model = None
        self.processor = None
        self.feature_dim = None
        self.W = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        print(f"[clip_encoder] loading {_CLIP_MODEL_NAME} ...")
        self.model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
        self.model.eval()
        # CLIP base/patch32 emits 512-dim image features
        self.feature_dim = self.model.config.projection_dim   # usually 512
        rng = np.random.default_rng(self.seed)
        self.W = rng.standard_normal((D, self.feature_dim)).astype(np.float32)

    def _to_features(self, image_path: str | Path) -> np.ndarray:
        import torch
        from PIL import Image
        self._ensure_loaded()
        img = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            # Manual projection: vision_model -> pooler_output -> visual_projection
            vis_out = self.model.vision_model(**inputs)
            pooled = vis_out.pooler_output     # (B, hidden_dim)
            feats = self.model.visual_projection(pooled)    # (B, projection_dim)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().squeeze()

    def encode(self, image_path: str | Path) -> np.ndarray:
        feat = self._to_features(image_path)
        projection = self.W @ feat
        return (projection > 0).astype(np.int8)

    def encode_many(self, paths: Iterable[str | Path]) -> np.ndarray:
        import torch
        from PIL import Image
        self._ensure_loaded()
        paths = list(paths)
        # Batch process for speed
        images = [Image.open(p).convert("RGB") for p in paths]
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            vis_out = self.model.vision_model(**inputs)
            pooled = vis_out.pooler_output
            feats = self.model.visual_projection(pooled)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_np = feats.cpu().numpy()
        projections = feats_np @ self.W.T   # (N, D)
        return (projections > 0).astype(np.int8)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text prompt with CLIP's TEXT encoder, into the same HD space.

        Lets you do cross-modal search: "find images that match this text."
        """
        import torch
        self._ensure_loaded()
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
            text_out = self.model.text_model(**inputs)
            pooled = text_out.pooler_output
            feats = self.model.text_projection(pooled)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_np = feats.cpu().numpy().squeeze()
        projection = self.W @ feats_np
        return (projection > 0).astype(np.int8)
