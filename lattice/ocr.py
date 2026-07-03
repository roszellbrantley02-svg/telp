"""
lattice/ocr.py - Telp's reading eyes: text inside images and video frames.

CLIP names what a frame SHOWS; this organ reads what a frame SAYS - titles,
signs, slides, captions burned into the pixels. Local EasyOCR (torch-based,
GPU on the 5090, models cached on D:), no cloud.

read_text(image) -> cleaned text lines, confidence-filtered.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

_MODEL_DIR = Path(os.environ.get("TELP_MODEL_DIR")
                  or (r"D:\hf_home" if Path(r"D:\\").exists()
                      else Path.home() / ".cache" / "telp")) / "easyocr"
_READER = None


def get_reader():
    """Process-wide OCR reader (loads once, GPU when available)."""
    global _READER
    if _READER is None:
        import easyocr
        import torch
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        gpu = torch.cuda.is_available()
        print(f"[ocr] loading EasyOCR (gpu={gpu}) ...", flush=True)
        _READER = easyocr.Reader(
            ["en"], gpu=gpu, model_storage_directory=str(_MODEL_DIR),
            download_enabled=True, verbose=False)
    return _READER


def read_text(image_path, min_conf: float = 0.40) -> list[str]:
    """Text lines visible in an image, cleaned and confidence-filtered."""
    reader = get_reader()
    try:
        results = reader.readtext(str(image_path))
    except Exception:
        return []
    lines = []
    for _bbox, text, conf in results:
        text = re.sub(r"\s+", " ", str(text)).strip()
        if conf >= min_conf and len(text) >= 2 and any(c.isalnum() for c in text):
            lines.append(text)
    return lines


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Telp's reading eyes")
    ap.add_argument("image")
    a = ap.parse_args()
    for line in read_text(a.image):
        print(line)
