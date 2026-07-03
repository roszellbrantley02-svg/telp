"""
lattice/hearing.py - Telp's ears: local speech recognition (Whisper).

For videos with no captions, the eyes alone miss half the story. This organ
transcribes the AUDIO track locally (no cloud, no API key) into timestamped
chunks shaped exactly like YouTube caption chunks, so the same fusion path
binds what was HEARD to what was SEEN.

Model: whisper 'base' by default - fast on the GPU, honest quality; bump to
'small'/'medium' for hard audio. Weights cached on D: (C: is full).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

# whisper shells out to `ffmpeg` - make a local static build reachable even
# when ffmpeg isn't on the system PATH (TELP_FFMPEG_DIR overrides)
import glob as _glob
_cands = ([os.environ["TELP_FFMPEG_DIR"]] if os.environ.get("TELP_FFMPEG_DIR")
          else _glob.glob(r"C:\Users\*\upscaler\bin\ffmpeg\*\bin"))
for _d in _cands:
    if Path(_d).exists() and _d not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
        break

_MODEL_DIR = Path(os.environ.get("TELP_MODEL_DIR")
                  or (r"D:\hf_home" if Path(r"D:\\").exists()
                      else Path.home() / ".cache" / "telp")) / "whisper"
_MODEL = None
_MODEL_SIZE = None


def get_model(size: str = "base"):
    """Process-wide Whisper model (loads once, GPU when available)."""
    global _MODEL, _MODEL_SIZE
    if _MODEL is None or _MODEL_SIZE != size:
        import torch
        import whisper
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[hearing] loading whisper-{size} on {device} ...", flush=True)
        _MODEL = whisper.load_model(size, device=device,
                                    download_root=str(_MODEL_DIR))
        _MODEL_SIZE = size
    return _MODEL


def transcribe(media_path, size: str = "base") -> list[tuple[float, str]]:
    """Listen to a video/audio file -> [(start_seconds, text), ...].
    Same shape as YouTube caption chunks, so fusion code is shared."""
    model = get_model(size)
    result = model.transcribe(str(media_path), fp16=(model.device.type == "cuda"))
    out = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            out.append((float(seg.get("start", 0.0)), text))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Telp's ears - local ASR")
    ap.add_argument("media")
    ap.add_argument("--size", default="base")
    a = ap.parse_args()
    for t, txt in transcribe(a.media, size=a.size):
        print(f"[{t:7.1f}s] {txt}")
