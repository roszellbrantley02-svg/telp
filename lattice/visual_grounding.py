"""
lattice/visual_grounding.py - Phase 12.8: words have pictures.

WHY
---
"The star has no meaning without a picture." — user, this session.

Until now Telp had R_IMAGE role declared but nothing bound to it.
Every concrete noun lived as text only.  This module gives him an
actual visual referent per word: a small 32x32 RGB image, encoded
via the existing HDCImageCodecRGB into a 10000-D HV, bound to the
word via the existing semantic-grounding hook.

GENERATION STRATEGIES (per-word)
--------------------------------
For COLOR words (red, blue, green, ...):
  Solid color tile.  red = filled red square.  Visual = literal.

For ANIMAL / OBJECT words:
  Rendered via Unicode emoji where possible (🐻=bear, ⭐=star, 🌧=rain,
  🐭=mouse).  Uses Pillow + a Unicode font that supports the
  Unicode emoji range.  Falls back to a colored geometric "icon"
  if the font can't render the emoji (different shape+color per word
  so each still has a unique visual signature).

For ABSTRACT words (happy, sad, ...):
  Skip — no visual referent.

STORAGE
-------
state/word_images/{word}.png  — generated 32x32 PNG per word
HV is stored in lattice.word_semantics[word]["image"]

QUERIES
-------
words_visually_similar_to(lattice, "red") → other red things
words_visually_similar_to(lattice, "bear") → other animal-shaped icons
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_TELP_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = _TELP_ROOT / "state" / "word_images"


# --- Color tiles --------------------------------------------------


COLOR_RGB = {
    "red":     (220,  30,  30),
    "blue":    ( 30,  60, 220),
    "green":   ( 30, 180,  60),
    "yellow":  (240, 220,  30),
    "white":   (250, 250, 250),
    "black":   ( 10,  10,  10),
    "brown":   (130,  80,  40),
    "purple":  (140,  50, 180),
    "orange":  (245, 140,  20),
    "pink":    (250, 150, 180),
    "gray":    (140, 140, 140),
    "gold":    (220, 180,  20),
}


def _make_color_tile(rgb: tuple[int, int, int],
                          size: int = 32) -> Image.Image:
    img = Image.new("RGB", (size, size), rgb)
    return img


# --- Emoji rendering -----------------------------------------------


WORD_EMOJI = {
    # Animals
    "bear":    "🐻", "cat":     "🐈", "dog":     "🐕",
    "spider":  "🕷",  "lamb":    "🐑", "mouse":   "🐭",
    "fish":    "🐟", "bird":    "🐦", "horse":   "🐴",
    "cow":     "🐄", "sheep":   "🐑", "frog":    "🐸",
    "duck":    "🦆", "rabbit":  "🐇", "fox":     "🦊",
    "owl":     "🦉", "snake":   "🐍", "bee":     "🐝",
    "ant":     "🐜", "pig":     "🐖", "goose":   "🦆",
    # Celestial / nature
    "star":    "⭐", "sun":     "☀",  "moon":    "🌙",
    "sky":     "☁",  "cloud":   "☁",  "rain":    "🌧",
    "snow":    "❄",  "rainbow": "🌈", "fire":    "🔥",
    "tree":    "🌳", "flower":  "🌸", "leaf":    "🍃",
    "mountain":"⛰",  "water":   "💧", "wind":    "💨",
    # Objects / food
    "ball":    "🏀", "house":   "🏠", "car":     "🚗",
    "bus":     "🚌", "train":   "🚂", "plane":   "✈",
    "bike":    "🚲", "boat":    "⛵", "book":    "📖",
    "toy":     "🧸", "cookie":  "🍪", "apple":   "🍎",
    "banana":  "🍌", "orange":  "🍊", "milk":    "🥛",
    "juice":   "🧃", "pizza":   "🍕", "bread":   "🍞",
    "cheese":  "🧀", "egg":     "🥚", "honey":   "🍯",
    "bed":     "🛏",  "chair":   "🪑", "cup":     "🥛",
    "shirt":   "👕", "pants":   "👖", "shoes":   "👟",
    "hat":     "🎩", "coat":    "🧥",
    # Family / people
    "mother":  "👩", "father":  "👨", "baby":    "👶",
    "child":   "🧒", "friend":  "🤗", "brother": "👦",
    "sister":  "👧",
    # Emotion (anchor to canonical emoji)
    "happy":   "😊", "sad":     "😢", "scared":  "😨",
    "angry":   "😠", "excited": "🤩", "tired":   "😴",
    "surprised": "😲", "love":   "❤",
}


def _try_load_emoji_font(size: int = 24) -> Optional[ImageFont.ImageFont]:
    """Try to load a font that supports Unicode emoji."""
    candidate_paths = [
        "C:/Windows/Fonts/seguiemj.ttf",  # Windows color emoji
        "C:/Windows/Fonts/seguisym.ttf",  # Windows symbol fallback
        "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux
        "C:/Windows/Fonts/segoeui.ttf",  # generic Windows
    ]
    for p in candidate_paths:
        if Path(p).exists():
            try:
                # Color emoji fonts may need fixed-size loading
                return ImageFont.truetype(p, size)
            except Exception:
                try:
                    # Try without size for color emoji bitmap fonts
                    return ImageFont.truetype(p)
                except Exception:
                    continue
    return None


def _make_emoji_tile(emoji: str, size: int = 32) -> Image.Image:
    """Render an emoji onto a 32x32 RGB tile."""
    img = Image.new("RGB", (size, size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    font = _try_load_emoji_font(size=size - 8)
    if font is not None:
        try:
            # Center the emoji
            bbox = draw.textbbox((0, 0), emoji, font=font, embedded_color=True)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x = (size - w) // 2 - bbox[0]
            y = (size - h) // 2 - bbox[1]
            draw.text((x, y), emoji, font=font, embedded_color=True)
        except Exception:
            # Some fonts don't accept embedded_color
            try:
                draw.text((4, 4), emoji, font=font, fill=(0, 0, 0))
            except Exception:
                pass
    return img


# --- Fallback geometric icons (each word gets a unique shape+color)


def _make_geometric_icon(word: str, size: int = 32) -> Image.Image:
    """Deterministic geometric icon per word.  Used when no emoji
    is mapped.  Each word gets a unique color + shape from its hash.
    """
    import hashlib
    h = hashlib.sha1(word.lower().encode()).digest()
    color = (50 + h[0] % 200, 50 + h[1] % 200, 50 + h[2] % 200)
    bg = (250, 250, 250)
    shape_idx = h[3] % 4
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    pad = 4
    if shape_idx == 0:    # circle
        draw.ellipse([pad, pad, size - pad, size - pad], fill=color)
    elif shape_idx == 1:  # square
        draw.rectangle([pad, pad, size - pad, size - pad], fill=color)
    elif shape_idx == 2:  # triangle
        draw.polygon([(size // 2, pad),
                          (pad, size - pad),
                          (size - pad, size - pad)], fill=color)
    else:                  # diamond
        draw.polygon([(size // 2, pad),
                          (size - pad, size // 2),
                          (size // 2, size - pad),
                          (pad, size // 2)], fill=color)
    return img


# --- Per-word image dispatch --------------------------------------


def make_word_image(word: str, size: int = 32) -> Image.Image:
    """Pick the right generator for the word and return an Image."""
    w = word.lower()
    if w in COLOR_RGB:
        return _make_color_tile(COLOR_RGB[w], size=size)
    if w in WORD_EMOJI:
        return _make_emoji_tile(WORD_EMOJI[w], size=size)
    return _make_geometric_icon(w, size=size)


def save_word_image(word: str, size: int = 32) -> Path:
    """Generate and save the image for a word, return the path."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    img = make_word_image(word, size=size)
    path = IMAGE_DIR / f"{word.lower()}.png"
    img.save(path)
    return path


# --- HV encoding ---------------------------------------------------


_CODEC = None


def _codec():
    """Lazy-load the HDCImageCodecRGB. Prefer the chart codec
    (already trained), fall back to the generic CIFAR codec."""
    global _CODEC
    if _CODEC is not None:
        return _CODEC
    from lattice.image_vq_rgb import HDCImageCodecRGB
    # CIFAR codec FIRST — trained on 32x32 natural images, much closer
    # to our 32x32 emoji/icon tiles than the chart-panel codec.
    for path in [
        _TELP_ROOT / "state" / "hdc_image_codec.pkl",
        _TELP_ROOT / "state" / "hdc_chart_codec.pkl",
    ]:
        if path.exists():
            try:
                _CODEC = HDCImageCodecRGB.load(str(path))
                print(f"[visual] loaded codec from {path.name}",
                        file=sys.stderr)
                return _CODEC
            except Exception:
                continue
    raise FileNotFoundError("No HDC image codec available")


def image_hv_for(word: str) -> Optional[np.ndarray]:
    """Generate the word's visual referent and encode it as an HV."""
    try:
        path = save_word_image(word)
        codec = _codec()
        return codec.encode(str(path))
    except Exception as e:
        print(f"[visual] failed for {word!r}: {e}", file=sys.stderr)
        return None


# --- Lattice grounding -------------------------------------------


def ground_visual_in_lattice(lattice, verbose: bool = True) -> dict:
    """Attach image HVs to every word that has either a color or
    an emoji mapping.  Other words get a geometric-icon fallback."""
    n_grounded = 0
    n_failed = 0
    for word in list(lattice.word_mem.labels()):
        # Only ground concrete nouns / colors / emotion anchors —
        # function words (the, a, is) don't get pictures
        w = word.lower()
        if w in COLOR_RGB or w in WORD_EMOJI:
            hv = image_hv_for(word)
            if hv is not None:
                lattice.bind_semantic(word, "image", hv)
                n_grounded += 1
            else:
                n_failed += 1
        if verbose and n_grounded % 25 == 0 and n_grounded > 0:
            print(f"[visual] grounded {n_grounded} so far", file=sys.stderr)
    stats = {
        "n_grounded": n_grounded,
        "n_failed":   n_failed,
        "n_total":    len(lattice.word_mem),
    }
    if verbose:
        print(f"[visual] DONE: {stats}", file=sys.stderr)
    return stats


def words_visually_similar_to(lattice, word: str,
                                            top_k: int = 8) -> list[tuple[str, float]]:
    """Find words with similar image HVs."""
    from lattice.phoneme_hdc import similarity
    target = lattice.word_semantics.get(word, {}).get("image")
    if target is None:
        return []
    scored = []
    for w in lattice.word_mem.labels():
        if w == word:
            continue
        h = lattice.word_semantics.get(w, {}).get("image")
        if h is None:
            continue
        scored.append((w, similarity(target, h)))
    scored.sort(key=lambda kv: -kv[1])
    return scored[:top_k]


# --- CLI smoke test ----------------------------------------------


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--word", default=None,
                       help="Generate + show image for one word")
    p.add_argument("--ground", action="store_true",
                       help="Ground every word in the saved lattice")
    args = p.parse_args()

    if args.word:
        path = save_word_image(args.word)
        print(f"saved {path}")
        hv = image_hv_for(args.word)
        if hv is not None:
            print(f"HV shape={hv.shape} sum={hv.sum()}")
        sys.exit(0)

    if args.ground:
        from lattice.reading_lattice import ReadingLattice
        lattice = ReadingLattice.load()
        stats = ground_visual_in_lattice(lattice)
        lattice.save()
        print(f"grounded {stats['n_grounded']} of {stats['n_total']} words")

        # Show a few visual neighborhoods
        print("\n=== Visual neighborhoods ===")
        for w in ["red", "blue", "bear", "cat", "star", "ball", "happy", "sad"]:
            nbrs = words_visually_similar_to(lattice, w, top_k=5)
            if nbrs:
                ns = ", ".join(f"{n}({s:.2f})" for n, s in nbrs)
                print(f"  {w:8} -> {ns}")
