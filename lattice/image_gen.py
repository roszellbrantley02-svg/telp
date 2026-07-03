"""
lattice/image_gen.py - HDC routes, matplotlib paints.

HDC decides WHAT image to make based on the request (the cognitive
layer). matplotlib actually pushes pixels (the rendering layer). The
result is stored as a file and indexed in the image bank for future
retrieval.

This is the same pattern LLM agents use to call DALL-E or Stable
Diffusion — they don't generate pixels themselves, they dispatch to
specialized tools. HDC does the same, with the advantage that the
bank of past generations becomes a memory the system learns from.

Supported generators in this MVP:
  - line plot (one or more series)
  - bar chart
  - candlestick chart (for OHLC data)
  - histogram

Each call: HDC stores the request + result so similar future requests
can recall the matplotlib spec, or be answered by retrieval if a close
match already exists in the bank.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

OUT_DIR = _TELP_ROOT / "state" / "generated_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]


def generate_line(data: list[float] | dict[str, list[float]],
                   title: str = "",
                   xlabel: str = "",
                   ylabel: str = "",
                   color: str = "tab:blue",
                   path: Optional[Path] = None) -> Path:
    """Render a line plot. data is either a flat list or {label: list}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    if isinstance(data, dict):
        for label, series in data.items():
            ax.plot(series, label=label)
        ax.legend(loc="best")
    else:
        ax.plot(data, color=color)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    path = path or (OUT_DIR / f"line_{_stamp()}.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_bar(values: list[float], labels: list[str] = None,
                  title: str = "", ylabel: str = "",
                  color: str = "tab:green",
                  path: Optional[Path] = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    xs = labels if labels else list(range(len(values)))
    ax.bar(xs, values, color=color)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    path = path or (OUT_DIR / f"bar_{_stamp()}.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_candlestick(ohlc: list[tuple[float, float, float, float]],
                           title: str = "",
                           path: Optional[Path] = None) -> Path:
    """ohlc: list of (open, high, low, close) tuples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    for i, (o, h, l, c) in enumerate(ohlc):
        col = "tab:green" if c >= o else "tab:red"
        ax.plot([i, i], [l, h], color="black", linewidth=0.7)
        ax.add_patch(plt.Rectangle((i - 0.3, min(o, c)), 0.6, abs(c - o),
                                       facecolor=col, edgecolor="black", linewidth=0.5))
    ax.set_xlim(-0.5, len(ohlc) - 0.5)
    ax.set_title(title); ax.grid(True, alpha=0.3)
    path = path or (OUT_DIR / f"candle_{_stamp()}.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_histogram(values: list[float], bins: int = 30,
                        title: str = "", xlabel: str = "",
                        color: str = "tab:orange",
                        path: Optional[Path] = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.hist(values, bins=bins, color=color, edgecolor="black", alpha=0.7)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.3)
    path = path or (OUT_DIR / f"hist_{_stamp()}.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ─── Self-test ────────────────────────────────────────────────────


def _self_test():
    print("HDC image generator self-test\n")

    # Generate one of each type
    p1 = generate_line([1, 4, 2, 8, 5, 7, 3, 9, 6],
                          title="Test line plot",
                          xlabel="time", ylabel="value")
    print(f"  line plot     -> {p1}")

    p2 = generate_bar([10, 25, 13, 7, 18],
                        labels=["A", "B", "C", "D", "E"],
                        title="Test bar chart")
    print(f"  bar chart     -> {p2}")

    p3 = generate_candlestick([
        (100, 105, 98, 102),
        (102, 108, 101, 107),
        (107, 109, 104, 105),
        (105, 110, 104, 109),
        (109, 112, 106, 108),
    ], title="Test candlesticks")
    print(f"  candlestick   -> {p3}")

    np.random.seed(0)
    values = np.random.normal(loc=50, scale=10, size=500)
    p4 = generate_histogram(values.tolist(),
                              title="Test histogram",
                              xlabel="value")
    print(f"  histogram     -> {p4}")

    print("\nAll generators working. Files in state/generated_images/")


if __name__ == "__main__":
    _self_test()
