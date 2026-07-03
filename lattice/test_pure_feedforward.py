"""
lattice/test_pure_feedforward.py - country -> capital with NO LLM.

The fair version of test_feedforward.py. Uses PureHDCTextEncoder
(no MiniLM, no LLM) and re-runs the country -> capital generalization
test.

Honest prediction: heldout generalization will be MUCH worse than the
80% we saw with MiniLM, because the Hebbian-bundled XOR key only
generalizes when the input-output relation has consistent geometric
structure in the encoding space. With pure lexical encoding, the
country-capital relation is essentially arbitrary.

If pure HDC scores < 20% on heldout but > 80% on training, we've
confirmed HDC's earlier wins were riding on the LLM encoder. If it
scores higher, HDC has more capacity than we thought.

Usage:
    python -m lattice.test_pure_feedforward
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.feedforward import HDCFeedForward
from lattice.pure_encoder import PureHDCTextEncoder
from train.v5_hdc_prototype import hamming_distance, D
from lattice.test_feedforward import COUNTRY_CAPITAL, HELDOUT, DISTRACTORS


def main():
    print("HDC feed-forward with PURE encoder (no LLM)\n")
    print("This is the honest test — does HDC generalize on its own?\n")

    enc = PureHDCTextEncoder()

    train_pairs = []
    for c, cap in COUNTRY_CAPITAL:
        train_pairs.append((c, enc.encode(c), cap, enc.encode(cap)))

    heldout_pairs = [(c, enc.encode(c), cap, enc.encode(cap))
                       for c, cap in HELDOUT]

    distractor_hvs = {d: enc.encode(d) for d in DISTRACTORS}

    print(f"Training on {len(train_pairs)} (country, capital) pairs ...")
    ff = HDCFeedForward()

    all_capitals_hv = {}
    for _c, _, cap, cap_hv in train_pairs:
        all_capitals_hv[cap] = cap_hv
    for _c, _, cap, cap_hv in heldout_pairs:
        all_capitals_hv[cap] = cap_hv
    for d, d_hv in distractor_hvs.items():
        all_capitals_hv[d] = d_hv
    for name, hv in all_capitals_hv.items():
        ff.add_cleanup_item(name, hv)

    ff.train(train_pairs)
    print(f"  cleanup memory: {len(ff.cleanup_items)} candidate outputs\n")

    # ── Training recall ──
    print("=== TRAINING-SET RECALL ===")
    train_correct = 0
    for c, c_hv, cap, _ in train_pairs:
        results = ff.forward(c_hv, top_k=1)
        ok = (results[0][0] == cap)
        train_correct += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"  [{mark}] {c:18s} -> {results[0][0]:18s} (expected {cap})")
    print(f"\n  Training accuracy: {train_correct}/{len(train_pairs)} "
          f"= {train_correct/len(train_pairs):.0%}")

    # ── Heldout generalization ──
    print("\n=== HELDOUT GENERALIZATION (THE HONEST TEST) ===")
    print(f"  {'COUNTRY':18s} | top-1            | rank of correct")
    print(f"  {'-' * 18} | {'-' * 16} | {'-' * 15}")
    heldout_top1 = 0
    heldout_top3 = 0
    for c, c_hv, cap, _ in heldout_pairs:
        results = ff.forward(c_hv, top_k=20)
        names = [r[0] for r in results]
        rank = names.index(cap) + 1 if cap in names else "out"
        is_top1 = (results[0][0] == cap)
        is_top3 = isinstance(rank, int) and rank <= 3
        heldout_top1 += int(is_top1)
        heldout_top3 += int(is_top3)
        mark = "T1" if is_top1 else ("T3" if is_top3 else "  ")
        print(f"  {c:18s} | [{mark}] {results[0][0]:14s} | {str(rank):>15s}")
    n = len(heldout_pairs)
    print(f"\n  HELDOUT Top-1: {heldout_top1}/{n} = {heldout_top1/n:.0%}")
    print(f"  HELDOUT Top-3: {heldout_top3}/{n} = {heldout_top3/n:.0%}")

    # ── Compare with MiniLM result ──
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  MiniLM encoder (LLM-derived):  heldout Top-1 = 80%")
    print(f"  Pure HDC encoder (no LLM):     heldout Top-1 = {heldout_top1/n:.0%}")
    print()
    delta = heldout_top1 / n - 0.80
    if delta < -0.40:
        print("  CONFIRMED: HDC's wins were riding on the LLM encoder.")
        print("  Pure HDC alone doesn't have semantic geometry needed for")
        print("  country-capital style relational generalization.")
    elif delta < -0.10:
        print("  Pure HDC is significantly weaker but still functional.")
        print("  The LLM helps, but HDC has SOME independent capacity.")
    elif delta < 0.10:
        print("  Pure HDC roughly matches the LLM-encoded version.")
        print("  Surprising — HDC may be doing more work than I credited.")
    else:
        print("  Pure HDC beats the LLM-encoded version. Unexpected.")


if __name__ == "__main__":
    main()
