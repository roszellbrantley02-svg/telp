"""
lattice/test_feedforward.py - does an HDC feed-forward layer generalize?

The canonical test for "can a layer store factual knowledge and apply
it to held-out inputs": country -> capital.

We train an HDCFeedForward on (country, capital) pairs and test whether
it can return Germany's capital after seeing only USA/France/Japan/etc.

If it generalizes: the bundled XOR keys captured a geometric relation
that holds across nations, and HDC can do what feed-forward layers do.

If it doesn't generalize: the relation isn't consistent enough in HD
space, and we need a different architecture (learned encodings,
multiple keys per layer, etc.).

Either outcome is real research data.

Usage:
    python -m lattice.test_feedforward
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.feedforward import HDCFeedForward
from lattice.text_encoder import TextEncoder
from train.v5_hdc_prototype import hamming_distance, D


COUNTRY_CAPITAL = [
    ("USA",         "Washington"),
    ("France",      "Paris"),
    ("Japan",       "Tokyo"),
    ("United Kingdom", "London"),
    ("Italy",       "Rome"),
    ("Spain",       "Madrid"),
    ("China",       "Beijing"),
    ("Russia",      "Moscow"),
    ("Brazil",      "Brasilia"),
    ("Canada",      "Ottawa"),
    ("Australia",   "Canberra"),
    ("Mexico",      "Mexico City"),
    ("Argentina",   "Buenos Aires"),
    ("South Korea", "Seoul"),
    ("Egypt",       "Cairo"),
    ("Greece",      "Athens"),
    ("Sweden",      "Stockholm"),
    ("Poland",      "Warsaw"),
    ("Turkey",      "Ankara"),
    ("Netherlands", "Amsterdam"),
]

HELDOUT = [
    ("Germany",     "Berlin"),
    ("India",       "Delhi"),
    ("Thailand",    "Bangkok"),
    ("Portugal",    "Lisbon"),
    ("Norway",      "Oslo"),
]


# Distractors — other cities not in the training set's capital list
DISTRACTORS = [
    "Sydney", "Toronto", "Hamburg", "Mumbai", "Marseille",
    "Munich", "Barcelona", "Glasgow", "Naples", "Chicago",
]


def main():
    print("HDC feed-forward layer — country -> capital generalization test\n")
    print("Loading encoder ...")
    enc = TextEncoder()

    # Encode all entities
    print("Encoding training set + heldout + distractors ...")
    train_pairs = []
    for c, cap in COUNTRY_CAPITAL:
        c_hv  = enc.encode(c)
        cap_hv = enc.encode(cap)
        train_pairs.append((c, c_hv, cap, cap_hv))

    heldout_pairs = [(c, enc.encode(c), cap, enc.encode(cap))
                       for c, cap in HELDOUT]

    distractor_hvs = {d: enc.encode(d) for d in DISTRACTORS}

    # ── Train the HDC feed-forward layer ──
    print(f"\nTraining on {len(train_pairs)} (country, capital) pairs ...")
    ff = HDCFeedForward()

    # Add ALL capitals (train + heldout + distractors) to cleanup memory.
    # The layer's job is to pick the right one. If it could only pick
    # from training capitals, the test would be unfair (heldout target
    # not even in the option set).
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
    print(f"  cleanup memory: {len(ff.cleanup_items)} candidate outputs")
    print(f"    (train capitals + heldout capitals + distractors)")
    print(f"  transform_key learned from {len(train_pairs)} XORs")

    # ── Test 1: training-set recall ──
    print("\n=== TEST 1: training-set recall ===")
    print("  Each known country should return its known capital.")
    train_correct = 0
    for c, c_hv, cap, _ in train_pairs:
        results = ff.forward(c_hv, top_k=3)
        top = results[0]
        ok = (top[0] == cap)
        train_correct += int(ok)
        mark = "OK" if ok else "MISS"
        d = top[1]
        print(f"  [{mark}] {c:18s} -> {top[0]:18s} (d={d}, "
              f"expected {cap})")
    print(f"\n  Training accuracy: {train_correct}/{len(train_pairs)} "
          f"= {train_correct/len(train_pairs):.0%}")

    # ── Test 2: generalization to heldout ──
    print("\n=== TEST 2: HELDOUT generalization ===")
    print("  Countries NOT in training. Layer must use learned XOR")
    print("  transformation to predict their capitals.")
    print()
    print(f"  {'COUNTRY':18s} | top-1 prediction         | rank of correct | direct d to correct")
    print(f"  {'-' * 18} | {'-' * 24} | {'-' * 15} | {'-' * 19}")

    heldout_top1 = 0
    heldout_top3 = 0
    for c, c_hv, cap_correct, cap_correct_hv in heldout_pairs:
        # Top-5 to see ranking
        results = ff.forward(c_hv, top_k=20)
        names = [r[0] for r in results]
        top1 = results[0][0]
        rank = names.index(cap_correct) + 1 if cap_correct in names else "out"
        # Also report: how close is the transformed vector to the correct
        # capital's vector directly (raw, before cleanup)?
        raw_xor = ff.forward_raw(c_hv)
        direct_d = hamming_distance(raw_xor, cap_correct_hv)

        is_top1 = (top1 == cap_correct)
        is_top3 = isinstance(rank, int) and rank <= 3
        heldout_top1 += int(is_top1)
        heldout_top3 += int(is_top3)
        mark = "T1" if is_top1 else ("T3" if is_top3 else "  ")
        print(f"  {c:18s} | [{mark}] {top1:18s} | {str(rank):>15s} | "
              f"d={direct_d}  ({direct_d/D*100:.1f}% of D)")

    n = len(heldout_pairs)
    print(f"\n  HELDOUT Top-1 accuracy: {heldout_top1}/{n} = {heldout_top1/n:.0%}")
    print(f"  HELDOUT Top-3 accuracy: {heldout_top3}/{n} = {heldout_top3/n:.0%}")

    # ── Test 3: random-pair sanity check ──
    print("\n=== TEST 3: sanity (untrained pairs should NOT work) ===")
    print("  Predict capital for ENCODER-LEVEL untrained country word.")
    print("  Tests that improvement on heldout isn't just nearest-neighbor")
    print("  in the encoder's embedding space.")
    fake_countries = ["Lemonade", "Telescope", "Banana", "Justice"]
    for fc in fake_countries:
        fc_hv = enc.encode(fc)
        results = ff.forward(fc_hv, top_k=1)
        print(f"  {fc:15s} -> {results[0][0]} (d={results[0][1]})")

    # ── Summary verdict ──
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if heldout_top1 / n >= 0.40:
        print("  STRONG: HDC feed-forward layer generalizes country -> capital")
        print("  >= 40% top-1 on heldout pairs is real signal.")
    elif heldout_top3 / n >= 0.60:
        print("  PARTIAL: top-3 captures the relation, top-1 noisy")
        print("  The layer learned SOMETHING but cleanup is imperfect.")
    elif heldout_top1 / n > 0.20 or heldout_top3 / n > 0.40:
        print("  WEAK: above chance but not reliable")
        print("  The Hebbian-bundle key picks up partial geometry but")
        print("  doesn't fully transfer across countries.")
    else:
        print("  FAIL: no meaningful generalization")
        print("  The country-capital relation isn't consistent enough in")
        print("  LSH-binarized space. Need different architecture or encoder.")


if __name__ == "__main__":
    main()
