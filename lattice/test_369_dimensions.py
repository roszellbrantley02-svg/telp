"""
lattice/test_369_dimensions.py - does the 3-6-9 hypothesis hold empirically?

Tesla's claim: "if you only knew the magnificence of 3, 6, and 9, you
would have the key to the universe."

Real test: run identical RBI country-capital experiments at many D
values. Compare:
  - "Tesla-aligned" (divisible by 3, 6, 9, or contains 3-6-9 patterns)
  - "Random" (no special relationship to 3-6-9)
  - "Primes" (no factors at all)
  - "Powers of 2" (the standard CS choice)

If 3-6-9 has special power, we should see Tesla-aligned dimensions
outperform the others on retrieval accuracy. If the effect is noise
(my prior), all groups should perform similarly.

This is a real falsifiable test of a mystical claim. Honest result
either way.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))


# Dimension groups to test
DIM_GROUPS = {
    "Tesla-aligned (÷9, ÷3, contains 369)": [
        369, 999, 3690, 6939, 9990, 11997,
    ],
    "Random non-special": [
        4097, 5111, 7777, 10003, 11111,
    ],
    "Primes": [
        4099, 9973, 10007, 11003,
    ],
    "Powers of 2": [
        4096, 8192, 16384,
    ],
    "Pure 3-6-9 patterns": [
        333, 666, 369369, 963,   # if 369 itself matters
    ],
}


def run_rbi_at_dim(dim: int, seed: int = 42) -> dict:
    """Train RBI at given dimension, return heldout accuracy."""
    from lattice.relation_indexing import RelationBoundEncoder
    from lattice.corpus_tagged import build_tagged_corpus
    from lattice.corpus import COUNTRIES
    from train.v5_hdc_prototype import hamming_distance

    # Same protocol as test_relation_indexing
    tagged = build_tagged_corpus()
    enc = RelationBoundEncoder(dim=dim, seed=seed)
    enc.train(tagged)

    train_pairs = [(c[0], c[1]) for c in COUNTRIES[:25]]
    heldout_pairs = [(c[0], c[1]) for c in COUNTRIES[25:]]
    all_capitals = list({c[1] for c in COUNTRIES})

    # Training-set check
    train_correct = 0
    for c, cap in train_pairs:
        results = enc.query(c, "HAS_CAPITAL", all_capitals, top_k=1)
        if results and results[0][0] == cap:
            train_correct += 1

    # Heldout check
    h_top1 = 0
    h_top3 = 0
    for c, cap in heldout_pairs:
        results = enc.query(c, "HAS_CAPITAL", all_capitals, top_k=3)
        names = [r[0] for r in results]
        if names and names[0] == cap:
            h_top1 += 1
        if cap in names[:3]:
            h_top3 += 1

    return {
        "dim": dim,
        "train_acc": train_correct / len(train_pairs),
        "heldout_top1": h_top1 / len(heldout_pairs),
        "heldout_top3": h_top3 / len(heldout_pairs),
    }


def digital_root(n: int) -> int:
    while n > 9:
        n = sum(int(d) for d in str(n))
    return n


def main():
    print("=" * 70)
    print("HDC 3-6-9 DIMENSIONALITY TEST")
    print("=" * 70)
    print("\nRunning RBI country->capital test across many dimensions ...")
    print("Same corpus, same protocol, only D varies.\n")

    all_results = []
    for group, dims in DIM_GROUPS.items():
        print(f"\n=== {group} ===")
        print(f"  {'dim':>8s} | {'digital_root':>13s} | {'train':>7s} | {'h-top1':>7s} | {'h-top3':>7s}")
        print(f"  {'-'*8} | {'-'*13} | {'-'*7} | {'-'*7} | {'-'*7}")
        group_results = []
        for d in dims:
            try:
                r = run_rbi_at_dim(d)
                r["group"] = group
                r["digital_root"] = digital_root(d)
                all_results.append(r)
                group_results.append(r)
                print(f"  {d:>8d} | {r['digital_root']:>13d} | "
                      f"{r['train_acc']*100:>6.0f}% | "
                      f"{r['heldout_top1']*100:>6.0f}% | "
                      f"{r['heldout_top3']*100:>6.0f}%")
            except Exception as e:
                print(f"  {d:>8d} | ERROR: {e}")

        # Group summary
        if group_results:
            avg_top1 = np.mean([r["heldout_top1"] for r in group_results])
            avg_top3 = np.mean([r["heldout_top3"] for r in group_results])
            std_top1 = np.std([r["heldout_top1"] for r in group_results])
            print(f"  GROUP AVG: top-1 = {avg_top1*100:.0f}% (std {std_top1*100:.0f}), "
                  f"top-3 = {avg_top3*100:.0f}%")

    # ── Comparison ──
    print("\n" + "=" * 70)
    print("COMPARISON: average heldout top-1 by group")
    print("=" * 70)
    by_group: dict[str, list[float]] = {}
    for r in all_results:
        by_group.setdefault(r["group"], []).append(r["heldout_top1"])
    for group, accs in by_group.items():
        print(f"  {group:42s}: {np.mean(accs)*100:>5.1f}% "
              f"(±{np.std(accs)*100:.1f}, n={len(accs)})")

    # ── Honest verdict ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    tesla_avg = np.mean(by_group.get("Tesla-aligned (÷9, ÷3, contains 369)", [0]))
    random_avg = np.mean(by_group.get("Random non-special", [0]))
    prime_avg = np.mean(by_group.get("Primes", [0]))
    pow2_avg = np.mean(by_group.get("Powers of 2", [0]))
    diff = tesla_avg - random_avg
    print(f"  Tesla-aligned avg: {tesla_avg*100:.1f}%")
    print(f"  Random avg:        {random_avg*100:.1f}%")
    print(f"  Primes avg:        {prime_avg*100:.1f}%")
    print(f"  Powers of 2 avg:   {pow2_avg*100:.1f}%")
    print(f"  Tesla - Random:    {diff*100:+.1f} percentage points")
    print()
    if abs(diff) < 0.05:
        print("  RESULT: dimensions don't matter measurably. The 3-6-9")
        print("  hypothesis is NOT empirically supported by this test.")
    elif diff > 0.10:
        print("  RESULT: Tesla-aligned dimensions outperformed others.")
        print("  Worth investigating further. Could be real, could be noise.")
    elif diff < -0.10:
        print("  RESULT: Tesla-aligned dimensions UNDER-performed.")
        print("  3-6-9 not magical, possibly anti-magical.")
    else:
        print("  RESULT: small, ambiguous effect.")
    print()
    print("  Note: D below ~3000 may simply be too small for adequate HDC")
    print("  capacity; that's a confound to control for in deeper testing.")


if __name__ == "__main__":
    main()
