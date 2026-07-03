"""
lattice/test_369_capacity.py - capacity stress test for 3-6-9 dimensions.

Push the bundle capacity at each D value. Find the point where retrieval
breaks. Compare Tesla-aligned vs other D groups.

If 3-6-9 dimensions hold more facts per hypervector (or break more
gracefully), that would be a real empirical advantage.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))


# Same dimension groups
DIM_GROUPS = {
    "Tesla-aligned": [369, 999, 3690, 6939, 9990],
    "Random":        [4097, 5111, 7777, 10003, 11111],
    "Primes":        [4099, 9973, 10007, 11003],
    "Powers of 2":   [4096, 8192, 16384],
}

N_FACTS_TO_TEST = [25, 50, 100, 200, 400, 800]


def stress_test(dim: int, n_facts: int, seed: int = 42) -> float:
    """Bundle N (subject, relation, object) triples into one hypervector.
    Return retrieval accuracy."""
    rng = np.random.default_rng(seed)
    relation = rng.integers(0, 2, size=dim, dtype=np.int8)
    subjects = np.stack([rng.integers(0, 2, size=dim, dtype=np.int8)
                            for _ in range(n_facts)])
    objects  = np.stack([rng.integers(0, 2, size=dim, dtype=np.int8)
                            for _ in range(n_facts)])
    # Each fact: bind(bind(R, S), O)
    facts = np.bitwise_xor(np.bitwise_xor(relation[None, :], subjects), objects)
    # Bundle (majority vote)
    threshold = n_facts / 2
    knowledge = (facts.sum(axis=0) > threshold).astype(np.int8)

    # Test: for each (S, R), can we recover O?
    correct = 0
    for i in range(n_facts):
        # query = knowledge XOR R XOR S
        query = np.bitwise_xor(np.bitwise_xor(knowledge, relation), subjects[i])
        # Find nearest object
        xor = np.bitwise_xor(objects, query[None, :])
        dists = xor.sum(axis=1)
        if np.argmin(dists) == i:
            correct += 1
    return correct / n_facts


def main():
    print("=" * 78)
    print("HDC CAPACITY STRESS TEST — does 3-6-9 give more facts per vector?")
    print("=" * 78)

    # Build a giant results table
    all_dims = []
    for group, dims in DIM_GROUPS.items():
        for d in dims:
            all_dims.append((group, d))

    # Header
    n_str = " ".join(f"{n:>4d}" for n in N_FACTS_TO_TEST)
    print(f"\n  {'group':>15s}  {'dim':>6s}  | facts: {n_str}")
    print(f"  {'-' * 15}  {'-' * 6}  | " + "-" * (5 * len(N_FACTS_TO_TEST)))

    results: dict[str, dict[int, list[float]]] = {}   # {group: {n: [accs]}}
    for group, dim in all_dims:
        row = []
        results.setdefault(group, {})
        for n in N_FACTS_TO_TEST:
            acc = stress_test(dim, n)
            row.append(acc)
            results[group].setdefault(n, []).append(acc)
        row_str = " ".join(f"{a*100:>3.0f}%" for a in row)
        print(f"  {group:>15s}  {dim:>6d}  |        {row_str}")

    # ── Group averages ──
    print("\n" + "=" * 78)
    print("GROUP AVERAGES — does any group break later than others?")
    print("=" * 78)
    print(f"\n  {'group':>15s}  | facts:  {n_str}")
    print(f"  {'-'*15}  | " + "-" * (5 * len(N_FACTS_TO_TEST) + 7))
    for group, by_n in results.items():
        avgs = [np.mean(by_n[n]) for n in N_FACTS_TO_TEST]
        row = " ".join(f"{a*100:>3.0f}%" for a in avgs)
        print(f"  {group:>15s}  |        {row}")

    # ── Capacity / D ratio analysis ──
    print("\n" + "=" * 78)
    print("CAPACITY-PER-DIMENSION ratio (where retrieval drops below 95%)")
    print("=" * 78)
    print(f"  Higher number = more efficient packing\n")
    for group, dims in DIM_GROUPS.items():
        ratios = []
        for d in dims:
            # Find the highest N where acc >= 0.95
            best_n = 0
            for n in N_FACTS_TO_TEST:
                if stress_test(d, n) >= 0.95:
                    best_n = n
            ratio = best_n / d
            ratios.append(ratio)
            print(f"  {group:>15s}  D={d:>6d}  capacity={best_n:>4d}  ratio={ratio*1000:>5.2f} per 1000 dims")
        if ratios:
            print(f"  {'GROUP AVG':>15s}                            ratio={np.mean(ratios)*1000:>5.2f}")
            print()

    # ── Verdict ──
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    # Compare Tesla-aligned mean ratio vs others
    print("\n  If 3-6-9 had a real effect, Tesla-aligned dims would show")
    print("  higher capacity-per-dim than random/primes/powers-of-2.")
    print("  Compare the GROUP AVG ratios above.")


if __name__ == "__main__":
    main()
