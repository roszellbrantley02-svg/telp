"""
lattice/test_analogy.py - Word2Vec-style analogy via HDC algebra.

The famous test:
  king - man + woman = queen

In HDC, the equivalent for our domain:
  Germany - Berlin + Paris = France
  or equivalently:
  France XOR Paris XOR Berlin = Germany

The deep question: does HDC have STRUCTURED DIRECTIONS the way
Word2Vec embeddings do? If yes, the "country-to-capital" delta
vector should be CONSISTENT across many pairs. Then we can solve
analogies for held-out pairs by averaging the deltas.

Three tests:

  T1: direction consistency
    For known pairs, are the XOR-differences (D1, D2, ...) similar
    to each other? Measured by pairwise Hamming distance within the
    set, vs random baseline.

  T2: one-shot analogy
    Given a single example pair as the "anchor" (e.g., France/Paris),
    can we solve analogies for OTHER countries?
    transform = France XOR Paris
    Germany XOR transform -> ?  (should be Berlin)

  T3: many-shot analogy
    Average across N example pairs to get a robust transform,
    then test on held-out countries.

Substrate: we use RBI's learned context vectors. They captured
country-shape and capital-shape correctly in earlier tests; the
question is whether the XOR-difference between them encodes a
consistent direction.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.relation_indexing import RelationBoundEncoder
from lattice.corpus_tagged import build_tagged_corpus
from lattice.corpus import COUNTRIES
from train.v5_hdc_prototype import D, hamming_distance


KNOWN_PAIRS  = [(c[0], c[1]) for c in COUNTRIES[:20]]
HELDOUT_PAIRS = [(c[0], c[1]) for c in COUNTRIES[20:]]


def main():
    print("HDC Analogy Algebra — does HDC have Word2Vec-style structured directions?\n")

    # Train RBI on the full corpus
    print("Training RBI on tagged corpus ...")
    enc = RelationBoundEncoder(dim=D, seed=42)
    enc.train(build_tagged_corpus())
    print(f"  vocab={enc.vocab_size() if hasattr(enc, 'vocab_size') else len(enc.index_vectors)}")

    def vec(word: str) -> np.ndarray:
        """Use the RBI context vector for a word."""
        return enc.get_context(word)

    # ── T1: are pair-differences consistent across known pairs? ──
    print("\n=== TEST 1: direction consistency ===")
    print("  For each known pair, compute D = country XOR capital.")
    print("  Pairwise Hamming distance among the D's measures consistency.\n")
    deltas = []
    for country, capital in KNOWN_PAIRS:
        c = vec(country)
        cap = vec(capital)
        if c.sum() == 0 or cap.sum() == 0:
            continue
        d = np.bitwise_xor(c, cap)
        deltas.append((f"{country}->{capital}", d))
    print(f"  Collected {len(deltas)} pair-deltas.")

    if len(deltas) >= 2:
        intra_distances = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                d = hamming_distance(deltas[i][1], deltas[j][1])
                intra_distances.append(d)
        mean_intra = np.mean(intra_distances) / D
        random_baseline = 0.5
        print(f"  Mean pairwise Hamming (country-capital deltas): "
              f"{mean_intra*100:.1f}% of D")
        print(f"  Random baseline:                                "
              f"{random_baseline*100:.1f}%")
        consistency = 1.0 - (mean_intra / random_baseline)
        print(f"  Direction consistency: {consistency*100:.1f}%  "
              f"(higher = more consistent direction)")
        if mean_intra < 0.45:
            print(f"  STRONG: deltas are clearly more consistent than random.")
        elif mean_intra < 0.48:
            print(f"  MODEST: some consistency but weak.")
        else:
            print(f"  NONE: deltas are essentially random.")

    # ── T2: one-shot analogy ──
    print("\n=== TEST 2: one-shot analogy from a single anchor pair ===")
    print("  Use the FIRST known pair as anchor. Predict capitals of all others.\n")
    anchor_country, anchor_capital = KNOWN_PAIRS[0]
    transform = np.bitwise_xor(vec(anchor_country), vec(anchor_capital))
    print(f"  Anchor: {anchor_country} -> {anchor_capital}")
    print(f"  Transform = XOR(anchor_country, anchor_capital)\n")

    # All capitals as cleanup space
    all_capitals = list({cap for _, cap in KNOWN_PAIRS + HELDOUT_PAIRS})
    cap_vecs = {cap: vec(cap) for cap in all_capitals}

    correct = 0
    tested = 0
    for country, expected in KNOWN_PAIRS[1:] + HELDOUT_PAIRS:
        c_vec = vec(country)
        if c_vec.sum() == 0:
            continue
        predicted_vec = np.bitwise_xor(c_vec, transform)
        # Cleanup
        best_name, best_d = None, None
        for cap_name, cap_v in cap_vecs.items():
            if cap_v.sum() == 0:
                continue
            d = hamming_distance(predicted_vec, cap_v)
            if best_d is None or d < best_d:
                best_d = d
                best_name = cap_name
        ok = (best_name == expected)
        correct += int(ok)
        tested += 1
    print(f"  One-shot analogy accuracy: {correct}/{tested} = "
          f"{correct/max(1, tested):.0%}")

    # ── T3: many-shot analogy (averaged transform) ──
    print("\n=== TEST 3: many-shot analogy (average transform from 19 pairs) ===")
    print("  Average the XOR-deltas across all KNOWN pairs.")
    print("  Apply to held-out countries.\n")

    # Compute majority-vote average of deltas
    delta_stack = np.stack([d for _, d in deltas])
    threshold = delta_stack.shape[0] / 2
    avg_transform = (delta_stack.sum(axis=0) > threshold).astype(np.int8)

    correct = 0
    print(f"  {'COUNTRY':14s} | top-1 prediction  | rank of correct")
    print(f"  {'-' * 14} | {'-' * 17} | {'-' * 15}")
    for country, expected in HELDOUT_PAIRS:
        c_vec = vec(country)
        if c_vec.sum() == 0:
            print(f"  {country:14s} | (no RBI vector)   | n/a")
            continue
        predicted_vec = np.bitwise_xor(c_vec, avg_transform)
        # Cleanup: rank all capitals
        scored = sorted(
            [(cap, hamming_distance(predicted_vec, cv))
              for cap, cv in cap_vecs.items() if cv.sum() > 0],
            key=lambda x: x[1]
        )
        top1 = scored[0][0]
        names = [s[0] for s in scored]
        rank = names.index(expected) + 1 if expected in names else "out"
        ok = (top1 == expected)
        correct += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"  {country:14s} | [{mark}] {top1:14s} | {str(rank):>15s}")
    n = len(HELDOUT_PAIRS)
    print(f"\n  HELDOUT analogy accuracy: {correct}/{n} = {correct/n:.0%}")

    # ── Bonus: cross-relation analogy (country->continent direction) ──
    print("\n=== BONUS: continent direction (country -> continent) ===")
    print("  Same setup but for the IN_CONTINENT relation.\n")
    cont_pairs = [(c[0], c[2]) for c in COUNTRIES[:20]]
    cont_held  = [(c[0], c[2]) for c in COUNTRIES[20:]]
    cont_deltas = []
    for country, continent in cont_pairs:
        cv = vec(country)
        kv = vec(continent)
        if cv.sum() == 0 or kv.sum() == 0:
            continue
        cont_deltas.append(np.bitwise_xor(cv, kv))
    if cont_deltas:
        stack = np.stack(cont_deltas)
        avg_cont = (stack.sum(axis=0) > stack.shape[0]/2).astype(np.int8)
        all_continents = list({c[2] for c in COUNTRIES})
        cv_lookup = {c: vec(c) for c in all_continents}
        correct = 0
        for country, expected in cont_held:
            cv = vec(country)
            if cv.sum() == 0: continue
            pred = np.bitwise_xor(cv, avg_cont)
            scored = sorted(
                [(c, hamming_distance(pred, v)) for c, v in cv_lookup.items() if v.sum() > 0],
                key=lambda x: x[1]
            )
            ok = scored[0][0] == expected
            correct += int(ok)
            mark = "OK" if ok else "MISS"
            print(f"  [{mark}] {country:14s} -> {scored[0][0]} (expected {expected})")
        print(f"\n  Continent analogy: {correct}/{len(cont_held)}")


if __name__ == "__main__":
    main()
