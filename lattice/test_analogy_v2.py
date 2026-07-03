"""
lattice/test_analogy_v2.py - the CORRECT formulation of HDC analogy.

The first attempt (test_analogy.py) failed because it computed
Germany_context XOR Berlin_context. Both have noise; XORing them
amplifies it.

The right formulation:
    delta_i = country_i.context XOR capital_i.index

Why this works:
    country.context contains bind(HAS_CAPITAL_role, capital.index)
    XOR by capital.index cancels its contribution to that bound term:
        bind(HAS_CAPITAL, capital.index) XOR capital.index = HAS_CAPITAL
    Other relations in country.context (IN_CONTINENT, SPEAKS, etc.)
    become noise after the XOR — different noise per pair.
    Averaging deltas across many pairs cancels the noise, leaving
    HAS_CAPITAL_role.

This is analogy DERIVED from data, not declared.
If it works, we get king-man+woman=queen for HDC by a different
math but the same outcome: structure emerges from training pairs.

Test:
  T1: are delta_i = country_ctx XOR capital_idx consistent across pairs?
  T2: does the averaged delta act as HAS_CAPITAL_role?
  T3: does applying it to held-out countries recover their capitals?

If yes on all three, HDC analogy via algebra works after all —
just with the right delta formulation.
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


KNOWN  = [(c[0], c[1]) for c in COUNTRIES[:20]]
HELDOUT = [(c[0], c[1]) for c in COUNTRIES[20:]]


def main():
    print("HDC Analogy v2 — using country.CONTEXT XOR capital.INDEX\n")

    print("Training RBI on tagged corpus ...")
    enc = RelationBoundEncoder(dim=D, seed=42)
    enc.train(build_tagged_corpus())

    # ── T1: delta consistency ──
    print("\n=== TEST 1: delta consistency (NEW formulation) ===")
    print("  delta_i = country_i.CONTEXT XOR capital_i.INDEX")
    print("  Should be similar across pairs (all ~= HAS_CAPITAL_role + noise).\n")
    deltas = []
    for country, capital in KNOWN:
        ctx = enc.get_context(country)
        cap_idx = enc.get_index(capital)
        if ctx.sum() == 0:
            continue
        deltas.append(np.bitwise_xor(ctx, cap_idx))
    print(f"  Collected {len(deltas)} deltas.")

    if len(deltas) >= 2:
        intra = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                intra.append(hamming_distance(deltas[i], deltas[j]))
        mean_intra = np.mean(intra) / D
        print(f"  Mean pairwise Hamming: {mean_intra*100:.1f}% of D")
        print(f"  Random baseline:       50.0%")
        if mean_intra < 0.40:
            print("  STRONG: deltas are clearly consistent.")
        elif mean_intra < 0.48:
            print("  MODEST: some consistency.")
        else:
            print("  WEAK / NONE.")

    # ── T2: does the averaged delta act as HAS_CAPITAL_role? ──
    print("\n=== TEST 2: averaged delta vs the explicit HAS_CAPITAL_role ===")
    delta_stack = np.stack(deltas)
    avg_delta = (delta_stack.sum(axis=0) > len(deltas)/2).astype(np.int8)
    explicit_role = enc._role("HAS_CAPITAL")
    d_to_role = hamming_distance(avg_delta, explicit_role)
    print(f"  Avg delta vs explicit HAS_CAPITAL_role: {d_to_role} bits "
          f"= {d_to_role/D*100:.1f}% of D")
    if d_to_role / D < 0.30:
        print("  STRONG: averaged delta closely matches the explicit role.")
    elif d_to_role / D < 0.45:
        print("  MODEST: deltas approximate the role.")
    else:
        print("  WEAK.")

    # ── T3: held-out analogy ──
    print("\n=== TEST 3: HELDOUT analogy via averaged delta ===")
    all_capitals = list({c[1] for c in COUNTRIES})
    correct = 0
    print(f"  {'COUNTRY':14s} | top-1            | rank of correct")
    print(f"  {'-' * 14} | {'-' * 16} | {'-' * 15}")
    for country, expected in HELDOUT:
        ctx = enc.get_context(country)
        if ctx.sum() == 0:
            continue
        pred = np.bitwise_xor(ctx, avg_delta)
        scored = sorted(
            [(c, hamming_distance(pred, enc.get_index(c)))
              for c in all_capitals],
            key=lambda x: x[1]
        )
        names = [s[0] for s in scored]
        top1 = names[0]
        rank = names.index(expected) + 1
        ok = (top1 == expected)
        correct += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"  {country:14s} | [{mark}] {top1:12s} | {rank:>15d}")
    n = len(HELDOUT)
    print(f"\n  HELDOUT analogy (derived role): {correct}/{n} = {correct/n:.0%}")

    # ── Compare with the directly-stored role ──
    print("\n=== TEST 4: compare with the EXPLICIT role query (RBI standard) ===")
    correct2 = 0
    for country, expected in HELDOUT:
        results = enc.query(country, "HAS_CAPITAL", all_capitals, top_k=1)
        if results and results[0][0] == expected:
            correct2 += 1
    print(f"  Standard RBI query (uses explicit role): {correct2}/{n} = "
          f"{correct2/n:.0%}")

    # ── Bonus: one-shot analogy ──
    print("\n=== TEST 5: one-shot analogy (use a SINGLE known pair) ===")
    one_shot_delta = deltas[0]
    anchor_country = KNOWN[0][0]
    anchor_capital = KNOWN[0][1]
    print(f"  Anchor: {anchor_country} -> {anchor_capital}")
    correct3 = 0
    for country, expected in HELDOUT:
        ctx = enc.get_context(country)
        if ctx.sum() == 0:
            continue
        pred = np.bitwise_xor(ctx, one_shot_delta)
        scored = sorted(
            [(c, hamming_distance(pred, enc.get_index(c)))
              for c in all_capitals],
            key=lambda x: x[1]
        )
        if scored[0][0] == expected:
            correct3 += 1
    print(f"  One-shot accuracy: {correct3}/{n} = {correct3/n:.0%}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"  Standard RBI (explicit role):     {correct2}/{n}")
    print(f"  Averaged-delta analogy (derived): {correct}/{n}")
    print(f"  One-shot analogy:                  {correct3}/{n}")
    if correct / n >= 0.70:
        print("\n  BREAKTHROUGH: HDC analogy via averaged deltas WORKS.")
        print("  The 'structured direction' emerges from training pairs.")
        print("  Word2Vec's king-queen trick now has an HDC analog.")


if __name__ == "__main__":
    main()
