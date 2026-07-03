"""
lattice/test_hdc_gnn.py - can HDC message passing crack heldout HAS_CAPITAL?

Setup:
  - Enriched corpus (capitals have their own facts: continent, language, etc.)
  - Hold out (country, HAS_CAPITAL, capital) for the 7 heldout countries
  - Hold out (capital, HAS_CAPITAL_INV, country) too
  - Capitals STILL have their own non-capital relations in training
  - So they're NOT isolated nodes — message passing has something to work with

Test:
  Train HDC GNN with K rounds of message passing.
  Vary K from 1 to 6.
  Measure heldout HAS_CAPITAL accuracy at each K.

If accuracy improves with K, message passing is doing real work.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.corpus_enriched import build_enriched_corpus
from lattice.hdc_gnn import HDCGraphNN
from lattice.corpus import COUNTRIES
from train.v5_hdc_prototype import D


HELDOUT_COUNTRIES = {c[0] for c in COUNTRIES[25:]}
HELDOUT_PAIRS = [(c[0], c[1]) for c in COUNTRIES[25:]]
ALL_CAPITALS = sorted({c[1] for c in COUNTRIES})


def filter_triples(tagged_corpus):
    """Remove HAS_CAPITAL and HAS_CAPITAL_INV triples for heldout countries."""
    out = []
    removed = 0
    for sent, triples in tagged_corpus:
        kept = []
        for t in triples:
            s, r, o = t
            if r == "HAS_CAPITAL" and s in HELDOUT_COUNTRIES:
                removed += 1; continue
            if r == "HAS_CAPITAL_INV" and o in HELDOUT_COUNTRIES:
                removed += 1; continue
            kept.append(t)
        for t in kept:
            out.append(t)
    return out, removed


def evaluate(gnn: HDCGraphNN) -> tuple[int, int]:
    """Top-1 + Top-3 accuracy on heldout HAS_CAPITAL queries."""
    h_top1 = 0
    h_top3 = 0
    for country, expected in HELDOUT_PAIRS:
        results = gnn.query(country, "HAS_CAPITAL", ALL_CAPITALS, top_k=20)
        names = [r[0] for r in results]
        if not names:
            continue
        if names[0] == expected:
            h_top1 += 1
        if expected in names[:3]:
            h_top3 += 1
    return h_top1, h_top3


def main():
    print("HDC Graph Message Passing — can it crack heldout HAS_CAPITAL?\n")

    print("Building enriched corpus (capitals have their own facts) ...")
    corpus = build_enriched_corpus()
    print(f"  sentences: {len(corpus)}")

    print("Filtering out HAS_CAPITAL triples for heldout countries ...")
    triples, removed = filter_triples(corpus)
    print(f"  removed {removed} capital-related triples for heldout")
    print(f"  remaining training triples: {len(triples)}")

    # Verify capitals still have presence
    heldout_caps = {p[1] for p in HELDOUT_PAIRS}
    cap_appearances = {c: 0 for c in heldout_caps}
    for s, r, o in triples:
        if s in cap_appearances:
            cap_appearances[s] += 1
        if o in cap_appearances:
            cap_appearances[o] += 1
    print(f"  heldout capitals' triple presence:")
    for c, n in cap_appearances.items():
        print(f"    {c:12s}: {n} triples (must be > 0 for GNN to work)")

    # ── Test message passing with different K ──
    print("\n=== Heldout HAS_CAPITAL accuracy vs message-passing rounds K ===\n")
    print(f"  {'K rounds':>10s} | {'Top-1':>10s} | {'Top-3':>10s}")
    print(f"  {'-'*10} | {'-'*10} | {'-'*10}")

    results_summary = []
    for K in [0, 1, 2, 3, 4, 5, 6]:
        # Fresh GNN each time
        gnn = HDCGraphNN(dim=D, seed=42, decay=0.5)
        gnn.add_triples(triples)
        gnn.run_rounds(K)
        top1, top3 = evaluate(gnn)
        results_summary.append((K, top1, top3))
        print(f"  {K:>10d} | {top1:>5d}/{len(HELDOUT_PAIRS)} | "
              f"{top3:>5d}/{len(HELDOUT_PAIRS)}")

    # Detail at the best K
    best_K = max(results_summary, key=lambda x: x[1])[0]
    print(f"\n=== Detail at K={best_K} ===")
    gnn = HDCGraphNN(dim=D, seed=42, decay=0.5)
    gnn.add_triples(triples)
    gnn.run_rounds(best_K)
    for country, expected in HELDOUT_PAIRS:
        results = gnn.query(country, "HAS_CAPITAL", ALL_CAPITALS, top_k=3)
        names = [r[0] for r in results]
        rank = names.index(expected) + 1 if expected in names else "out"
        mark = "OK" if names and names[0] == expected else "  "
        print(f"  [{mark}] {country:14s} top-3: {names}  (rank of {expected}: {rank})")

    # Summary
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    best_top1 = max(r[1] for r in results_summary)
    best_top3 = max(r[2] for r in results_summary)
    n = len(HELDOUT_PAIRS)
    print(f"  Best Top-1 across K: {best_top1}/{n} = {best_top1/n:.0%}")
    print(f"  Best Top-3 across K: {best_top3}/{n} = {best_top3/n:.0%}")
    print()
    print(f"  Diff HDC (no graph):       0/7  = 0%")
    print(f"  RBI w/ kept triples:       100/7 (those triples ARE present)")
    print(f"  RBI multi-hop (chain):     6/7  = 86%")
    print(f"  HDC GNN message passing:   {best_top1}/{n} = {best_top1/n:.0%}")


if __name__ == "__main__":
    main()
