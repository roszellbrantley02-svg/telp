"""
lattice/test_random_indexing.py - does HDC learn meaning from text?

Two tests, two layers of evidence:

  TEST 1: semantic similarity
    After training on the country-fact corpus, do words that should be
    related (Germany/Berlin) land closer than unrelated words
    (Germany/Beijing)?

  TEST 2: country->capital generalization
    Re-run the feed-forward test from earlier, but with Random Indexing
    vectors instead of MiniLM. Held out countries: does the HDC layer
    generalize?

Honest expectations:
  - TEST 1 should pass clearly (RI captures co-occurrence)
  - TEST 2 will probably land between 20% (pure HDC) and 80% (MiniLM)
    on heldout. ~40-60% would be a real "HDC learning semantics" win.

Usage:
    python -m lattice.test_random_indexing
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.corpus import build_corpus, COUNTRIES
from lattice.random_indexing import RandomIndexingEncoder
from lattice.feedforward import HDCFeedForward
from train.v5_hdc_prototype import hamming_distance, D


# ─── Train RI encoder ─────────────────────────────────────────────


def train_encoder() -> RandomIndexingEncoder:
    print("Building corpus ...")
    corpus = build_corpus()
    print(f"  {len(corpus)} sentences")

    print("Training Random Indexing encoder ...")
    enc = RandomIndexingEncoder(dim=D, sparsity=20, window=5, seed=42)
    enc.train(corpus)
    print(f"  vocab size: {enc.vocab_size()} words")
    return enc


# ─── Test 1: semantic similarity ──────────────────────────────────


def test_semantic_similarity(enc: RandomIndexingEncoder) -> bool:
    print("\n=== TEST 1: semantic similarity (co-occurrence pairs) ===")
    print("  Each row: target word vs. expected-close word vs. distractor.")
    print("  Closer-than-distractor counts as a pass.\n")

    triples = [
        # (target, expected_close, distractor)
        ("germany",  "berlin",   "tokyo"),
        ("france",   "paris",    "beijing"),
        ("japan",    "tokyo",    "rome"),
        ("italy",    "rome",     "moscow"),
        ("china",    "beijing",  "ottawa"),
        ("russia",   "moscow",   "delhi"),
        ("egypt",    "cairo",    "athens"),
        # Continent-level
        ("germany",  "europe",   "asia"),
        ("japan",    "asia",     "africa"),
        ("egypt",    "africa",   "oceania"),
        # Language-level
        ("france",   "french",   "japanese"),
        ("japan",    "japanese", "german"),
    ]

    passes = 0
    print(f"  {'TARGET':10s} | {'CLOSE':12s} | {'DISTRACTOR':12s} | d(close)/d(distract)")
    print(f"  {'-' * 10} | {'-' * 12} | {'-' * 12} | --------")
    for tgt, close, distract in triples:
        t_hv = enc.encode_word(tgt)
        c_hv = enc.encode_word(close)
        d_hv = enc.encode_word(distract)
        if t_hv.sum() == 0 or c_hv.sum() == 0 or d_hv.sum() == 0:
            print(f"  {tgt:10s} | {close:12s} | {distract:12s} | [SKIP - unknown word]")
            continue
        d_close = hamming_distance(t_hv, c_hv)
        d_distract = hamming_distance(t_hv, d_hv)
        ok = d_close < d_distract
        passes += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"  {tgt:10s} | {close:12s} | {distract:12s} | "
              f"{d_close}/{d_distract}  [{mark}]")

    n = len(triples)
    print(f"\n  Semantic similarity score: {passes}/{n} = {passes/n:.0%}")
    print(f"  Pass if >= 80% (RI should clearly capture co-occurrence)")
    return passes / n >= 0.80


# ─── Test 2: country -> capital generalization ───────────────────


# Same training/heldout split as the MiniLM test
TRAIN_PAIRS = [(c[0], c[1]) for c in COUNTRIES[:25]]
HELDOUT     = [(c[0], c[1]) for c in COUNTRIES[25:]]


def test_country_capital(enc: RandomIndexingEncoder) -> bool:
    print("\n=== TEST 2: country -> capital with RI vectors ===")
    print(f"  Train on {len(TRAIN_PAIRS)} pairs, test on {len(HELDOUT)} heldout.\n")

    # Build feed-forward layer with RI-derived hypervectors
    ff = HDCFeedForward()

    # Add all capitals to cleanup memory
    all_capitals = list(set(cap for _, cap in TRAIN_PAIRS + HELDOUT))
    for cap in all_capitals:
        ff.add_cleanup_item(cap, enc.encode(cap))

    # Train on the 25 country pairs
    train_data = [(c, enc.encode(c), cap, enc.encode(cap))
                    for c, cap in TRAIN_PAIRS]
    ff.train(train_data)

    # ── Training-set recall ──
    print("  Training-set recall:")
    train_correct = 0
    for c, cap in TRAIN_PAIRS:
        results = ff.forward(enc.encode(c), top_k=1)
        ok = (results[0][0] == cap)
        train_correct += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"    [{mark}] {c:18s} -> {results[0][0]:18s} (exp {cap})")
    print(f"\n  Training accuracy: {train_correct}/{len(TRAIN_PAIRS)} = "
          f"{train_correct/len(TRAIN_PAIRS):.0%}")

    # ── Heldout ──
    print("\n  HELDOUT (countries never in feed-forward training):\n")
    print(f"  {'COUNTRY':14s} | top-1            | rank of correct")
    print(f"  {'-' * 14} | {'-' * 16} | {'-' * 15}")
    h_top1 = 0
    h_top3 = 0
    for c, cap in HELDOUT:
        results = ff.forward(enc.encode(c), top_k=20)
        names = [r[0] for r in results]
        rank = names.index(cap) + 1 if cap in names else "out"
        is_top1 = (names[0] == cap)
        is_top3 = isinstance(rank, int) and rank <= 3
        h_top1 += int(is_top1)
        h_top3 += int(is_top3)
        mark = "T1" if is_top1 else ("T3" if is_top3 else "  ")
        print(f"  {c:14s} | [{mark}] {names[0]:14s} | {str(rank):>15s}")

    n = len(HELDOUT)
    print(f"\n  HELDOUT Top-1: {h_top1}/{n} = {h_top1/n:.0%}")
    print(f"  HELDOUT Top-3: {h_top3}/{n} = {h_top3/n:.0%}")
    return h_top1 / n >= 0.40


# ─── Main ──────────────────────────────────────────────────────────


def main():
    print("HDC Random Indexing — does HDC learn semantics from text alone?\n")
    enc = train_encoder()

    # Show some learned neighborhoods for inspection
    print("\nSpot-check: learned nearest neighbors")
    for word in ["germany", "europe", "currency", "capital"]:
        nbrs = enc.top_neighbors(word, k=5)
        if nbrs:
            nbr_str = ", ".join(f"{w}({d})" for w, d in nbrs)
            print(f"  {word:15s} -> {nbr_str}")

    t1 = test_semantic_similarity(enc)
    t2 = test_country_capital(enc)

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"  Test 1 (semantic similarity):       "
          f"{'PASS' if t1 else 'FAIL'}")
    print(f"  Test 2 (country->capital heldout):  "
          f"{'PASS' if t2 else 'FAIL'}")
    print()
    print("  COMPARISON ACROSS ENCODERS:")
    print("  ┌─────────────────────────┬──────────────────┐")
    print("  │ Encoder                  │ Heldout Top-1    │")
    print("  ├─────────────────────────┼──────────────────┤")
    print("  │ MiniLM (LLM)             │ 80%              │")
    print("  │ Random Indexing (HDC)    │ run-dependent    │")
    print("  │ Pure HDC (no learning)   │ 20%              │")
    print("  └─────────────────────────┴──────────────────┘")


if __name__ == "__main__":
    main()
