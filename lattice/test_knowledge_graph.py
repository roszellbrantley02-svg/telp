"""
lattice/test_knowledge_graph.py - does HRR HDC store and retrieve facts?

Two main tests:

  TEST 1: pure storage and retrieval
    Store all 25 (country, HAS_CAPITAL, capital) facts. Query each one.
    Should be 100% if HDC algebra works as designed. Failures mean
    bundle capacity is overrun.

  TEST 2: capacity curve
    Add facts one by one. Measure retrieval accuracy as bundle grows.
    When does cross-talk noise overwhelm the cleanup memory?

  TEST 3 (bonus): hybrid with RI-derived entity vectors
    Use RI-trained vectors for entities (semantic) instead of random.
    Query for held-out countries — does the algebra extend to them
    via semantic similarity?

Usage:
    python -m lattice.test_knowledge_graph
"""
from __future__ import annotations

import sys
from pathlib import Path

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.knowledge_graph import HDCKnowledgeBase
from lattice.corpus import COUNTRIES
from lattice.random_indexing import RandomIndexingEncoder
from lattice.corpus import build_corpus
from train.v5_hdc_prototype import D


# Same split as previous tests
TRAIN = [(c[0], c[1]) for c in COUNTRIES[:25]]
HELDOUT = [(c[0], c[1]) for c in COUNTRIES[25:]]


# ─── TEST 1: storage + retrieval ───────────────────────────────────


def test_storage_retrieval():
    print("=== TEST 1: store + retrieve all 25 country->capital facts ===\n")
    kb = HDCKnowledgeBase()

    # Add all 25 facts
    for country, capital in TRAIN:
        kb.add_fact(country, "HAS_CAPITAL", capital)
    print(f"  facts stored: {len(kb)}")
    print(f"  entities: {len(kb.entities)}  relations: {len(kb.relations)}")

    # All capitals are the cleanup space
    all_capitals = [cap for _, cap in TRAIN]

    print("\n  Query each stored country:")
    correct = 0
    for country, expected_capital in TRAIN:
        results = kb.query(country, "HAS_CAPITAL", top_k=3,
                              restrict_to=all_capitals)
        top1 = results[0][0]
        ok = (top1 == expected_capital)
        correct += int(ok)
        mark = "OK" if ok else "MISS"
        d_top1 = results[0][1]
        d_exp = next((d for n, d in results if n == expected_capital), "out")
        print(f"  [{mark}] {country:18s} -> {top1:18s} "
              f"(d={d_top1}, expected {expected_capital} at d={d_exp})")
    n = len(TRAIN)
    acc = correct / n
    print(f"\n  Storage retrieval accuracy: {correct}/{n} = {acc:.0%}")
    return acc


# ─── TEST 2: capacity curve ───────────────────────────────────────


def test_capacity_curve():
    print("\n=== TEST 2: capacity — does accuracy degrade as bundle grows? ===\n")
    print("  Store facts one at a time; after each, query ALL stored facts")
    print("  and report accuracy.\n")
    print(f"  {'n_facts':10s} | {'accuracy':10s} | {'min margin':12s}")
    print(f"  {'-' * 10} | {'-' * 10} | {'-' * 12}")
    kb = HDCKnowledgeBase()
    all_capitals_so_far = []
    for i, (country, capital) in enumerate(TRAIN):
        kb.add_fact(country, "HAS_CAPITAL", capital)
        all_capitals_so_far.append(capital)
        # Query everything stored
        stored = TRAIN[:i + 1]
        n_correct = 0
        margins = []
        for c, cap in stored:
            results = kb.query(c, "HAS_CAPITAL", top_k=2,
                                  restrict_to=all_capitals_so_far)
            if not results:
                continue
            if results[0][0] == cap:
                n_correct += 1
            # Margin: distance to wrong answer minus distance to right
            if len(results) >= 2 and results[0][0] == cap:
                margins.append(results[1][1] - results[0][1])
        acc = n_correct / (i + 1)
        min_margin = min(margins) if margins else "n/a"
        print(f"  {i+1:>10d} | {acc*100:>9.1f}% | {str(min_margin):>12s}")
    return acc


# ─── TEST 3: hybrid — RI entities + HRR facts ────────────────────


def test_hybrid_with_ri():
    print("\n=== TEST 3: hybrid — RI semantic entities + HRR fact storage ===\n")
    print("  Train RI on corpus -> get semantic entity vectors.")
    print("  Use those as the KB's entity vectors.")
    print("  Store training facts (country, HAS_CAPITAL, capital).")
    print("  Query HELDOUT countries — does algebra extend?\n")

    # Train RI
    print("  Training RI on country-fact corpus ...")
    ri = RandomIndexingEncoder(dim=D, sparsity=20, window=5, seed=42)
    ri.train(build_corpus())
    print(f"    vocab: {ri.vocab_size()} words")

    kb = HDCKnowledgeBase()

    # Override entity vectors with RI-derived semantic vectors
    all_entities = set()
    for c, cap in TRAIN + HELDOUT:
        all_entities.add(c)
        all_entities.add(cap)
    n_with_ri = 0
    for name in all_entities:
        v = ri.encode(name)
        if v.sum() > 0:   # has RI signal
            kb.set_entity_vector(name, v)
            n_with_ri += 1
    print(f"    {n_with_ri}/{len(all_entities)} entities given RI-derived vectors")

    # Add training facts
    for c, cap in TRAIN:
        kb.add_fact(c, "HAS_CAPITAL", cap)
    print(f"    {len(kb)} facts stored\n")

    all_capitals = list(set(cap for _, cap in TRAIN + HELDOUT))

    print("  TRAINING-set queries (sanity check):")
    train_correct = 0
    for c, expected in TRAIN:
        results = kb.query(c, "HAS_CAPITAL", top_k=1, restrict_to=all_capitals)
        ok = (results[0][0] == expected)
        train_correct += int(ok)
    print(f"    training accuracy: {train_correct}/{len(TRAIN)} = "
          f"{train_correct/len(TRAIN):.0%}")

    print("\n  HELDOUT queries (the real test):")
    print(f"  {'COUNTRY':14s} | top-1 prediction  | rank of correct")
    print(f"  {'-' * 14} | {'-' * 17} | {'-' * 15}")
    h_top1 = 0
    h_top3 = 0
    for c, expected in HELDOUT:
        results = kb.query(c, "HAS_CAPITAL", top_k=20,
                              restrict_to=all_capitals)
        names = [r[0] for r in results]
        rank = names.index(expected) + 1 if expected in names else "out"
        is_top1 = (names[0] == expected)
        is_top3 = isinstance(rank, int) and rank <= 3
        h_top1 += int(is_top1)
        h_top3 += int(is_top3)
        mark = "T1" if is_top1 else ("T3" if is_top3 else "  ")
        print(f"  {c:14s} | [{mark}] {names[0]:14s} | {str(rank):>15s}")
    n = len(HELDOUT)
    print(f"\n  HELDOUT Top-1: {h_top1}/{n} = {h_top1/n:.0%}")
    print(f"  HELDOUT Top-3: {h_top3}/{n} = {h_top3/n:.0%}")


# ─── Main ──────────────────────────────────────────────────────────


def main():
    print("HDC Knowledge Graph (HRR) — pure-HDC relational reasoning\n")

    acc1 = test_storage_retrieval()
    acc2 = test_capacity_curve()
    test_hybrid_with_ri()

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if acc1 >= 0.95:
        print(f"  TEST 1 (storage):   PASS ({acc1*100:.0f}%) — HDC algebra encodes")
        print(f"                            and retrieves relational facts cleanly.")
    elif acc1 >= 0.50:
        print(f"  TEST 1 (storage):   PARTIAL ({acc1*100:.0f}%) — bundle capacity issues")
    else:
        print(f"  TEST 1 (storage):   FAIL ({acc1*100:.0f}%) — fundamental cross-talk problem")
    print()
    print("  This tests the BUILDING BLOCK. Generalization to held-out")
    print("  entities (test 3) is the harder question.")


if __name__ == "__main__":
    main()
