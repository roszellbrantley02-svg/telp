"""
lattice/test_cross_domain_v2.py - crack cross-domain analogy with shared HAS_PRIMARY.

Same 5 domains as v1, but every primary feature uses the SAME relation
role (HAS_PRIMARY). Now the algebra SHOULD transfer because the
underlying role vector is shared across domains.

Tests:
  T1: within-domain queries still work (sanity)
  T2: HAS_PRIMARY direction now recovered by averaged delta — should
      be near-identical to the explicit role
  T3: cross-domain analogies via one-shot delta
      Germany:Berlin :: Lion:???    expected: Savannah
      Pizza:Italy    :: Tennis:???  expected: Racket
      Shakespeare:Hamlet :: Lion:?? expected: Savannah
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.relation_indexing import RelationBoundEncoder
from lattice.corpus_unified import (
    build_unified_corpus, PRIMARY_BY_DOMAIN, ATTRIBS_BY_DOMAIN, HELDOUT,
)
from train.v5_hdc_prototype import D, hamming_distance


def main():
    print("CROSS-DOMAIN ANALOGY v2 — shared HAS_PRIMARY relation\n")
    print("Hypothesis: with one shared role across all 5 domains, the algebra")
    print("should transfer. Germany:Berlin :: Lion:Savannah should work.\n")

    print("Training RBI on unified corpus ...")
    enc = RelationBoundEncoder(dim=D, seed=42)
    enc.train(build_unified_corpus())
    print(f"  vocab: {len(enc.index_vectors)}, relations: {len(enc.role_vectors)}, "
          f"triples: {enc.triple_count}\n")

    # ── T1: within-domain sanity ──
    print("=== TEST 1: within-domain queries (uses shared HAS_PRIMARY) ===")
    for domain, pairs in PRIMARY_BY_DOMAIN.items():
        cands = ATTRIBS_BY_DOMAIN[domain]
        correct = 0
        for ent, attr in pairs:
            results = enc.query(ent, "HAS_PRIMARY", cands, top_k=1)
            if results and results[0][0] == attr:
                correct += 1
        ent_h, attr_h = HELDOUT[domain]
        results = enc.query(ent_h, "HAS_PRIMARY", cands, top_k=1)
        heldout_ok = (results and results[0][0] == attr_h)
        print(f"  {domain:10s}: train {correct}/{len(pairs)}  "
              f"heldout {ent_h}->{attr_h}: {'OK' if heldout_ok else 'MISS'}")

    # ── T2: derived direction matches explicit role ──
    print("\n=== TEST 2: derive HAS_PRIMARY direction from each domain's deltas ===")
    explicit = enc._role("HAS_PRIMARY")
    domain_deltas = {}
    for domain, pairs in PRIMARY_BY_DOMAIN.items():
        deltas = []
        for ent, attr in pairs:
            ctx = enc.get_context(ent)
            attr_idx = enc.get_index(attr)
            if ctx.sum() == 0 or attr_idx.sum() == 0:
                continue
            deltas.append(np.bitwise_xor(ctx, attr_idx))
        stack = np.stack(deltas)
        avg = (stack.sum(axis=0) > len(deltas)/2).astype(np.int8)
        domain_deltas[domain] = avg
        d_to_role = hamming_distance(avg, explicit)
        print(f"  {domain:10s}: derived delta vs explicit HAS_PRIMARY role: "
              f"{d_to_role} bits ({d_to_role/D*100:.1f}%)")

    print("\n  Pairwise distance between derived domain deltas:")
    print(f"  {'':12s} | " + " | ".join(f"{d:>10s}" for d in domain_deltas))
    print(f"  {'-'*12} | " + " | ".join("-"*10 for _ in domain_deltas))
    for d1 in domain_deltas:
        row = []
        for d2 in domain_deltas:
            if d1 == d2:
                row.append("    -")
            else:
                d = hamming_distance(domain_deltas[d1], domain_deltas[d2])
                row.append(f"{d/D*100:>5.1f}%")
        print(f"  {d1:12s} | " + " | ".join(f"{r:>10s}" for r in row))

    # ── T3: cross-domain analogies via ONE-SHOT delta ──
    print("\n=== TEST 3: cross-domain analogies via one-shot delta ===")
    print("  Take delta(A, B) from ONE pair. Apply to entity C. Find nearest.\n")

    all_attribs = [(d, a) for d, attrs in ATTRIBS_BY_DOMAIN.items() for a in attrs]

    test_cases = [
        # (anchor_a, anchor_b, query_c, expected_d, expected_domain)
        ("France",      "Paris",        "Lion",          "Savannah",      "animal"),
        ("France",      "Paris",        "Pizza",         "Italy",         "food"),
        ("France",      "Paris",        "Tennis",        "Racket",        "sport"),
        ("France",      "Paris",        "Shakespeare",   "Hamlet",        "author"),
        ("Lion",        "Savannah",     "Germany",       "Berlin",        "country"),
        ("Lion",        "Savannah",     "Pizza",         "Italy",         "food"),
        ("Lion",        "Savannah",     "Tennis",        "Racket",        "sport"),
        ("Lion",        "Savannah",     "Shakespeare",   "Hamlet",        "author"),
        ("Pizza",       "Italy",        "Germany",       "Berlin",        "country"),
        ("Pizza",       "Italy",        "Lion",          "Savannah",      "animal"),
        ("Pizza",       "Italy",        "Tennis",        "Racket",        "sport"),
        ("Pizza",       "Italy",        "Shakespeare",   "Hamlet",        "author"),
        ("Shakespeare", "Hamlet",       "Germany",       "Berlin",        "country"),
        ("Shakespeare", "Hamlet",       "Lion",          "Savannah",      "animal"),
        ("Shakespeare", "Hamlet",       "Pizza",         "Italy",         "food"),
        ("Shakespeare", "Hamlet",       "Tennis",        "Racket",        "sport"),
        ("Tennis",      "Racket",       "Germany",       "Berlin",        "country"),
        ("Tennis",      "Racket",       "Lion",          "Savannah",      "animal"),
        ("Tennis",      "Racket",       "Pizza",         "Italy",         "food"),
        ("Tennis",      "Racket",       "Shakespeare",   "Hamlet",        "author"),
    ]

    correct = 0
    print(f"  {'A':12s} : {'B':12s}  ::  {'C':12s} : {'predicted':14s} (expected)")
    print("  " + "-" * 75)
    for a, b, c, expected, expected_domain in test_cases:
        a_ctx = enc.get_context(a)
        b_idx = enc.get_index(b)
        c_ctx = enc.get_context(c)
        if any(v.sum() == 0 for v in [a_ctx, b_idx, c_ctx]):
            print(f"  {a:12s} : {b:12s}  ::  {c:12s} : (vector missing)")
            continue
        delta = np.bitwise_xor(a_ctx, b_idx)
        pred = np.bitwise_xor(c_ctx, delta)
        # Find nearest across ALL attribs (cross-domain)
        scored = []
        for d, attr in all_attribs:
            attr_idx = enc.get_index(attr)
            if attr_idx.sum() == 0: continue
            scored.append((d, attr, hamming_distance(pred, attr_idx)))
        scored.sort(key=lambda x: x[2])
        top = scored[0]
        mark = "OK" if top[1] == expected else "  "
        correct += int(top[1] == expected)
        print(f"  [{mark}] {a:10s} : {b:10s}  ::  {c:10s} : {top[1]:14s}  ({expected} / {expected_domain})")

    n = len(test_cases)
    print(f"\n  Cross-domain one-shot accuracy: {correct}/{n} = {correct/n:.0%}")

    # ── T4: many-shot averaged delta — should be even cleaner ──
    print("\n=== TEST 4: many-shot averaged delta (all 5 domains combined) ===")
    print("  Average delta across pairs from ALL 5 domains. Apply to each test entity.\n")
    all_pairs = []
    for d, pairs in PRIMARY_BY_DOMAIN.items():
        all_pairs.extend(pairs)
    all_deltas = []
    for ent, attr in all_pairs:
        ctx = enc.get_context(ent)
        attr_idx = enc.get_index(attr)
        if ctx.sum() and attr_idx.sum():
            all_deltas.append(np.bitwise_xor(ctx, attr_idx))
    stack = np.stack(all_deltas)
    universal_delta = (stack.sum(axis=0) > len(all_deltas)/2).astype(np.int8)

    # Test every entity, every domain
    test_entities = [
        ("Germany", "Berlin"),
        ("Lion", "Savannah"),
        ("Pizza", "Italy"),
        ("Shakespeare", "Hamlet"),
        ("Tennis", "Racket"),
        ("Wolf", "Forest"),          # heldout
        ("Pretzel", "Germany"),      # heldout
        ("Kafka", "Metamorphosis"),  # heldout
        ("Basketball", "Ball"),      # heldout
    ]
    correct_uni = 0
    for ent, expected in test_entities:
        ctx = enc.get_context(ent)
        if ctx.sum() == 0:
            print(f"  {ent:14s}: no vector")
            continue
        pred = np.bitwise_xor(ctx, universal_delta)
        scored = []
        for d, attr in all_attribs:
            attr_idx = enc.get_index(attr)
            if attr_idx.sum() == 0: continue
            scored.append((d, attr, hamming_distance(pred, attr_idx)))
        scored.sort(key=lambda x: x[2])
        top = scored[0]
        ok = (top[1] == expected)
        correct_uni += int(ok)
        mark = "OK" if ok else "  "
        print(f"  [{mark}] {ent:14s} + universal_delta -> {top[1]:14s} "
              f"(expected {expected})")
    print(f"\n  Universal delta accuracy: {correct_uni}/{len(test_entities)} = "
          f"{correct_uni/len(test_entities):.0%}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Test 3 — one-shot cross-domain analogies: {correct}/{n} = {correct/n:.0%}")
    print(f"  Test 4 — universal delta:                  {correct_uni}/{len(test_entities)} = {correct_uni/len(test_entities):.0%}")
    print()
    if correct/n >= 0.7:
        print("  BREAKTHROUGH: shared HAS_PRIMARY relation enables cross-domain analogy.")
        print("  Germany:Berlin :: Lion:Savannah now works via pure HDC algebra.")
        print("  The 'primary feature' direction TRANSFERS across domains.")
    elif correct/n >= 0.4:
        print("  PARTIAL: shared role helps but not perfect.")
    else:
        print("  STUCK: even shared role isn't enough.")


if __name__ == "__main__":
    main()
