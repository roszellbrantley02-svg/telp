"""
lattice/test_cross_domain.py - do domains share geometric structure in HD space?

Train RBI on 5 domains with parallel structure:
  Country  -> HAS_CAPITAL     -> Capital
  Animal   -> HAS_HABITAT     -> Habitat
  Food     -> HAS_ORIGIN      -> Origin
  Author   -> HAS_FAMOUS_WORK -> Work
  Sport    -> HAS_EQUIPMENT   -> Equipment

Each domain uses DIFFERENT relation role vectors. The hypothesis: even
with different roles, the "primary feature direction" (entity_ctx XOR
attribute_idx, averaged across pairs) might have similar GEOMETRY across
domains — revealing emergent meta-structure in HD space.

Three tests:

  T1: within-domain queries still work (sanity)
  T2: are the "primary direction" vectors similar across domains?
      Compare avg delta for country->capital vs animal->habitat etc
  T3: cross-domain analogy by algebra
      "Germany is to Berlin as Lion is to ???"
      Apply the country->capital direction to Lion. Find nearest.
      Expected: it should point to Lion's habitat (Savannah) IF the
      directions are interchangeable. Or to something else interesting.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.relation_indexing import RelationBoundEncoder
from lattice.corpus_multidomain import (
    build_multidomain_corpus, PRIMARY_RELATIONS, HELDOUT,
    COUNTRIES_MD, ANIMALS, FOODS, AUTHORS, SPORTS,
)
from train.v5_hdc_prototype import D, hamming_distance


def main():
    print("Cross-Domain Analogy Test — does HD space have meta-structure?\n")

    # Train RBI on multi-domain corpus
    print("Training RBI on combined 5-domain corpus ...")
    enc = RelationBoundEncoder(dim=D, seed=42)
    enc.train(build_multidomain_corpus())
    stats = enc.stats()
    print(f"  vocab: {stats['vocab']}, relations: {stats['relations']}, "
          f"triples: {stats['triples']}\n")

    # ── T1: within-domain sanity ──
    print("=== TEST 1: within-domain queries (sanity check) ===")
    all_attribs_by_domain = {
        "country": list({c[1] for c in COUNTRIES_MD}),
        "animal":  list({a[1] for a in ANIMALS}),
        "food":    list({f[1] for f in FOODS}),
        "author":  list({a[1] for a in AUTHORS}),
        "sport":   list({s[1] for s in SPORTS}),
    }
    for domain, (rel, pairs) in PRIMARY_RELATIONS.items():
        cands = all_attribs_by_domain[domain]
        correct = 0
        for ent, attr in pairs:
            results = enc.query(ent, rel, cands, top_k=1)
            if results and results[0][0] == attr:
                correct += 1
        # Heldout
        ent_h, attr_h = HELDOUT[domain]
        results = enc.query(ent_h, rel, cands, top_k=1)
        heldout_ok = (results and results[0][0] == attr_h)
        print(f"  {domain:10s}: train {correct}/{len(pairs)}  "
              f"heldout {ent_h}->{attr_h}: {'OK' if heldout_ok else 'MISS'}")

    # ── T2: are "primary direction" vectors similar across domains? ──
    print("\n=== TEST 2: are 'primary feature directions' similar across domains? ===")
    print("  Compute averaged delta = entity.context XOR attribute.index for each pair.")
    print("  Compare across domains.\n")
    domain_deltas = {}
    for domain, (rel, pairs) in PRIMARY_RELATIONS.items():
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
        print(f"  {domain:10s}: {len(deltas)} deltas averaged")

    # Pairwise Hamming distance
    print("\n  Pairwise distance between 'primary direction' vectors:")
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

    # Compare to known relation role distances
    print("\n  For reference — distance between the EXPLICIT relation role vectors:")
    for d1, (r1, _) in PRIMARY_RELATIONS.items():
        for d2, (r2, _) in PRIMARY_RELATIONS.items():
            if d1 >= d2: continue
            role1 = enc._role(r1)
            role2 = enc._role(r2)
            d = hamming_distance(role1, role2) / D * 100
            print(f"    {r1:18s} vs {r2:18s}: {d:.1f}%")

    # ── T3: cross-domain analogy via algebra ──
    print("\n=== TEST 3: cross-domain analogy via algebra ===")
    print("  Take the country->capital direction and apply it to entities in OTHER domains.")
    print("  Q: 'Germany is to Berlin as Lion is to ???'\n")

    country_direction = domain_deltas["country"]

    # Apply country_direction to ONE entity from each other domain
    test_entities = {
        "country": "Germany",      # the held-out country
        "animal":  "Lion",
        "food":    "Pizza",
        "author":  "Shakespeare",
        "sport":   "Tennis",
    }
    print(f"  Using country->capital averaged delta as the transform:\n")
    for domain, ent in test_entities.items():
        ctx = enc.get_context(ent)
        if ctx.sum() == 0:
            print(f"    {ent:14s}: no context vector")
            continue
        # Apply the country direction
        pred = np.bitwise_xor(ctx, country_direction)
        # Find nearest in ALL attribute candidates (across all domains)
        all_cands = []
        for d, cs in all_attribs_by_domain.items():
            for c in cs:
                all_cands.append((d, c))
        scored = []
        for d, c in all_cands:
            c_idx = enc.get_index(c)
            if c_idx.sum() == 0: continue
            scored.append((d, c, hamming_distance(pred, c_idx)))
        scored.sort(key=lambda x: x[2])
        top3 = scored[:3]
        print(f"    {ent:14s} + country_direction -> ", end="")
        print(", ".join(f"[{d[:4]}:{c}]" for d, c, _ in top3))

    # Apply each direction to entities in EVERY domain
    print(f"\n  Cross-domain transform table (entity + direction -> top1 attribute):")
    print(f"  {'entity':14s} | " + " | ".join(f"{d:>16s}" for d in domain_deltas))
    print(f"  {'-'*14} | " + " | ".join("-"*16 for _ in domain_deltas))
    for domain, ent in test_entities.items():
        ctx = enc.get_context(ent)
        if ctx.sum() == 0:
            continue
        row = []
        for direction_domain in domain_deltas:
            direction = domain_deltas[direction_domain]
            pred = np.bitwise_xor(ctx, direction)
            # Restrict to that domain's candidates (fair comparison)
            cands = all_attribs_by_domain[direction_domain]
            best = None
            best_d = None
            for c in cands:
                c_idx = enc.get_index(c)
                if c_idx.sum() == 0: continue
                d = hamming_distance(pred, c_idx)
                if best_d is None or d < best_d:
                    best_d = d
                    best = c
            row.append(best[:15] if best else "?")
        print(f"  {ent:14s} | " + " | ".join(f"{r:>16s}" for r in row))

    # ── T4: the natural-language analogy test ──
    print("\n=== TEST 4: 'A is to B as C is to ???' natural analogies ===")
    print("  Classical analogy: take delta(A, B), apply to C, expect D.\n")
    analogies = [
        ("France",     "Paris",     "Lion",     "(habitat?)"),
        ("Germany",    "Berlin",    "Pizza",    "(origin?)"),
        ("Japan",      "Tokyo",     "Shakespeare","(work?)"),
        ("Italy",      "Rome",      "Tennis",   "(equipment?)"),
        ("Lion",       "Savannah",  "Germany",  "(capital?)"),
        ("Lion",       "Savannah",  "Pizza",    "(origin?)"),
        ("Pizza",      "Italy",     "Lion",     "(habitat?)"),
        ("Shakespeare","Hamlet",    "Mozart",   "(famous work? Mozart isn't in corpus)"),
    ]
    for a, b, c, hint in analogies:
        a_ctx = enc.get_context(a)
        b_idx = enc.get_index(b)
        c_ctx = enc.get_context(c)
        if a_ctx.sum() == 0 or b_idx.sum() == 0 or c_ctx.sum() == 0:
            print(f"  {a:14s}:{b:14s} :: {c:14s}:???  -- skip (no vector)")
            continue
        # one-shot delta from (a, b)
        delta = np.bitwise_xor(a_ctx, b_idx)
        # apply to c
        pred = np.bitwise_xor(c_ctx, delta)
        # Find nearest across ALL attributes (not restricted by domain)
        all_cands = [(d, c) for d, cs in all_attribs_by_domain.items() for c in cs]
        scored = []
        for d, c2 in all_cands:
            c_idx = enc.get_index(c2)
            if c_idx.sum() == 0: continue
            scored.append((d, c2, hamming_distance(pred, c_idx)))
        scored.sort(key=lambda x: x[2])
        top1 = scored[0]
        print(f"  {a:11s} : {b:11s} :: {c:11s} : {top1[1]:14s}  [{top1[0]:7s}]  d={top1[2]}  {hint}")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("  - If domain delta vectors are CLOSE in HD space (low pairwise Hamming),")
    print("    HDC has emergent meta-structure: 'primary feature' is a real direction")
    print("    that transcends specific relations.")
    print("  - If cross-domain analogies produce TOPICALLY RELATED answers")
    print("    (e.g., Lion + country_direction -> a habitat-like thing), the system")
    print("    has learned generalizable transformation geometry.")
    print("  - If both fail and everything's random, HDC space respects domain")
    print("    boundaries — domains live in separate corners.")


if __name__ == "__main__":
    main()
