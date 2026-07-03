"""
lattice/test_10domain.py - scale cross-domain analogy to 10 domains.

Tests:
  T1: within-domain retrieval — 100% across all 10 domains?
  T2: domain delta similarity — universal HAS_PRIMARY direction stable?
  T3: cross-domain analogies — 100% across many domain pairs?
  T4: dual-relation test — do HAS_PRIMARY and HAS_CATEGORY remain
      separable and both transferable?

If this scales cleanly, HDC's algebraic transfer is essentially
unbounded by domain count.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_TELP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TELP_ROOT))

from lattice.relation_indexing import RelationBoundEncoder
from lattice.corpus_10domain import (
    build_10domain_corpus, PRIMARY_PAIRS_BY_DOMAIN, PRIMARIES_BY_DOMAIN,
    HELDOUT, DOMAINS,
)
from train.v5_hdc_prototype import D, hamming_distance


def main():
    print("CROSS-DOMAIN ANALOGY @ 10 DOMAINS\n")

    print("Training RBI on 10-domain unified corpus ...")
    enc = RelationBoundEncoder(dim=D, seed=42)
    enc.train(build_10domain_corpus())
    print(f"  vocab: {len(enc.index_vectors)}, relations: {len(enc.role_vectors)}, "
          f"triples: {enc.triple_count}\n")

    # ── T1: within-domain ──
    print("=== TEST 1: within-domain (HAS_PRIMARY) ===")
    all_train_correct = 0
    all_train_total = 0
    all_heldout_correct = 0
    for domain, pairs in PRIMARY_PAIRS_BY_DOMAIN.items():
        cands = PRIMARIES_BY_DOMAIN[domain]
        correct = 0
        for ent, attr in pairs:
            results = enc.query(ent, "HAS_PRIMARY", cands, top_k=1)
            if results and results[0][0] == attr:
                correct += 1
        ent_h, attr_h = HELDOUT[domain]
        results = enc.query(ent_h, "HAS_PRIMARY", cands, top_k=1)
        heldout_ok = (results and results[0][0] == attr_h)
        all_train_correct += correct
        all_train_total += len(pairs)
        all_heldout_correct += int(heldout_ok)
        print(f"  {domain:10s}: train {correct}/{len(pairs)}  "
              f"heldout {ent_h}->{attr_h}: {'OK' if heldout_ok else 'MISS'}")
    print(f"\n  TOTAL: train {all_train_correct}/{all_train_total} = "
          f"{all_train_correct/all_train_total*100:.1f}%, "
          f"heldout {all_heldout_correct}/{len(DOMAINS)}")

    # ── T2: domain delta similarity ──
    print("\n=== TEST 2: are derived HAS_PRIMARY directions still identical? ===")
    explicit = enc._role("HAS_PRIMARY")
    domain_deltas = {}
    for domain, pairs in PRIMARY_PAIRS_BY_DOMAIN.items():
        deltas = []
        for ent, attr in pairs:
            ctx = enc.get_context(ent)
            attr_idx = enc.get_index(attr)
            if ctx.sum() and attr_idx.sum():
                deltas.append(np.bitwise_xor(ctx, attr_idx))
        if not deltas:
            continue
        stack = np.stack(deltas)
        avg = (stack.sum(axis=0) > len(deltas)/2).astype(np.int8)
        domain_deltas[domain] = avg

    n_perfect = 0
    print(f"  Each domain's averaged delta vs explicit HAS_PRIMARY role:")
    for domain, avg in domain_deltas.items():
        d = hamming_distance(avg, explicit)
        if d == 0: n_perfect += 1
        print(f"    {domain:10s}: {d} bits diff ({d/D*100:.2f}%)")
    print(f"  Domains with EXACT 0-bit match to role: {n_perfect}/{len(domain_deltas)}")

    # Pairwise distance among domain deltas
    print("\n  Pairwise distance between domain deltas (should be 0% if identical):")
    domains_list = list(domain_deltas.keys())
    max_pair_dist = 0
    for i, d1 in enumerate(domains_list):
        for d2 in domains_list[i+1:]:
            d = hamming_distance(domain_deltas[d1], domain_deltas[d2])
            max_pair_dist = max(max_pair_dist, d)
    print(f"    max pairwise distance: {max_pair_dist} bits = {max_pair_dist/D*100:.2f}%")

    # ── T3: cross-domain analogies — many pairs ──
    print("\n=== TEST 3: cross-domain analogies (one-shot, 10x10 = up to 90 pairs) ===")
    domain_anchor = {}   # one (entity, attr) anchor per domain
    for d, pairs in PRIMARY_PAIRS_BY_DOMAIN.items():
        domain_anchor[d] = pairs[0]   # use first pair as anchor
    # Targets: one entity per domain, with its known expected attr
    domain_target = {}
    for d in DOMAINS:
        # Use a held-out OR mid-training entity
        domain_target[d] = HELDOUT[d]   # (ent, expected_attr)

    all_attribs = []
    for d, attrs in PRIMARIES_BY_DOMAIN.items():
        for a in attrs:
            all_attribs.append((d, a))

    correct = 0
    total = 0
    failures = []
    print(f"  Testing {len(DOMAINS)} anchors x {len(DOMAINS)} targets = "
          f"{len(DOMAINS)**2 - len(DOMAINS)} cross-domain pairs ...")
    for anchor_domain, (anchor_ent, anchor_attr) in domain_anchor.items():
        for target_domain, (target_ent, expected) in domain_target.items():
            if anchor_domain == target_domain:
                continue
            a_ctx = enc.get_context(anchor_ent)
            a_idx = enc.get_index(anchor_attr)
            t_ctx = enc.get_context(target_ent)
            if any(v.sum() == 0 for v in [a_ctx, a_idx, t_ctx]):
                continue
            delta = np.bitwise_xor(a_ctx, a_idx)
            pred = np.bitwise_xor(t_ctx, delta)
            # Find nearest across ALL attribs
            scored = []
            for d, attr in all_attribs:
                attr_idx = enc.get_index(attr)
                if attr_idx.sum() == 0: continue
                scored.append((d, attr, hamming_distance(pred, attr_idx)))
            scored.sort(key=lambda x: x[2])
            top = scored[0]
            ok = (top[1] == expected)
            total += 1
            if ok:
                correct += 1
            else:
                failures.append((anchor_domain, anchor_ent, anchor_attr,
                                  target_domain, target_ent, expected, top[1]))
    print(f"\n  Cross-domain accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    if failures:
        print(f"\n  First 5 failures:")
        for f in failures[:5]:
            print(f"    {f[1]}:{f[2]} :: {f[4]}:??? -> got {f[6]} (expected {f[5]})")

    # ── T4: dual-relation test ──
    print("\n=== TEST 4: dual-relation transfer (HAS_PRIMARY + HAS_CATEGORY) ===")
    cat_explicit = enc._role("HAS_CATEGORY")
    prim_role_dist = hamming_distance(explicit, cat_explicit)
    print(f"  Explicit HAS_PRIMARY vs HAS_CATEGORY role distance: "
          f"{prim_role_dist} bits ({prim_role_dist/D*100:.1f}%)")
    print(f"  (Should be ~50% — they're orthogonal random vectors)")

    # Compute category direction from each domain
    cat_correct = 0
    cat_total = 0
    print(f"\n  Cross-domain HAS_CATEGORY analogies:")
    for anchor_domain in DOMAINS:
        data = DOMAINS[anchor_domain][0]
        # Build category pairs for this domain
        cat_pairs_anchor = [(e[0], e[2]) for e in data[:-1]]
        if not cat_pairs_anchor: continue
        anchor_ent, anchor_cat = cat_pairs_anchor[0]
        a_ctx = enc.get_context(anchor_ent)
        a_idx = enc.get_index(anchor_cat)
        if a_ctx.sum() == 0 or a_idx.sum() == 0: continue
        cat_delta = np.bitwise_xor(a_ctx, a_idx)
        # Test on other domains
        for target_domain in DOMAINS:
            if target_domain == anchor_domain: continue
            target_data = DOMAINS[target_domain][0]
            target_ent = target_data[-1][0]   # heldout
            expected_cat = target_data[-1][2]
            t_ctx = enc.get_context(target_ent)
            if t_ctx.sum() == 0: continue
            pred = np.bitwise_xor(t_ctx, cat_delta)
            # All categories from all domains
            all_cats = []
            for d, dat in DOMAINS.items():
                for e in dat[0]:
                    all_cats.append((d, e[2]))
            scored = []
            for d, cat in all_cats:
                cat_idx = enc.get_index(cat)
                if cat_idx.sum() == 0: continue
                scored.append((d, cat, hamming_distance(pred, cat_idx)))
            scored.sort(key=lambda x: x[2])
            top = scored[0]
            ok = (top[1] == expected_cat)
            cat_total += 1
            if ok: cat_correct += 1
    print(f"\n  Cross-domain category accuracy: {cat_correct}/{cat_total} = "
          f"{cat_correct/cat_total*100:.1f}%")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SCALE TEST VERDICT")
    print("=" * 70)
    print(f"  Within-domain (10 domains):              "
          f"{all_train_correct}/{all_train_total} = {all_train_correct/all_train_total*100:.0f}%")
    print(f"  Heldout (10 domains):                    "
          f"{all_heldout_correct}/{len(DOMAINS)}")
    print(f"  HAS_PRIMARY domain deltas identical:     "
          f"{n_perfect}/{len(domain_deltas)} domains")
    print(f"  Cross-domain HAS_PRIMARY analogies:      "
          f"{correct}/{total} = {correct/total*100:.0f}%")
    print(f"  Cross-domain HAS_CATEGORY analogies:     "
          f"{cat_correct}/{cat_total} = {cat_correct/cat_total*100:.0f}%")


if __name__ == "__main__":
    main()
